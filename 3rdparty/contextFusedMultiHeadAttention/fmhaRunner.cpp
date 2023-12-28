/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fmhaRunner.h"
#include "fused_multihead_attention_v2.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

namespace tensorrt_llm
{
namespace kernels
{

union __half2_uint32_t_union
{
    half2 fp162;
    uint32_t u32;
};

union __float_uint32_t_union
{
    float fp32;
    uint32_t u32;
};

static inline void set_alpha(uint32_t& alpha, float norm, Data_type dtype)
{
    if (dtype == DATA_TYPE_FP16)
    {
        __half2_uint32_t_union temp;
        temp.fp162 = __float2half2_rn(norm);
        alpha = temp.u32;
    }
    else if (dtype == DATA_TYPE_FP32)
    {
        __float_uint32_t_union temp;
        temp.fp32 = norm;
        alpha = temp.u32;
    }
    else if (dtype == DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha = reinterpret_cast<const uint32_t&>(inorm);
    }
    else if (dtype == DATA_TYPE_BF16)
    {
        // TODO HACK!! BF16 Outputs are computed in FP32 for FP8.
        // This is because cublas does not allow current FP32 output.
        alpha = reinterpret_cast<const uint32_t&>(norm);
    }
    else
    {
        assert(false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

class FusedMHARunnerV2::mhaImpl
{
public:
    mhaImpl(const Data_type data_type, const int numHeads, const int headSize, const float qScaling, int sm_)
        : mDataType(data_type)
        , mNumHeads(numHeads)
        , mHeadSize(headSize)
        , mQScaling(qScaling)
        , sm(sm_)
    {
        FT_CHECK_WITH_INFO(
            (sm == kSM_80 || sm == kSM_86 || sm == kSM_89 || sm == kSM_90), "Unsupported architecture");
        FT_CHECK_WITH_INFO((mDataType == DATA_TYPE_FP16 || mDataType == DATA_TYPE_BF16), "Unsupported data type");

        pagedKVXmmaKernel = getPagedKVXMMAKernelsV2(mDataType, sm);
        xmmaKernel = getXMMAKernelsV2(mDataType, sm);

        params.clear();
        pagedKVParams.clear();

        // get device attributes
        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(&launch_params.multi_processor_count, cudaDevAttrMultiProcessorCount, device_id);
        cudaDeviceGetAttribute(&launch_params.device_l2_cache_size, cudaDevAttrL2CacheSize, device_id);
    }

    ~mhaImpl() {}

    // Support packed QKV.
    void setup(const int b, const int s, const int sliding_window_size, const int total_seqlen, const bool has_alibi,
        const bool scale_alibi, const int tp_size, const int tp_rank)
    {
        const float inv_sqrt_scale = (1.f / (sqrtf(mHeadSize) * mQScaling));
        // Note that we apply scales and bias in the order of
        // (bmm1_output * scale_bmm1 + alibi) * scale_after_alibi
        const float scale_after_alibi = scale_alibi ? inv_sqrt_scale : 1.0f;
        const float scale_bmm1 = scale_alibi ? 1.0f : inv_sqrt_scale;
        const float scale_softmax = 1.f; // Seems to be only required for int8
        const float scale_bmm2 = 1.f;

        Data_type scale_type = launch_params.force_fp32_acc ? DATA_TYPE_FP32 : mDataType;
        set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
        set_alpha(params.scale_softmax, scale_softmax, scale_type);
        set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

        params.b = b;
        params.h = mNumHeads;
        params.s = s;
        params.d = mHeadSize;

        // Total sequence length needed by TMA descriptor
        // it should be actual total seq length if non-padded input is given.
        mTotalSeqLen = total_seqlen;

        params.qkv_stride_in_bytes = (mNumHeads + 2 * params.h_kv) * mHeadSize * sizeof(half);
        params.o_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);

        // Hopper: fallback to original fmha_v2 when head_size <= 64 and seq_len <= 256
        const bool isSm90 = (sm == kSM_90);
        const bool isSm8x = (sm == kSM_86 || sm == kSM_89);
        const bool isSm80 = (sm == kSM_80);
        if (isSm90 && params.d <= 64 && params.s <= 256)
        {
            launch_params.flash_attention = false;
            // get max sequence length for non-flash-attentio
            launch_params.kernel_s = getSFromMaxSeqLen(params.s);
        }
        else
        { // always use flash attention kernels for Ampere/Ada
            launch_params.flash_attention = true;
            // flash attention kernles s = 0 (support any seq length)
            launch_params.kernel_s = 0;
            launch_params.force_unroll = true;
            // enable tiled kernels on Ampere/Ada
            if (launch_params.flash_attention && params.s <= 64)
            {
                // flash attention tiled kernels allows larger free dim tile size (M, N) with flexibility
                // in unroll dimension tile size (K). for short sequence length (s<=128), tiled kernels
                // can suffer from tile quantization loss therefore use flash attention non-tiled instead
                launch_params.granular_tiling = false;
            }
            else if (isSm8x && params.d < 256)
            {
                // flash attention tiled kernel is faster on Ada and Ampere derivatives when head_size>=256
                launch_params.granular_tiling = false;
            }
            else if (isSm80 || isSm8x)
            {
                // otherwise, choose tiled kernel for Ampere/Ada
                launch_params.granular_tiling = true;
            }
        }

        // when flash attention is enabled on Hopper, we need to set the tma descriptors
        if (isSm90 && launch_params.flash_attention)
        {
            launch_params.warp_specialization = true;
            launch_params.use_tma = true;
        }

        // alibi.
        if (has_alibi)
        {
            params.has_alibi = true;
            params.alibi_params = AlibiParams(mNumHeads, s, tp_size, tp_rank, scale_after_alibi);
        }

        // Sliding_window_causal mask.
        if (s > sliding_window_size && launch_params.attention_mask_type == ContextAttentionMaskType::CAUSAL)
        {
            params.sliding_window_size = sliding_window_size;
            launch_params.attention_mask_type = ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL;
        }
    }

    // Support paged_kv_cache and chunked_attention.
    void setup_paged_kv(const int b, const int s_q, const int s_kv, const int blocks_per_context_sequence,
        const int tokens_per_kv_block, const int sliding_window_size, const int total_seqlen, const bool has_alibi,
        const bool scale_alibi, const int tp_size, const int tp_rank)
    {
        const float inv_sqrt_scale = (1.f / (sqrtf(mHeadSize) * mQScaling));
        // Note that we apply scales and bias in the order of
        // (bmm1_output * scale_bmm1 + alibi) * scale_after_alibi
        const float scale_after_alibi = scale_alibi ? inv_sqrt_scale : 1.0f;
        const float scale_bmm1 = scale_alibi ? 1.0f : inv_sqrt_scale;
        const float scale_softmax = 1.f; // Seems to be only required for int8
        const float scale_bmm2 = 1.f;

        Data_type scale_type = launch_params.force_fp32_acc ? DATA_TYPE_FP32 : mDataType;
        set_alpha(pagedKVParams.scale_bmm1, scale_bmm1, scale_type);
        set_alpha(pagedKVParams.scale_softmax, scale_softmax, scale_type);
        set_alpha(pagedKVParams.scale_bmm2, scale_bmm2, scale_type);

        pagedKVParams.b = b;
        pagedKVParams.h = mNumHeads;
        pagedKVParams.s = s_q;
        pagedKVParams.d = mHeadSize;

        // Total sequence length needed by TMA descriptor
        // it should be actual total seq length if non-padded input is given.
        mTotalSeqLen = total_seqlen;

        FT_CHECK_WITH_INFO(tokens_per_kv_block >= 128, "FMHA with paged kv cache needs tokens_per_block >= 128 !");
        // Needed by TMA descriptors.
        launch_params.blocks_per_context_sequence = blocks_per_context_sequence;
        pagedKVParams.q_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);
        pagedKVParams.kv_stride_in_bytes = tokens_per_kv_block * mHeadSize * sizeof(half);
        pagedKVParams.o_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);

        // Hopper: fallback to original fmha_v2 when head_size <= 64 and seq_len <= 256
        const bool isSm90 = (sm == kSM_90);
        const bool isSm8x = (sm == kSM_86 || sm == kSM_89);
        const bool isSm80 = (sm == kSM_80);

        // always use flash attention kernels.
        launch_params.flash_attention = true;
        // flash attention kernles s = 0 (support any seq length)
        launch_params.kernel_s = 0;
        launch_params.kernel_kv_s = s_kv;
        launch_params.force_unroll = true;

        // enable warp-specialization kernels when s > 512.
        if (isSm90 && s_kv > 512)
        {
            launch_params.warp_specialization = true;
            launch_params.use_tma = true;
        }
        else
        {
            // enable tiled kernels on Ampere/Ada
            if (launch_params.flash_attention && s_kv <= 64)
            {
                // flash attention tiled kernels allows larger free dim tile size (M, N) with flexibility
                // in unroll dimension tile size (K). for short sequence length (s<=128), tiled kernels
                // can suffer from tile quantization loss therefore use flash attention non-tiled instead
                launch_params.granular_tiling = false;
            }
            else if (isSm8x && params.d < 256)
            {
                // flash attention tiled kernel is faster on Ada and Ampere derivatives when head_size>=256
                launch_params.granular_tiling = false;
            }
            else if (isSm90 || isSm80 || isSm8x)
            {
                // otherwise, choose tiled kernel for Ampere/Ada
                launch_params.granular_tiling = true;
            }
        }

        // alibi.
        if (has_alibi)
        {
            pagedKVParams.has_alibi = true;
            pagedKVParams.alibi_params = AlibiParams(mNumHeads, s_kv, tp_size, tp_rank, scale_after_alibi);
        }

        // Sliding_window_causal mask.
        if (s_kv > sliding_window_size && launch_params.attention_mask_type == ContextAttentionMaskType::CAUSAL)
        {
            pagedKVParams.sliding_window_size = sliding_window_size;
            launch_params.attention_mask_type = ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL;
        }
    }

    // NOTE: assume that heads_interleaved = false (b, s, 3, h, d), and sequences are padded/non-padded
    // TMA descriptors are used as grid_constant parameters (remove MemCpyH2D operations)
    void set_tma_descriptors()
    {
        // split D into multiple groups in order to match the TMA swizzle mode (128B)
        const uint32_t d_in_bytes = params.d * sizeof(uint16_t);
        const uint32_t d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

        // separate q, k, and v tma descriptors
        Multiple_tma_descriptor<4> qkv_tma_descriptor;

        // tensor size
        uint32_t tensor_size_qkv[4];
        if (params.h_kv < params.h)
        {
            // if multi-query or grouped-query
            tensor_size_qkv[2] = 1;
            tensor_size_qkv[1] = (params.h + 2 * params.h_kv);
            tensor_size_qkv[0] = params.d; // params.d;
        }
        else
        {
            tensor_size_qkv[2] = 3;
            tensor_size_qkv[1] = params.h;
            tensor_size_qkv[0] = params.d; // params.d;
        }

        // box size for k and v
        uint32_t box_size[4];
        // Update this on device?
        box_size[2] = 1;
        box_size[1] = 1;
        box_size[0] = params.d / d_groups;

        // stride size in bytes. Assumes least significant dim is 1 (?)
        uint64_t tensor_stride_qkv[3];
        tensor_stride_qkv[0] = tensor_size_qkv[0] * sizeof(uint16_t);     // d
        tensor_stride_qkv[1] = tensor_size_qkv[1] * tensor_stride_qkv[0]; // d*h
        tensor_stride_qkv[2] = tensor_size_qkv[2] * tensor_stride_qkv[1]; // d*h*3

        // traversal stride
        uint32_t traversal_stride_qkv[4] = {1, 1, 1, 1};

        // OOB fill zeros
        uint32_t oob_fill = 0;

        // FP32 to TF32 conversion disabled
        uint32_t fp32_to_tf32 = 0;

        // gmma descriptor mode
        const uint32_t d_bytes_per_group = (params.d * sizeof(uint16_t)) / d_groups;
        const cudaTmaDescSwizzle swizzle_mode = (d_bytes_per_group > 64
                ? cudaTmaDescSwizzle::SWIZZLE_128B
                : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

        uint32_t q_step = 0, kv_step = 0;
        for (unsigned int i = 0u; i < sizeof(sTmaMetaInfo) / sizeof(sTmaMetaInfo[0]); ++i)
        {
            if (sTmaMetaInfo[i].mD == params.d)
            {
                q_step = sTmaMetaInfo[i].mQStep;
                kv_step = sTmaMetaInfo[i].mKVStep;
                break;
            }
        }

        // QKV [TOTAL, 3, h, d]
        // NOTE: we may need to use actual seqlen to set oob_value
        const char* qkv_ptr = reinterpret_cast<const char*>(params.qkv_ptr);
        tensor_size_qkv[3] = mTotalSeqLen;

        // Q: STEP_Q
        box_size[3] = q_step;
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_qkv, tensor_stride_qkv, traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32,
            &params.tma_desc_q);

        // K/V: STEP_KV
        box_size[3] = kv_step;
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_qkv, tensor_stride_qkv, traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32,
            &params.tma_desc_k);
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_qkv, tensor_stride_qkv, traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32,
            &params.tma_desc_v);
    }

    // Q are contiguous in the shape of [B, S, H, D]
    // Paged KV has [B, 2, NumBlocksPerSequence] buffers,
    //  and each points to the contiguous buffer with shape [H, TokensPerBlock, D]
    // TMA descriptors need cudaMemcpyAsync since we need multiple tma descriptors in device memory.
    void set_paged_kv_tma_descriptors(cudaStream_t stream)
    {
        // split D into multiple groups in order to match the TMA swizzle mode (128B)
        const uint32_t d_in_bytes = pagedKVParams.d * sizeof(uint16_t);
        const uint32_t d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

        uint32_t q_step = 0, kv_step = 0;
        for (unsigned int i = 0u; i < sizeof(sTmaPagedKVMetaInfo) / sizeof(sTmaPagedKVMetaInfo[0]); ++i)
        {
            if (sTmaPagedKVMetaInfo[i].mD == pagedKVParams.d)
            {
                q_step = sTmaPagedKVMetaInfo[i].mQStep;
                kv_step = sTmaPagedKVMetaInfo[i].mKVStep;
                break;
            }
        }

        // Separate q, and paged kv tma descriptors.
        Multiple_tma_descriptor<4> q_tma_descriptor;
        Multiple_tma_descriptor<4> paged_kv_tma_descriptor(
            pagedKVParams.b * 2 * launch_params.blocks_per_context_sequence);
        // Contiguous Q
        // query tensor size [B x S, 1, H, D]
        uint32_t tensor_size_q[4];
        tensor_size_q[3] = mTotalSeqLen;
        tensor_size_q[2] = 1;
        tensor_size_q[1] = pagedKVParams.h;
        tensor_size_q[0] = pagedKVParams.d;

        // box size for k and v
        uint32_t box_size_q[4];
        box_size_q[3] = q_step;
        box_size_q[2] = 1;
        box_size_q[1] = 1;
        box_size_q[0] = pagedKVParams.d / d_groups;

        // stride size in bytes.
        uint64_t tensor_stride_q[3];
        tensor_stride_q[0] = tensor_size_q[0] * sizeof(uint16_t);
        tensor_stride_q[1] = tensor_size_q[1] * tensor_stride_q[0];
        tensor_stride_q[2] = tensor_size_q[2] * tensor_stride_q[1];

        // traversal stride
        uint32_t traversal_stride[4] = {1, 1, 1, 1};

        // OOB fill zeros
        uint32_t oob_fill = 0;

        // FP32 to TF32 conversion disabled
        uint32_t fp32_to_tf32 = 0;

        // gmma descriptor mode
        const uint32_t d_bytes_per_group = (pagedKVParams.d * sizeof(uint16_t)) / d_groups;
        const cudaTmaDescSwizzle swizzle_mode = (d_bytes_per_group > 64
                ? cudaTmaDescSwizzle::SWIZZLE_128B
                : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

        // Q ptr.
        const char* q_ptr = reinterpret_cast<const char*>(pagedKVParams.q_ptr);

        // Q: STEP_Q.
        q_tma_descriptor.set_tma_desctriptor(q_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_q, tensor_stride_q, traversal_stride, box_size_q, oob_fill, fp32_to_tf32,
            &pagedKVParams.tma_desc_q);

        // Paged KV
        // Per batch tensor size.
        uint32_t tensor_size_kv[4];
        tensor_size_kv[3] = 1;
        tensor_size_kv[2] = pagedKVParams.h_kv;
        tensor_size_kv[1] = pagedKVParams.paged_kv_cache.mTokensPerBlock;
        tensor_size_kv[0] = pagedKVParams.d;

        // Box size for k and v.
        uint32_t box_size_kv[4];
        box_size_kv[3] = 1;
        box_size_kv[2] = 1;
        box_size_kv[1] = kv_step;
        box_size_kv[0] = pagedKVParams.d / d_groups;

        // Stride size in bytes.
        uint64_t tensor_stride_kv[3];
        tensor_stride_kv[0] = tensor_size_kv[0] * sizeof(uint16_t);
        tensor_stride_kv[1] = tensor_size_kv[1] * tensor_stride_kv[0];
        tensor_stride_kv[2] = tensor_size_kv[2] * tensor_stride_kv[1];

        // 2 stands for k, and v blocks.
        // We only need to prepare as many tma descriptos as the number of paged kv blocks for context.
        for (int block_idx = 0; block_idx < pagedKVParams.b * 2 * launch_params.blocks_per_context_sequence;
             block_idx++)
        {
            int block_ptr_idx = int(block_idx / launch_params.blocks_per_context_sequence)
                    * pagedKVParams.paged_kv_cache.mMaxBlocksPerSeq
                + (block_idx % launch_params.blocks_per_context_sequence);
            paged_kv_tma_descriptor.set_tma_desctriptor(
                reinterpret_cast<char*>(launch_params.paged_kv_block_ptrs[block_ptr_idx]), cudaTmaDescFormat::F16_RN,
                cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
                tensor_size_kv, tensor_stride_kv, traversal_stride, box_size_kv, oob_fill, fp32_to_tf32, block_idx);
        }

        // set mMaxBlocksPerSeq to the number of blocks needed for context.
        pagedKVParams.paged_kv_cache.mMaxBlocksPerSeq = launch_params.blocks_per_context_sequence;

        paged_kv_tma_descriptor.copy_to_device(pagedKVParams.tma_desc_paged_kv, stream);
    }

    void setup_flags(const bool force_fp32_acc, const bool is_s_padded, const bool causal_mask, const int num_kv_heads)
    {
        // BF16 FMHA only accumulates on FP32
        launch_params.force_fp32_acc = mDataType == DATA_TYPE_BF16 || force_fp32_acc;
        launch_params.attention_mask_type
            = causal_mask ? ContextAttentionMaskType::CAUSAL : ContextAttentionMaskType::PADDING;

        // Paged KV Cache.
        pagedKVParams.h_kv = num_kv_heads;
        FT_CHECK_WITH_INFO(mNumHeads % num_kv_heads == 0, "number of Query heads should be multiple of KV heads !");
        pagedKVParams.h_q_per_kv = mNumHeads / num_kv_heads;
        pagedKVParams.is_s_padded = is_s_padded;

        // Contiguous Cache.
        params.h_kv = num_kv_heads;
        params.is_s_padded = is_s_padded;
    }

    bool fmha_supported()
    {
        return MHARunner::fmha_supported(mHeadSize, sm);
    }

    void run(const void* qkvPtr, const void* cuSeqlenPtr, void* outputPtr, cudaStream_t stream)
    {
        params.qkv_ptr = qkvPtr;
        params.o_ptr = outputPtr;
        params.cu_seqlens = reinterpret_cast<const int*>(cuSeqlenPtr);

        if (sm == kSM_90 && launch_params.use_tma)
        {
            // memcpy H2D has been removed by applying grid_constant tma descriptors.
            set_tma_descriptors();
        }

        xmmaKernel->run(params, launch_params, stream);
    }

    void run_paged_kv(const void* qPtr, void* pagedKVTmaDesc, const void* pagedKVBlockPtrsOnHost,
        const fastertransformer::KVBlockArray pagedKVCache, const void* cuQSeqlenPtr, const void* cuKVSeqlenPtr, void* outputPtr,
        cudaStream_t stream)
    {
        pagedKVParams.q_ptr = qPtr;
        pagedKVParams.tma_desc_paged_kv = reinterpret_cast<cudaTmaDesc*>(pagedKVTmaDesc);
        pagedKVParams.paged_kv_cache = pagedKVCache;
        pagedKVParams.o_ptr = outputPtr;
        pagedKVParams.cu_q_seqlens = reinterpret_cast<const int*>(cuQSeqlenPtr);
        pagedKVParams.cu_seqlens = reinterpret_cast<const int*>(cuKVSeqlenPtr);
        // paged kv block device ptrs on host (used by tma descriptors).
        launch_params.paged_kv_block_ptrs = reinterpret_cast<const int64_t*>(pagedKVBlockPtrsOnHost);

        if (sm == kSM_90 && launch_params.use_tma)
        {
            // memcpy H2D is needed as we use multiple tma descriptors in device memory.
            set_paged_kv_tma_descriptors(stream);
        }

        pagedKVXmmaKernel->run(pagedKVParams, launch_params, stream);
    }

    bool isValid(int s) const
    {
        return pagedKVXmmaKernel->isValid(s) && xmmaKernel->isValid(s);
    }

    int getSFromMaxSeqLen(const int max_seq_len)
    {
        int S = 1024;

        if (max_seq_len <= 64)
        {
            S = 64;
        }
        else if (max_seq_len <= 128)
        {
            S = 128;
        }
        else if (max_seq_len <= 256)
        {
            S = 256;
        }
        else if (max_seq_len <= 384)
        {
            S = 384;
        }
        else if (max_seq_len <= 512)
        {
            S = 512;
        }
        // for bert and vit, use flash attention when s >= 512
        else if (max_seq_len > 512)
        {
            S = max_seq_len;
        }

        return S;
    }

private:
    Fused_multihead_attention_params_v2 params;
    Fused_multihead_attention_paged_kv_params_v2 pagedKVParams;
    Launch_params launch_params;
    int sm;
    const FusedMultiHeadAttentionXMMAKernelV2* xmmaKernel;
    const FusedMultiHeadAttentionPagedKVXMMAKernelV2* pagedKVXmmaKernel;
    bool use_flash_attention = false;
    const Data_type mDataType;
    const int mNumHeads;
    const int mHeadSize;
    const float mQScaling;
    int mTotalSeqLen;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

FusedMHARunnerV2::FusedMHARunnerV2(
    const Data_type data_type, const int numHeads, const int headSize, const float qScaling)
    : pimpl(new mhaImpl(data_type, numHeads, headSize, qScaling, fastertransformer::getSMVersion()))
{
}

FusedMHARunnerV2::~FusedMHARunnerV2() = default;

void FusedMHARunnerV2::setup(const int b, const int s, const int sliding_window_size, const int total_seqlen,
    const bool has_alibi, const bool scale_alibi, const int tp_size, const int tp_rank)
{
    pimpl->setup(b, s, sliding_window_size, total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
}

void FusedMHARunnerV2::setup_paged_kv(const int b, const int s_q, const int s_kv, const int blocks_per_context_sequence,
    const int tokens_per_kv_block, const int sliding_window_size, const int total_seqlen, const bool has_alibi,
    const bool scale_alibi, const int tp_size, const int tp_rank)
{
    pimpl->setup_paged_kv(b, s_q, s_kv, blocks_per_context_sequence, tokens_per_kv_block, sliding_window_size,
        total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
}

bool FusedMHARunnerV2::fmha_supported()
{
    return pimpl->fmha_supported();
}

void FusedMHARunnerV2::setup_flags(
    const bool force_fp32_acc, const bool is_s_padded, const bool causal_mask, const int num_kv_heads)
{
    pimpl->setup_flags(force_fp32_acc, is_s_padded, causal_mask, num_kv_heads);
}

void FusedMHARunnerV2::run(const void* qkvPtr, const void* cuSeqlenPtr, void* outputPtr, cudaStream_t stream)
{
    pimpl->run(qkvPtr, cuSeqlenPtr, outputPtr, stream);
}

void FusedMHARunnerV2::run_paged_kv(const void* qPtr, void* pagedKVTmaDesc, const void* pagedKVBlockPtrsOnHost,
    const fastertransformer::KVBlockArray pagedKVCache, const void* cuQSeqlenPtr, const void* cuKVSeqlenPtr, void* outputPtr,
    cudaStream_t stream)
{
    pimpl->run_paged_kv(
        qPtr, pagedKVTmaDesc, pagedKVBlockPtrsOnHost, pagedKVCache, cuQSeqlenPtr, cuKVSeqlenPtr, outputPtr, stream);
}

bool FusedMHARunnerV2::isValid(int s) const
{
    return pimpl->isValid(s);
}

// static function to check if fmha is supported when building plugins
bool MHARunner::fmha_supported(const int headSize, const int sm)
{
    if (sm == kSM_80 || sm == kSM_86 || sm == kSM_89)
    {
        return (headSize == 16 || headSize == 32 || headSize == 40 || headSize == 64 || headSize == 80
            || headSize == 128 || headSize == 160 || headSize == 256);
    }
    else if (sm == kSM_90)
    {
        return (headSize == 32 || headSize == 64 || headSize == 128 || headSize == 256);
    }

    return false;
}

} // namespace kernels
} // namespace tensorrt_llm
