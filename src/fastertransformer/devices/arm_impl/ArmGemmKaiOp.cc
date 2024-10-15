#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include <cstring>
#include "autil/StringUtil.h"
#include "gemm_opt/ArmGemmKernel.h"
#include <cfloat>

#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p_bf16p12x1biasf32_8x12x4_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/matmul_clamp_f32_bf16p_bf16p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p4x12biasf32_f32_bf16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p8x4_bf16_neon.h"

#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"

namespace fastertransformer {

static inline size_t num_blocks_per_row(size_t k, size_t bl) {
    return k / bl;
}

static inline size_t num_bytes_per_block_qs4c32(size_t bl) {
    return (bl / 2) + sizeof(int16_t);
}

static void quant_qs4c32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32, uint8_t* rhs_qs4c32) {
    const size_t num_blocks_row = num_blocks_per_row(k, bl);
    const size_t num_bytes_block = num_bytes_per_block_qs4c32(bl);
    const size_t dst_stride = num_blocks_row * num_bytes_block;

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
        const float* src_ptr = rhs_f32 + row_idx * k;

        uint8_t* dst_ptr = (uint8_t*)rhs_qs4c32 + row_idx * dst_stride;

        for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
            float amax = 0.0f;
            float max = 0.0f;

            for (size_t b = 0; b < bl; ++b) {
                const float src0_0 = src_ptr[block_idx * bl + b];
                const float asrc0_0 = fabsf(src0_0);

                if (amax < asrc0_0) {
                    amax = asrc0_0;
                    max = src0_0;
                }
            }

            const float scale = max / -8.0;
            const float recip_scale = scale ? 1.0f / scale : 0.0f;

            // Store the scale at the beginning of the block
            *((uint16_t*)dst_ptr) = kai_cast_f16_f32(scale);
            dst_ptr += sizeof(uint16_t);

            const size_t block_size = 32;
            const size_t num_subblocks = bl / 32;

            for (size_t subblock_idx = 0; subblock_idx < num_subblocks; ++subblock_idx) {
                for (size_t i = 0; i < block_size / 2; ++i) {
                    const size_t src_base_addr = block_idx * bl + i + subblock_idx * block_size;
                    float v0_f32 = src_ptr[src_base_addr];
                    float v1_f32 = src_ptr[src_base_addr + block_size / 2];

                    v0_f32 *= recip_scale;
                    v1_f32 *= recip_scale;

                    const uint8_t v0_u8 = (uint8_t)std::min((int8_t)15, (int8_t)(v0_f32 + 8.5f));
                    const uint8_t v1_u8 = (uint8_t)std::min((int8_t)15, (int8_t)(v1_f32 + 8.5f));

                    const uint8_t rhs_v0 = (v1_u8 << 4) | v0_u8;

                    dst_ptr[0] = rhs_v0;
                    dst_ptr += sizeof(uint8_t);
                }
            }
        }
    }
}

BufferPtr ArmCpuDevice::gemm_kai_fp32(const GemmParams& params) {

    auto start = std::chrono::high_resolution_clock::now();

    params.check();

    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Dshape;

    size_t dim;
    size_t batch_size;
    size_t m;
    size_t k;
    size_t n;

    Ashape = params.A.shape();
    Bshape = params.B.shape();

    dim        = params.A.dim();
    batch_size = std::accumulate(Ashape.begin(), Ashape.end() - 2, (size_t)1, std::multiplies<size_t>());

    if (params.transA == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
    }

    if (params.transB == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
    }

    m = Ashape[dim - 2];
    k = Ashape[dim - 1];
    n = Bshape[dim - 1];

    auto data_type = params.compute_type == DataType::TYPE_INVALID ? params.A.type() : params.compute_type;
    if (data_type != params.A.type()) {
        std::cout << "[Warning] GEMM compute type differs from input type. Not supported" << std::endl;
        data_type = params.A.type();
    }


    Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
    Dshape.insert(Dshape.end(), {m, n});

    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((data_type == params.D->type()) && (Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              data_type,
                              autil::StringUtil::toString(Dshape).c_str(),
                              params.D->debugString().c_str());
    } else {
        output = allocateBuffer({data_type, Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }
    
    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t nr = kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla();
        const size_t kr = kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla();
        const size_t sr = kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla();

        // In a single row, we pack nr bias values followed by K rows of nr RHS values
        const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(n, k);
        // const size_t rhs_packed_cols = nr + k * nr;
        // const size_t rhs_packed_rows = rhs_packed_size / (rhs_packed_cols * sizeof(float));

        float* rhs_packed = new float[rhs_packed_size];

        const size_t bias_size = n;
        float* bias = new float[bias_size];
        memset(bias, 0, sizeof(float) * bias_size);

        const size_t lhs_stride = k * sizeof(float);
        const size_t rhs_stride = n * sizeof(float);
        const size_t dst_stride_row = n * sizeof(float);
        const size_t dst_stride_col = sizeof(float);

        float* rhs = (float* )params.B.data();
        float* lhs = (float* )params.A.data();

        // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be constant.
        kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(
            1, n, k, nr, kr, sr,  // Packing arguments
            rhs_stride,           // RHS stride
            rhs,                  // RHS
            bias,                 // Bias
            NULL,                 // Scale
            rhs_packed,           // RHS packed
            0, NULL);

        float* dst = (float* )output->data();

        kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(
                m, n, k,                  // Dimensions
                lhs,                      // LHS
                lhs_stride,               // LHS stride
                rhs_packed,               // RHS packed
                dst,                      // DST
                dst_stride_row,           // DST stride (row)
                dst_stride_col,           // DST stride (col)
                -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()  // Min and max for the clamp operation
        );

        // parallel_gemm_kai_fp32(m, n, k, lhs, rhs_packed, dst, lhs_stride, n, dst_stride_row);

        delete[] bias;
        delete[] rhs_packed;
    }

    auto end = std::chrono::high_resolution_clock::now();
    float during_time = std::chrono::duration<float>(end - start).count();
    printf("gemm_kai_fp32 m,n,k %ld %ld %ld %.3f\n", m, n, k, during_time * 1000);
    return output;
}

/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]
BufferPtr ArmCpuDevice::gemm_kai_bf16(const GemmParams& params, bool isRhsPacked) {

    auto start = std::chrono::high_resolution_clock::now();

    params.check();

    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Dshape;

    size_t dim;
    size_t m;
    size_t k;
    size_t n;

    Ashape = params.A.shape();
    Bshape = params.B.shape();

    dim        = params.A.dim();

    if (params.transA == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
    }

    if (params.transB == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
    }

    m = Ashape[dim - 2];
    k = Ashape[dim - 1];
    n = Bshape[dim - 1];

    auto data_type = params.compute_type == DataType::TYPE_INVALID ? params.A.type() : params.compute_type;
    if (data_type != params.A.type()) {
        std::cout << "[Warning] GEMM compute type differs from input type. Not supported" << std::endl;
        data_type = params.A.type();
    }

    Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
    Dshape.insert(Dshape.end(), {m, n});

    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((data_type == params.D->type()) && (Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              data_type,
                              autil::StringUtil::toString(Dshape).c_str(),
                              params.D->debugString().c_str());
    } else {
        output = allocateBuffer({data_type, Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }

    const size_t mr = kai_get_mr_matmul_clamp_f32_bf16p_bf16p12x1biasf32_8x12x4_neon_mmla();
    const size_t nr = kai_get_nr_matmul_clamp_f32_bf16p_bf16p12x1biasf32_8x12x4_neon_mmla();
    const size_t kr = kai_get_kr_matmul_clamp_f32_bf16p_bf16p12x1biasf32_8x12x4_neon_mmla();
    const size_t sr = kai_get_sr_matmul_clamp_f32_bf16p_bf16p12x1biasf32_8x12x4_neon_mmla();

    bfloat16_t* rhs_packed;
    float* bias;

    const size_t lhs_stride = k * sizeof(float);
    const size_t rhs_stride = n * sizeof(float);
    const size_t dst_stride_row = n * sizeof(float);
    const size_t dst_stride_col = sizeof(float);

    const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_f32p8x4_bf16_neon(m, k, mr, kr, sr);
    bfloat16_t *lhs_packed = new bfloat16_t[lhs_packed_size];

    // float* rhs_packed = (float* )params.B.data();
    float* rhs = (float* )params.B.data();
    float* lhs = (float* )params.A.data();

    // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be constant.
    int n_step = nr;
    if (!isRhsPacked) {
        const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p4x12biasf32_f32_bf16_neon(n, k);
        rhs_packed = new bfloat16_t[rhs_packed_size];

        const size_t bias_size = n;
        bias = new float[bias_size];
        memset(bias, 0, bias_size * sizeof(float));

        #pragma omp parallel for
        for (int n_start = 0; n_start < n; n_start += n_step) {
            const size_t rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f32p4x12biasf32_f32_bf16_neon(n_start);
            const size_t bias_offset = kai_get_bias_offset_rhs_pack_kxn_f32p4x12biasf32_f32_bf16_neon(n_start);
            const size_t packed_offset =kai_get_rhs_packed_offset_rhs_pack_kxn_f32p4x12biasf32_f32_bf16_neon(n_start, k);

            int tile_n = (n_start + n_step <= n) ? n_step : n - n_start;
            kai_run_rhs_pack_kxn_f32p4x12biasf32_f32_bf16_neon(
                1, tile_n, k, nr, kr, sr,  // Packing arguments
                rhs_stride,           // RHS stride
                ((uint8_t*)rhs + rhs_offset),                  // RHS
                ((uint8_t*)bias + bias_offset),                 // Bias
                NULL,                 // Scale
                ((uint8_t*)rhs_packed + packed_offset),           // RHS packed
                0, NULL);
        }
    } else {
        rhs_packed = (bfloat16_t* )params.B.data();
    }

    float* dst = (float* )output->data();

    int m_step = mr;
    #pragma omp parallel for
    for (int m_start = 0; m_start < m; m_start += m_step) {
        const size_t lhs_offset = kai_get_lhs_offset_lhs_pack_f32p8x4_bf16_neon(m_start, lhs_stride);
        const size_t lhs_packed_offset = kai_get_lhs_packed_offset_lhs_pack_f32p8x4_bf16_neon(m_start, k);
        int tile_m = (m_start + m_step <= m) ? m_step : m - m_start;

        kai_run_lhs_pack_f32p8x4_bf16_neon(
            tile_m, k, mr, kr, sr,
            0 /* m_idx_start; should stay as 0 */,
            ((uint8_t*)lhs + lhs_offset), // adjust Lhs start position
            lhs_stride,
            ((uint8_t*)lhs_packed + lhs_packed_offset));
    }

    #pragma omp parallel for
    for (int n_start = 0; n_start < n; n_start += n_step) {
        size_t lhs_offset = kai_get_lhs_offset_lhs_pack_f32p8x4_bf16_neon(0, k);
        size_t rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_f32p4x12biasf32_f32_bf16_neon(n_start, k);
        size_t dst_offset = kai_get_dst_offset_matmul_clamp_f32_bf16p_bf16p12x1biasf32_8x12x4_neon_mmla(0, n_start, n * sizeof(bfloat16_t));

        const void* lhs_ptr = (const void*)((const char *)lhs_packed + lhs_offset);
        const void* rhs_ptr = (const void*)((const char *)rhs_packed + rhs_offset);
        void* dst_ptr = (void*)((uint8_t*)dst + dst_offset);

        assert(n % n_step == 0);
        assert(n_step % n_step == 0);

        int tile_n = (n_start + n_step <= n) ? n_step : n - n_start;
        kai_run_matmul_clamp_f32_bf16p_bf16p12x1biasf32_8x12x4_neon_mmla(
            m, tile_n, k,                  // Dimensions
            lhs_ptr,                      // LHS
            rhs_ptr,               // RHS packed
            dst_ptr,                      // DST
            dst_stride_row,           // DST stride (row)
            dst_stride_col,           // DST stride (col)
            -FLT_MAX, FLT_MAX   // Min and max for the clamp operation
        );
    }

    delete[] lhs_packed;
    if (!isRhsPacked) {
        delete[] bias;
        delete[] rhs_packed;
    }

    auto end = std::chrono::high_resolution_clock::now();
    float during_time = std::chrono::duration<float>(end - start).count();
    printf("gemm_kai_bf16 m,n,k %ld %ld %ld %.3f\n", m, n, k, during_time * 1000);
    return output;
}

BufferPtr ArmCpuDevice::gemm_kai_a8w4_1x4(const GemmParams& params, bool isRhsPacked) {

    auto start = std::chrono::high_resolution_clock::now();
    params.check();

    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Dshape;

    size_t dim;
    size_t m;
    size_t k;
    size_t n;

    Ashape = params.A.shape();
    Bshape = params.B.shape();

    dim        = params.A.dim();

    if (params.transA == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
    }

    if (params.transB == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
    }

    m = Ashape[dim - 2];
    k = Ashape[dim - 1];
    n = Bshape[dim - 1];

    auto data_type = params.compute_type == DataType::TYPE_INVALID ? params.A.type() : params.compute_type;
    if (data_type != params.A.type()) {
        std::cout << "[Warning] GEMM compute type differs from input type. Not supported" << std::endl;
        data_type = params.A.type();
    }

    if (data_type != DataType::TYPE_FP32) {
        std::cout << "[Warning] GEMM data_type not TYPE_FP32. Not supported" << std::endl;
    }

    Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
    Dshape.insert(Dshape.end(), {m, n});

    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((data_type == params.D->type()) && (Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              data_type,
                              autil::StringUtil::toString(Dshape).c_str(),
                              params.D->debugString().c_str());
    } else {
        output = allocateBuffer({data_type, Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }

    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();

    const size_t lhs_stride = k * sizeof(float);
    const size_t rhs_stride = n * sizeof(float);
    const size_t dst_stride_row = n * sizeof(float);
    const size_t dst_stride_col = sizeof(float);

    const size_t bl = 32;
    const size_t num_blocks = k / bl;
    const size_t num_bytes_per_block_qs4c32 = (bl / 2) + sizeof(int16_t);
    const size_t rhs_native_size_qs4c32 = n * num_blocks * num_bytes_per_block_qs4c32;

    const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m, k, bl, mr, kr, sr);
    uint8_t* lhs_packed_mtx_qs8d32 = new uint8_t[lhs_packed_size];

    const size_t rhs_packed_size =
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n, k, nr, kr, bl);
    uint8_t* rhs_packed_mtx_qs4c32;
    uint8_t* rhs_native_mtx_qs4c32;
    float* lhs = (float* )params.A.data();
    
    // RHS packing
    int n_step = nr;
    if (isRhsPacked) {
        rhs_packed_mtx_qs4c32 = (uint8_t*)params.B.data();
    } else {
        float* rhs = (float* )params.B.data();
        rhs_packed_mtx_qs4c32 = new uint8_t[rhs_packed_size];
        rhs_native_mtx_qs4c32 = new uint8_t[rhs_native_size_qs4c32];

        quant_qs4c32_f32(
            n, k, bl, (const float*)rhs, (uint8_t*)rhs_native_mtx_qs4c32);

        struct kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0_params kai_rhs_params;
        kai_rhs_params.lhs_zero_point = 1;
        kai_rhs_params.rhs_zero_point = 8;

        #pragma omp parallel for
        for (int n_start = 0; n_start < n; n_start += n_step) {
            const size_t rhs_offset = kai_get_rhs_offset_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n_start, rhs_stride);
            const size_t packed_offset = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n_start, k, nr, kr, bl);

            int tile_n = (n_start + n_step <= n) ? n_step : n - n_start;

            kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
                1, tile_n, k,                                  // Dimensions
                nr, kr, sr,                               // Packing arguments
                bl,                                       // Block length
                (const uint8_t*)(rhs_native_mtx_qs4c32 + rhs_offset),  // RHS
                NULL,                                     // Bias
                ((uint8_t*)rhs_packed_mtx_qs4c32 + packed_offset),                    // RHS packed
                0, &kai_rhs_params);
        }
    }

    // LHS packing
    int m_step = mr;
    #pragma omp parallel for
    for (int m_start = 0; m_start < m; m_start += m_step) {
        const size_t lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32(m_start, lhs_stride);
        const size_t lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32(m_start, k, bl, mr, kr, sr);
        int tile_m = (m_start + m_step <= m) ? m_step : m - m_start;

        kai_run_lhs_quant_pack_qsi8d32p_f32(
            tile_m, k, bl, mr, kr, sr, 0,               // Packing arguments
            (const float*)((uint8_t*)lhs + lhs_offset),                  // LHS
            lhs_stride,                 // LHS stride
            ((uint8_t*)lhs_packed_mtx_qs8d32 + lhs_packed_offset));             // LHS packed
    }

    // Matmul
    #pragma omp parallel for
    for (int n_start = 0; n_start < n; n_start += n_step) {
        const size_t dst_stride = n * sizeof(float);
        const size_t lhs_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(0, k, bl);
        const size_t rhs_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(n_start, k, bl);
        const size_t dst_offset = kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(0, n_start, dst_stride); // FIXME with dst_stride?

        const void* lhs_ptr = (const void*)((const char *)lhs_packed_mtx_qs8d32 + lhs_offset);
        const void* rhs_ptr = (const void*)((const char *)rhs_packed_mtx_qs4c32 + rhs_offset);
        float* dst_ptr = (float*)((uint8_t*)output->data() + dst_offset);

        int tile_n = (n_start + n_step <= n) ? n_step : n - n_start;
        kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
            m, tile_n, k,           // Dimensions
            bl,
            lhs_ptr,           // LHS packed
            rhs_ptr,           // RHS packed
            dst_ptr,           // DST
            dst_stride_row,        // DST stride (row)
            dst_stride_col,     // DST stride (col)
            -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
        );
    }

    delete[] lhs_packed_mtx_qs8d32;
    if (!isRhsPacked) {
        delete[] rhs_native_mtx_qs4c32;
        delete[] rhs_packed_mtx_qs4c32;
    }
    auto end = std::chrono::high_resolution_clock::now();
    float during_time = std::chrono::duration<float>(end - start).count();
    printf("gemm_kai_a8w4_1x4 m,n,k %ld %ld %ld %.3f\n", m, n, k, during_time * 1000);
    return output;
}

}  // namespace fastertransformer
