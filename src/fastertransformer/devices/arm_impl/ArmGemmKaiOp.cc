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

#include "kai/ukernels/matmul/matmul_clamp_bf16_bf16_f32p/kai_matmul_clamp_bf16_bf16_f32p12x1biasf32_8x12x4_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_bf16_f32p/kai_matmul_clamp_bf16_bf16_f32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_matmul_transpose_pack_rhs_bias_bf16p16x4zf32_bf16_f32_neon_nr_12.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_8x4_f32_bf16_neon.h"

namespace fastertransformer {

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
BufferPtr ArmCpuDevice::gemm_kai_bf16(const GemmParams& params) {

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

    const size_t mr = kai_get_mr_matmul_clamp_bf16_bf16_f32p12x1biasf32_8x12x4_neon_mmla();
    const size_t nr = kai_get_nr_matmul_clamp_bf16_bf16_f32p12x1biasf32_8x12x4_neon_mmla();
    const size_t kr = kai_get_kr_matmul_clamp_bf16_bf16_f32p12x1biasf32_8x12x4_neon_mmla();
    const size_t sr = kai_get_sr_matmul_clamp_bf16_bf16_f32p12x1biasf32_8x12x4_neon_mmla();

    // In a single row, we pack nr bias values followed by K rows of nr RHS values
    const size_t rhs_packed_size = kai_get_rhs_packed_size_matmul_transpose_pack_rhs_bias_bf16p16x4zf32_bf16_f32_neon_nr_12(n, k);

    bfloat16_t* rhs_packed = new bfloat16_t[rhs_packed_size];

    const size_t bias_size = n;
    float* bias = new float[bias_size];
    memset(bias, 0, bias_size * sizeof(float));

    const size_t lhs_stride = k * sizeof(float);
    const size_t rhs_stride = n * sizeof(float);
    const size_t dst_stride_row = n * sizeof(float);
    const size_t dst_stride_col = sizeof(float);

    const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_8x4_f32_bf16_neon(m, k, mr, kr, sr);

    bfloat16_t *lhs_packed = new bfloat16_t[lhs_packed_size];

    // float* rhs_packed = (float* )params.B.data();
    float* rhs = (float* )params.B.data();
    float* lhs = (float* )params.A.data();

    // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be constant.
    kai_run_matmul_transpose_pack_rhs_bias_bf16p16x4zf32_bf16_f32_neon_nr_12(
        1, n, k, nr, kr, sr,  // Packing arguments
        rhs_stride,           // RHS stride
        rhs,                  // RHS
        bias,                 // Bias
        NULL,                 // Scale
        rhs_packed,           // RHS packed
        0, NULL);

    float* dst = (float* )output->data();

    const size_t lhs_offset = kai_get_lhs_packed_offset_lhs_pack_8x4_f32_bf16_neon(0, k, mr, kr, sr);
    // const size_t dst_offset = 0 * dst_stride_row;
    kai_run_lhs_pack_8x4_f32_bf16_neon(
        m, k, mr, kr, sr,
        0 /* m_idx_start; should stay as 0 */,
        ((uint8_t*)lhs + 0 * lhs_stride), // adjust Lhs start position
        lhs_stride,
        ((uint8_t*)lhs_packed + lhs_offset));

    int n_step = 12;
    #pragma omp parallel for
    for (int n_start = 0; n_start < n; n_start += n_step) {
        size_t lhs_offset = kai_get_lhs_offset_matmul_clamp_bf16_bf16_f32p12x1biasf32_8x12x4_neon_mmla(0, k);
        size_t rhs_offset = kai_get_rhs_packed_offset_matmul_clamp_bf16_bf16_f32p12x1biasf32_8x12x4_neon_mmla(n_start, k);
        size_t dst_offset = kai_get_dst_offset_matmul_clamp_bf16_bf16_f32p12x1biasf32_8x12x4_neon_mmla(0, n_start, n * sizeof(bfloat16_t));

        const void* lhs_ptr = (const void*)((const char *)lhs_packed + lhs_offset);
        const void* rhs_ptr = (const void*)((const char *)rhs_packed + rhs_offset);
        void* dst_ptr = (void*)((uint8_t*)dst + dst_offset);

        assert(n % n_step == 0);
        assert(n_step % n_step == 0);

        // last tile n
        int tile_n = (n_start + n_step <= n) ? n_step : n - n_start;
        kai_run_matmul_clamp_bf16_bf16_f32p12x1biasf32_8x12x4_neon_mmla(
            m, tile_n, k,                  // Dimensions
            lhs_ptr,                      // LHS
            0,               // LHS stride
            rhs_ptr,               // RHS packed
            dst_ptr,                      // DST
            dst_stride_row,           // DST stride (row)
            dst_stride_col,           // DST stride (col)
            -FLT_MAX, FLT_MAX   // Min and max for the clamp operation
        );
    }
    // kai_run_matmul_clamp_bf16_bf16_f32p12x1biasf32_8x12x4_neon_mmla(
    //         m, n, k,                  // Dimensions
    //         ((uint8_t*)lhs_packed + lhs_offset),                      // LHS
    //         0,               // LHS stride
    //         rhs_packed,               // RHS packed
    //         ((uint8_t *)dst + dst_offset),                      // DST
    //         dst_stride_row,           // DST stride (row)
    //         dst_stride_col,           // DST stride (col)
    //         -FLT_MAX, FLT_MAX   // Min and max for the clamp operation
    // );
    
    // parallel_gemm_kai_bf16(m, n, k, lhs, rhs_packed, dst, lhs_stride, n, dst_stride_row);

    delete[] bias;
    delete[] rhs_packed;
    delete[] lhs_packed;

    auto end = std::chrono::high_resolution_clock::now();
    float during_time = std::chrono::duration<float>(end - start).count();
    printf("gemm_kai_bf16 m,n,k %ld %ld %ld %.3f\n", m, n, k, during_time * 1000);
    return output;
}

}  // namespace fastertransformer
