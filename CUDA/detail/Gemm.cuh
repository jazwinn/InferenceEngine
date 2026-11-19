#pragma once

#include <limits>
#include <type_traits>

#include <cublas_v2.h>

#include "CUDA/detail/common.cuh"

namespace cuda{

    struct GemmOperand {
        cublasOperation_t op;
        int ld;
        bool ok;
    };
    
    class CublasHandle {
    public:
        CublasHandle() { CUBLAS_CHECK(cublasCreate(&m_handle)); }
        ~CublasHandle() { cublasDestroy(m_handle); }
        CublasHandle(const CublasHandle&) = delete;
        CublasHandle& operator=(const CublasHandle&) = delete;
        cublasHandle_t Get() const { return m_handle; }
    private:
        cublasHandle_t m_handle = nullptr;
    };

    inline cublasHandle_t Cublas() {
        static CublasHandle handle;
        return handle.Get();
    }

    template <typename T>
    constexpr cudaDataType_t CudaType() {
        if constexpr (std::is_same_v<T, __half>)             return CUDA_R_16F;
        else if constexpr (std::is_same_v<T, float>)         return CUDA_R_32F;
        else if constexpr (std::is_same_v<T, double>)        return CUDA_R_64F;
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) return CUDA_R_16BF;
        else static_assert(sizeof(T) == 0, "CudaType: unsupported type");
    }

    template <typename In, typename Out = In>
    void Gemm(cublasOperation_t opB, cublasOperation_t opA, int m, int n, int k,
              const In* B, int ldb, const In* A, int lda, Out* C, int ldc,
              double beta_value = 0.0)
    {
        constexpr bool f64 = std::is_same_v<In, double>;
        using Scale = std::conditional_t<f64, double, float>;
        const Scale alpha = 1, beta = static_cast<Scale>(beta_value);

        CUBLAS_CHECK(cublasGemmEx(Cublas(), opB, opA, m, n, k, &alpha,
                                  B, CudaType<In>(), ldb, A, CudaType<In>(), lda, &beta,
                                  C, CudaType<Out>(), ldc,
                                  f64 ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
    }

    inline GemmOperand ToGemmOperand(size_t rs, size_t cs, size_t row_count, size_t col_count) {
        constexpr size_t kMax = static_cast<size_t>(std::numeric_limits<int>::max());
        if (cs == 1 || col_count == 1) {
            const size_t ld = row_count == 1 ? col_count : rs;
            return {CUBLAS_OP_N, static_cast<int>(ld), ld >= col_count && ld >= 1 && ld <= kMax};
        }
        if (rs == 1 || row_count == 1) {
            const size_t ld = col_count == 1 ? row_count : cs;
            return {CUBLAS_OP_T, static_cast<int>(ld), ld >= row_count && ld >= 1 && ld <= kMax};
        }
        return {CUBLAS_OP_N, 0, false};
    }

}