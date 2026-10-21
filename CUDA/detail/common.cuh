#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define IE_CHECK(call, Status, success, to_string, kind)                  \
    do {                                                                  \
        const Status ie_check_status_ = (call);                           \
        if (ie_check_status_ != (success)) {                              \
            std::fprintf(stderr, "%s error %s at %s:%d\n", kind,          \
                         to_string(ie_check_status_), __FILE__, __LINE__);\
            std::abort();                                                 \
        }                                                                 \
    } while (0)

#define CUDA_CHECK(call)                                                  \
    IE_CHECK(call, cudaError_t, cudaSuccess, cudaGetErrorString, "CUDA")

#define CUBLAS_CHECK(call)                                                \
    IE_CHECK(call, cublasStatus_t, CUBLAS_STATUS_SUCCESS, cublasGetStatusString, "cuBLAS")


#define FOR_EACH_HALF_TYPE(X)    X(__half) X(__nv_bfloat16)
#define FOR_EACH_FLOAT_TYPE(X)   X(float) X(double)
#define FOR_EACH_INT_TYPE(X)     X(int) X(int64_t) X(size_t)
#define FOR_EACH_NUMERIC_TYPE(X) FOR_EACH_FLOAT_TYPE(X) FOR_EACH_INT_TYPE(X)
#define FOR_EACH_ANY_TYPE(X)     FOR_EACH_NUMERIC_TYPE(X) X(bool)


namespace cuda{

    constexpr int kMaxRank = 8;

    __host__ __device__ constexpr size_t CeilDiv(size_t a, size_t b) { 
        return (a + b - 1) / b; 
    }

}