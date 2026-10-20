#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                         cudaGetErrorString(err), __FILE__, __LINE__);    \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

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