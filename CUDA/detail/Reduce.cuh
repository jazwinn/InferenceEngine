#pragma once

#include <cassert>

#include "CUDA/detail/common.cuh"

namespace cuda{
    template <unsigned kThreads, typename T, typename Op>
    __device__ T BlockReduce(const T val, Op op){
        static_assert(kThreads > 0 && (kThreads & (kThreads - 1)) == 0, "BlockReduce: kThreads must be a power of 2");
        assert(blockDim.x == kThreads);

        __shared__ T buf[kThreads];

        __syncthreads();
        buf[threadIdx.x] = val;
        __syncthreads();

        for (unsigned s = kThreads / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) buf[threadIdx.x] = op(buf[threadIdx.x], buf[threadIdx.x + s]);
            __syncthreads();
        }

        return buf[0];
    }
}
