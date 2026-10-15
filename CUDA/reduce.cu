#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"
#include "CUDA/detail/Reduce.cuh"
#include "CUDA/detail/Ops.cuh"
#include "CUDA/detail/Layout.cuh"

#define INSTANTIATE_OPS(T) \
    template void LaunchSumDevice<T>(const T* in, T* out, size_t size); \
    template void LaunchMeanAxisDevice<T>(const T* in, T* out, size_t outer, size_t reduced, size_t inner);

namespace cuda{

    constexpr unsigned kReduceThreads = 256;

    template <typename T>
    __global__ void SumKernel(const T* in, T* out, size_t size){
        
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        const T val = (i < size) ? in[i] : T(0);
        T blockTotal = BlockReduce<kReduceThreads>(val, AddOp<T>{});
        if(threadIdx.x == 0) atomicAdd(out, blockTotal);

    }

    template <typename T>
    __global__ void MeanAxisKernel(const T* in, T* out, size_t reduced, size_t inner){
        const size_t o = blockIdx.x / inner;
        const size_t i = blockIdx.x % inner;
        const T* base = in + o * reduced * inner + i;

        T local = T(0);
        for (size_t r = threadIdx.x; r < reduced; r += blockDim.x) {
            local += base[r * inner];
        }

        const T total = BlockReduce<kReduceThreads>(local, AddOp<T>{});
        if (threadIdx.x == 0) out[blockIdx.x] = total / static_cast<T>(reduced);
    }

    template <typename T>
    void LaunchSumDevice(const T* in, T* out, size_t size){
        if (size == 0) return;

        const size_t blocks = CeilDiv(size, static_cast<size_t>(kReduceThreads));

        SumKernel<<<blocks, kReduceThreads>>>(in, out, size);

        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchMeanAxisDevice(const T* in, T* out, size_t outer, size_t reduced, size_t inner){
        const size_t blocks = outer * inner;
        if (blocks == 0 || reduced == 0) return;
        MeanAxisKernel<<<blocks, kReduceThreads>>>(in, out, reduced, inner);
        CUDA_CHECK(cudaGetLastError());
    }

    FOR_EACH_TYPE(INSTANTIATE_OPS);
}
