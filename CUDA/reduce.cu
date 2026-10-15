#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"
#include "CUDA/detail/Reduce.cuh"
#include "CUDA/detail/Ops.cuh"

#define INSTANTIATE_OPS(T) \
    template void LaunchSumDevice<T>(const T* in, T* out, size_t size);

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
    void LaunchSumDevice(const T* in, T* out, size_t size){
        if (size == 0) return;

        const size_t blocks = CeilDiv(size, static_cast<size_t>(kReduceThreads));

        SumKernel<<<blocks, kReduceThreads>>>(in, out, size);

        CUDA_CHECK(cudaGetLastError());
    }

    FOR_EACH_TYPE(INSTANTIATE_OPS);
}
