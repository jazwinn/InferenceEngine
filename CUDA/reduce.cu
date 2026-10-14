#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"

namespace cuda{

    constexpr unsigned kReduceThreads = 256;

    template <typename T>
    __global__ void SumKernel(const T* in, T* out, size_t size){
        __shared__ T buf[kReduceThreads];
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        buf[threadIdx.x] = (i < size) ? in[i] : T(0);
        __syncthreads();

        for(unsigned s = blockDim.x/2; s > 0; s >>=1){
            if(threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
            __syncthreads();
        }

        if(threadIdx.x == 0){
            atomicAdd(out, buf[0]);
        }

    }

    template <typename T>
    void LaunchSumDevice(const T* in, T* out, size_t size){
        if (size == 0) return;

        const size_t blocks = CeilDiv(size, static_cast<size_t>(kReduceThreads));

        SumKernel<<<blocks, kReduceThreads>>>(in, out, size);

        CUDA_CHECK(cudaGetLastError());
    }

    template void LaunchSumDevice<float>(const float*, float*, size_t);
    template void LaunchSumDevice<double>(const double*, double*, size_t);
}
