#include <stdexcept>

#include "CUDA/Kernel.cuh"
#include "CUDA/DeviceBuffer.h"
#include "CUDA/detail/common.cuh"

#define INSTANTIATE_MATMUL(T) \
    template void LaunchMatMul<T>(const T*, const T*, T*, size_t, size_t, size_t); \
    template void LaunchMatMul<T>(const DeviceBuffer<T>&, const DeviceBuffer<T>&, DeviceBuffer<T>&, size_t, size_t, size_t);


namespace cuda{
    
    template<typename T>
    __global__ void MatMulKernel(const T* A, const T* B, T* C, size_t M, size_t N, size_t K){
        
        size_t row = blockIdx.y * blockDim.y + threadIdx.y;
        size_t col = blockIdx.x * blockDim.x + threadIdx.x; 

        if(row < M && col < K){
            T sum = 0.f;

            for(size_t n = 0; n < N; ++n){
                sum += *(A + row * N + n) * *(B + n * K + col); 
            }

            C[row * K + col] = sum;
        }
    }

    template <typename T>
    void LaunchMatMul(const DeviceBuffer<T>& dA, const DeviceBuffer<T>& dB, DeviceBuffer<T>& dC, size_t M, size_t N, size_t K){
        if (dA.Size() < M * N || dB.Size() < N * K || dC.Size() < M * K)
            throw std::invalid_argument("LaunchMatMul: buffer too small for given M, N, K");

        dim3 block(16, 16);
        dim3 grid(CeilDiv(K, block.x), CeilDiv(M, block.y));

        MatMulKernel<<<grid, block>>>(dA.Data(), dB.Data(), dC.Data(), M, N, K);
        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchMatMul(const T* A, const T* B, T* C, size_t M, size_t N, size_t K){
        size_t countA = M * N;
        size_t countB = N * K;
        size_t countC = M * K;

        DeviceBuffer<T> dA(A, countA);
        DeviceBuffer<T> dB(B, countB);
        DeviceBuffer<T> dC(countC);

        LaunchMatMul(dA, dB, dC, M, N, K);

        dC.CopyToHost(C, countC);

    }



    FOR_EACH_TYPE(INSTANTIATE_MATMUL)
}
