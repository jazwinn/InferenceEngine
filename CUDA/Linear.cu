#include <stdexcept>

#include "CUDA/Kernel.cuh"
#include "CUDA/Memory.h"
#include "CUDA/detail/common.cuh"

#define INSTANTIATE_LINEAR(T) \
    template void LaunchLinearDevice<T>(const T*, const T*, const T*, T*, size_t, size_t, size_t); \
    template void LaunchLinear<T>(const DeviceBuffer<T>&, const DeviceBuffer<T>&, const DeviceBuffer<T>&, DeviceBuffer<T>&, size_t, size_t, size_t); \
    template void LaunchLinearHost<T>(const T*, const T*, const T*, T*, size_t, size_t, size_t);

namespace cuda{

    template <typename T>
    __global__ void LinearKernel(const T* X, const T* W, const T* B, T* Y, size_t M, size_t in, size_t out){
        size_t col = blockDim.x * blockIdx.x + threadIdx.x;
        size_t row = blockDim.y * blockIdx.y + threadIdx.y;

        if(row < M && col < out){

            T sum = B? B[col] : T(0);
            for(size_t n = 0; n < in; ++n){
                sum += X[row * in + n] * W[col * in + n];
            }
            
            
            Y[row * out + col] = sum;
        }

    }

    template <typename T>
    void LaunchLinearDevice(const T* dX, const T* dW, const T* dB, T* dY, size_t M, size_t in, size_t out){
        dim3 block(16, 16);
        dim3 grid(CeilDiv(out, block.x), CeilDiv(M, block.y));

        LinearKernel<<<grid, block>>>(dX, dW, dB, dY, M, in, out);
        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchLinear(const DeviceBuffer<T>& dX, 
                      const DeviceBuffer<T>& dW, 
                      const DeviceBuffer<T>& dB, 
                      DeviceBuffer<T>& dY, 
                      size_t M, size_t in, size_t out){
        if (dX.Size() < M * in || dW.Size() < out * in || dY.Size() < M * out ||
            (dB.Size() != 0 && dB.Size() < out))
            throw std::invalid_argument("LaunchLinear: buffer too small for given M, in, out");

        LaunchLinearDevice(dX.Data(), dW.Data(), dB.Size() ? dB.Data() : nullptr, dY.Data(), M, in, out);
    }

    template <typename T>
    void LaunchLinearHost(const T* X, const T* W, const T* B, T* Y, size_t M, size_t in, size_t out){
        DeviceBuffer<T> dX(X, M * in), dW(W, out * in), dY(M * out);
        DeviceBuffer<T> dB = B ? DeviceBuffer<T>(B, out) : DeviceBuffer<T>();
        LaunchLinearDevice(dX.Data(), dW.Data(), dB.Data(), dY.Data(), M, in, out);
        dY.CopyToHost(Y, M * out);
    }

    FOR_EACH_TYPE(INSTANTIATE_LINEAR)

}