#include <cstddef>
#include <stdexcept>

#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"
#include "CUDA/detail/Ops.cuh"

#define INSTANTIATE_ACTIVATION(T) \
    template void LaunchActivationDevice<T>(const T* in, T* out, size_t size, Activation act);

namespace cuda{

    template<typename T, typename Op>
    __global__ void ActivationKernel(const T* in, T* out, size_t size, Op op){

        size_t col = blockDim.x * blockIdx.x + threadIdx.x;

        if(col < size){
            out[col] = op(in[col]);
        }
    }

    template <typename T>
    void LaunchActivationDevice(const T* in, T* out, size_t size, Activation act){
        if (size == 0) return;

        constexpr size_t threads = 256;
        const size_t blocks = CeilDiv(size, threads);

        switch (act)
        {
        case Activation::ReLU:
            ActivationKernel<<<blocks, threads>>>(in, out, size, RELUops<T>{});
            break;
        case Activation::GELU:
            ActivationKernel<<<blocks, threads>>>(in, out, size, GELUops<T>{});
            break;
        case Activation::GELUTanh:
            ActivationKernel<<<blocks, threads>>>(in, out, size, GELUTanhops<T>{});
            break;
        case Activation::Sigmoid:
            ActivationKernel<<<blocks, threads>>>(in, out, size, Sigmoidops<T>{});
            break;
        case Activation::SiLU:
            ActivationKernel<<<blocks, threads>>>(in, out, size, SiLUops<T>{});
            break;
        case Activation::Tanh:
            ActivationKernel<<<blocks, threads>>>(in, out, size, Tanhops<T>{});
            break;
        default:
            throw std::invalid_argument("LaunchActivationDevice: unsupported activation.");
        }

        CUDA_CHECK(cudaGetLastError());
    }

    FOR_EACH_TYPE(INSTANTIATE_ACTIVATION)
}
