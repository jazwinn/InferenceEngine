#include <cstddef>
#include <stdexcept>

#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"

#define INSTANTIATE_ACTIVATION(T) \
    template void LaunchActivationDevice<T>(const T* in, T* out, size_t size, Activation act); \

namespace cuda{

    template <typename T>
    struct RELUops{
        __device__ T operator()(T x) const {
            return (x < T(0))? T(0) : x;
        }
    };

    template <typename T>
    struct GELUops{
        __device__ T operator()(T x) const {
            constexpr T ONE_OVER_SQRT_2 = T(0.7071067811865476);
            return T(0.5) * x * (T(1) + erf(x * ONE_OVER_SQRT_2));
        }
    };

    template <typename T>
    struct GELUTanhops{
        __device__ T operator()(T x) const {
            constexpr T SQRT_2_OVER_PI = T(0.7978845608028654);
            constexpr T COEFFICIENT = T(0.044715);
            const T inner = SQRT_2_OVER_PI * (x + COEFFICIENT * x * x * x);
            return T(0.5) * x * (T(1) + tanh(inner));
        }
    };

    template <typename T>
    struct Sigmoidops{
        __device__ T operator()(T x) const {
            return T(1) / (T(1) + exp(-x));
        }
    };

    template <typename T>
    struct SiLUops{
        __device__ T operator()(T x) const {
            return x / (T(1) + exp(-x));
        }
    };

    template <typename T>
    struct Tanhops{
        __device__ T operator()(T x) const {
            return tanh(x);
        }
    };



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

    FOR_EACH_TYPE(INSTANTIATE_ACTIVATION);
}
