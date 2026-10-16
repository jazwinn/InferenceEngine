#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"

#define INSTANTIATE_OPS(T) \
    template void LaunchRoPEDevice<T>(const T* in, T* out, const T* cos, const T* sin, size_t rows, size_t length, size_t half, size_t start_pos);

namespace cuda{

    template <typename T>
    __global__ void RoPEKernel(const T* in, T* out, const T* cos, const T* sin,
                               size_t pairs, size_t length, size_t half, size_t start_pos)
    {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < pairs){
            const size_t i = idx % half;
            const size_t row = idx / half;
            const size_t s = row % length;

            const size_t base = row * 2 * half;
            const size_t t = (start_pos + s) * half + i;

            const T x1 = in[base + i];
            const T x2 = in[base + i + half];
            const T c = cos[t];
            const T sn = sin[t];

            out[base + i] = x1 * c - x2 * sn;
            out[base + i + half] = x2 * c + x1 * sn;
        }
    }

    template <typename T>
    void LaunchRoPEDevice(const T* in, T* out, const T* cos, const T* sin,
                          size_t rows, size_t length, size_t half, size_t start_pos){
        const size_t pairs = rows * length * half;
        if (pairs == 0) return;
        constexpr size_t threads = 256;
        RoPEKernel<<<CeilDiv(pairs, threads), threads>>>(in, out, cos, sin, pairs, length, half, start_pos);
        CUDA_CHECK(cudaGetLastError());
    }

    FOR_EACH_TYPE(INSTANTIATE_OPS)
}
