#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"
#include "CUDA/detail/Layout.cuh"

#define INSTANTIATE_STRIDED_COPY(T) \
    template void LaunchStridedCopyDevice<T>(const T*, T*, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&);

namespace cuda{

    template <typename T>
    __global__ void StridedCopyKernel(const T* src, T*dst, StridedLayout<2> layout, size_t size){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        size_t off[2];
        ViewOffsets(layout, i, off);

        dst[off[1]] = src[off[0]];
    }

    template <typename T>
    void LaunchStridedCopyDevice(const T* src, T* dst,
                                 const std::vector<size_t>& shape,
                                 const std::vector<size_t>& src_stride,
                                 const std::vector<size_t>& dst_stride)
    {
        size_t size = 0;
        const StridedLayout<2> layout = MakeStridedLayout("StridedCopy", size, shape, src_stride, dst_stride);

        if(size == 0) return;//prevents lauching if grid = 0;

        constexpr size_t threads = 256;
        StridedCopyKernel<<<CeilDiv(size, threads), threads>>>(src, dst, layout, size);
        CUDA_CHECK(cudaGetLastError());
    }

    FOR_EACH_ANY_TYPE(INSTANTIATE_STRIDED_COPY)
}
