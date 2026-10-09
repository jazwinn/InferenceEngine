#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"

#define INSTANTIATE_STRIDED_COPY(T) \
    template void LaunchStridedCopyDevice<T>(const T*, T*, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&);

namespace cuda{

    constexpr int kMaxRank = 8;

    struct StridedLayout{
        size_t shape[kMaxRank];
        size_t src_stride[kMaxRank];
        size_t dst_stride[kMaxRank];
        int rank;
    };

    template <typename T>
    __global__ void StridedCopyKernel(const T* src, T*dst, StridedLayout layout, size_t size){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        size_t rem = i;
        size_t s = 0;
        size_t d = 0;
        for (int ax = layout.rank - 1; ax >= 0; --ax) {
            size_t idx = rem % layout.shape[ax];
            rem /= layout.shape[ax];
            s += idx * layout.src_stride[ax];
            d += idx * layout.dst_stride[ax];
        }

        dst[d] = src[s];
    }

    template <typename T>
    void LaunchStridedCopyDevice(const T* src, T* dst,
                                 const std::vector<size_t>& shape,
                                 const std::vector<size_t>& src_stride,
                                 const std::vector<size_t>& dst_stride)
    {
        if(shape.size() > kMaxRank){
            throw std::invalid_argument("StridedCopy: Shape dimension exceed kMaxRank");
        }
        if(src_stride.size() != shape.size() || dst_stride.size() != shape.size()){
            throw std::invalid_argument("StridedCopy: stride rank does not match shape rank");
        }

        StridedLayout layout{};
        layout.rank = static_cast<int>(shape.size());

        size_t size = 1;
        for(int n = 0; n < layout.rank; ++n){
            layout.shape[n] = shape[n];
            layout.src_stride[n] = src_stride[n];
            layout.dst_stride[n] = dst_stride[n];

            size *= shape[n];
        }

        if(size == 0) return;//prevents lauching if grid = 0;

        constexpr size_t threads = 256;
        StridedCopyKernel<<<CeilDiv(size, threads), threads>>>(src, dst, layout, size);
        CUDA_CHECK(cudaGetLastError());
    }

    FOR_EACH_ANY_TYPE(INSTANTIATE_STRIDED_COPY)
}