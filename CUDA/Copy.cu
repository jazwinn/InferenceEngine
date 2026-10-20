#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"
#include "CUDA/detail/Layout.cuh"

#define INSTANTIATE_STRIDED_COPY(T) \
    template void LaunchStridedCopyDevice<T>(const T*, T*, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&);

#define INSTANTIATE_EMBEDDING_FOR_INDEX(T, Index) \
    template void LaunchEmbeddingLookUpDevice<T, Index>(const Index*, T*, const T*, size_t, size_t, size_t);
#define INSTANTIATE_EMBEDDING(T) \
    INSTANTIATE_EMBEDDING_FOR_INDEX(T, int) \
    INSTANTIATE_EMBEDDING_FOR_INDEX(T, int64_t) \
    INSTANTIATE_EMBEDDING_FOR_INDEX(T, size_t)

namespace cuda{

    template <typename T>
    __global__ void StridedCopyKernel(const T* src, T*dst, StridedLayout<2> layout, size_t size){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        size_t off[2];
        ViewOffsets(layout, i, off);

        dst[off[1]] = src[off[0]];
    }

    template <typename T, typename Index>
    __global__ void EmbeddingLookUpKernel(const Index* ids, T* out, const T* table, size_t rows, size_t dim, size_t size){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < size){
            const size_t row = i / dim;
            const size_t col = i % dim;
            const size_t id = static_cast<size_t>(ids[row]);
            out[i] = id < rows ? table[id * dim + col] : T(0);
        }
    }

    template <typename T, typename Index>
    void LaunchEmbeddingLookUpDevice(const Index* ids, T* out, const T* table, size_t num_ids, size_t rows, size_t dim){
        const size_t size = num_ids * dim;
        if(size == 0) return;

        constexpr size_t threads = 256;
        EmbeddingLookUpKernel<<<CeilDiv(size, threads), threads>>>(ids, out, table, rows, dim, size);
        CUDA_CHECK(cudaGetLastError());
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
    FOR_EACH_FLOAT_TYPE(INSTANTIATE_EMBEDDING)
}
