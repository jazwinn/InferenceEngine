#include <cstddef>
#include <stdexcept>
#include <vector>

#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"
#include "CUDA/detail/Layout.cuh"
#include "CUDA/detail/Ops.cuh"

#define INSTANTIATE_OPS(T) \
    template void LaunchScalarDevice<T>(const T*, T*, T, size_t, BinaryOp); \
    template void LaunchBinaryDevice<T>(const T* A, const T* B, T* out, size_t size, BinaryOp op); \
    template void LaunchStridedBinaryDevice<T>(const T* A, const T* B, T* out, \
                                   const std::vector<size_t>& shape, \
                                   const std::vector<size_t>& a_stride, \
                                   const std::vector<size_t>& b_stride, \
                                   const std::vector<size_t>& out_stride, \
                                   BinaryOp op); \
    template void LaunchStridedScalarDevice<T>(T*, T, const std::vector<size_t>&, const std::vector<size_t>&, BinaryOp);

#define INSTANTIATE_FILL(T) \
    template void LaunchFillDevice<T>(T*, T, size_t); \
    template void LaunchStridedFillDevice<T>(T*, T, const std::vector<size_t>&, const std::vector<size_t>&);

namespace cuda{


    template <typename T, typename Op>
    __global__ void BinaryKernel(const T* A, const T* B, T* out, size_t size, Op op){
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            out[i] = op(A[i], B[i]);
        }
    }

    template <typename T, typename Op>
    __global__ void StridedBinaryKernel(const T* A, const T* B, T* out, StridedLayout<3> layout, size_t size, Op op){
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            size_t off[3];
            ViewOffsets(layout, i, off);
            out[off[2]] = op(A[off[0]], B[off[1]]);
        }
    }


    template <typename T, typename Op>
    __global__ void ScalarKernel(const T* in, T* out, T scalar, size_t size, Op op){

        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            out[i] = op(in[i], scalar);
        }
    }

    template <typename T, typename Op>
    __global__ void UnaryKernel(const T* in, T* out, size_t size, Op op){
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            out[i] = op(in[i]);
        }
    }

    template <typename T>
    __global__ void FillKernel(T* out, T value, size_t size){
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            out[i] = value;
        }
    }


    template <typename T>
    __global__ void StridedFillKernel(T* out, T value, StridedLayout<1> layout, size_t size){
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            size_t off[1];
            ViewOffsets(layout, i, off);
            out[off[0]] = value;
        }
    }

    template <typename T, typename Op>
    __global__ void StridedScalarKernel(T* out, T scalar, StridedLayout<1> layout, size_t size, Op op){
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            size_t off[1];
            ViewOffsets(layout, i, off);
            out[off[0]] = op(out[off[0]], scalar);
        }
    }

    template <typename T>
    void LaunchUnaryDevice(const T* in, T* out, size_t size, UnaryOp op){
        if (size == 0) return;

        constexpr size_t threads = 256;
        const size_t blocks = CeilDiv(size, threads);

        switch (op)
        {
        case UnaryOp::Sqrt:
            UnaryKernel<<<blocks, threads>>>(in, out, size, SqrtOp<T>{});
            break;
        default:
            throw std::invalid_argument("LaunchUnaryDevice: unsupported op.");
        }

        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchScalarDevice(const T* in, T* out, T scalar, size_t size, BinaryOp op){
        if (size == 0) return;

        constexpr size_t threads = 256;
        const size_t blocks = CeilDiv(size, threads);

        switch (op)
        {
        case BinaryOp::Add:
            ScalarKernel<<<blocks, threads>>>(in, out, scalar, size, AddOp<T>{});
            break;
        case BinaryOp::Sub:
            ScalarKernel<<<blocks, threads>>>(in, out, scalar, size, SubOp<T>{});
            break;
        case BinaryOp::Mul:
            ScalarKernel<<<blocks, threads>>>(in, out, scalar, size, MulOp<T>{});
            break;
        case BinaryOp::Div:
            ScalarKernel<<<blocks, threads>>>(in, out, scalar, size, DivOp<T>{});
            break;
        default:
            throw std::invalid_argument("LaunchScalarDevice: unsupported op.");
        }

        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchBinaryDevice(const T* A, const T* B, T* out, size_t size, BinaryOp op){
        if (size == 0) return;

        constexpr size_t threads = 256;
        const size_t blocks = CeilDiv(size, threads);

        switch (op)
        {
        case BinaryOp::Add:
            BinaryKernel<<<blocks, threads>>>(A, B, out, size, AddOp<T>{});
            break;
        case BinaryOp::Sub:
            BinaryKernel<<<blocks, threads>>>(A, B, out, size, SubOp<T>{});
            break;
        case BinaryOp::Mul:
            BinaryKernel<<<blocks, threads>>>(A, B, out, size, MulOp<T>{});
            break;
        case BinaryOp::Div:
            BinaryKernel<<<blocks, threads>>>(A, B, out, size, DivOp<T>{});
            break;
        default:
            throw std::invalid_argument("LaunchBinaryDevice: unsupported op.");
        }

        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchStridedBinaryDevice(const T* A, const T* B, T* out,
                                   const std::vector<size_t>& shape,
                                   const std::vector<size_t>& a_stride,
                                   const std::vector<size_t>& b_stride,
                                   const std::vector<size_t>& out_stride,
                                   BinaryOp op)
    {
        size_t size = 0;
        const StridedLayout<3> layout = MakeStridedLayout("StridedBinary", size, shape, a_stride, b_stride, out_stride);

        if(size == 0) return;//prevents lauching if grid = 0;

        constexpr size_t threads = 256;
        size_t blocks = CeilDiv(size, threads);

        switch (op)
        {
        case BinaryOp::Add:
            StridedBinaryKernel<<<blocks, threads>>>(A, B, out, layout, size, AddOp<T>{});
            break;
        case BinaryOp::Sub:
            StridedBinaryKernel<<<blocks, threads>>>(A, B, out, layout, size, SubOp<T>{});
            break;
        case BinaryOp::Mul:
            StridedBinaryKernel<<<blocks, threads>>>(A, B, out, layout, size, MulOp<T>{});
            break;
        case BinaryOp::Div:
            StridedBinaryKernel<<<blocks, threads>>>(A, B, out, layout, size, DivOp<T>{});
            break;
        default:
            throw std::invalid_argument("LaunchStridedBinaryDevice: unsupported op.");
        }

        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchFillDevice(T* out, T value, size_t size){
        if(size == 0) return;
        
        constexpr size_t threads = 256;
        size_t blocks = CeilDiv(size, threads);

        FillKernel<<<blocks, threads>>>(out, value, size);

        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchStridedFillDevice(T* out, T value,
                                 const std::vector<size_t>& shape,
                                 const std::vector<size_t>& stride)
    {
        size_t size = 0;
        const StridedLayout<1> layout = MakeStridedLayout("StridedFill", size, shape, stride);

        if(size == 0) return;

        constexpr size_t threads = 256;
        const size_t blocks = CeilDiv(size, threads);

        StridedFillKernel<<<blocks, threads>>>(out, value, layout, size);

        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchStridedScalarDevice(T* out, T scalar,
                                   const std::vector<size_t>& shape,
                                   const std::vector<size_t>& stride,
                                   BinaryOp op)
    {
        size_t size = 0;
        const StridedLayout<1> layout = MakeStridedLayout("StridedScalar", size, shape, stride);

        if(size == 0) return;

        constexpr size_t threads = 256;
        const size_t blocks = CeilDiv(size, threads);

        switch (op)
        {
        case BinaryOp::Add:
            StridedScalarKernel<<<blocks, threads>>>(out, scalar, layout, size, AddOp<T>{});
            break;
        case BinaryOp::Sub:
            StridedScalarKernel<<<blocks, threads>>>(out, scalar, layout, size, SubOp<T>{});
            break;
        case BinaryOp::Mul:
            StridedScalarKernel<<<blocks, threads>>>(out, scalar, layout, size, MulOp<T>{});
            break;
        case BinaryOp::Div:
            StridedScalarKernel<<<blocks, threads>>>(out, scalar, layout, size, DivOp<T>{});
            break;
        default:
            throw std::invalid_argument("LaunchStridedScalarDevice: unsupported op.");
        }

        CUDA_CHECK(cudaGetLastError());
    }

    INSTANTIATE_OPS(float)
    INSTANTIATE_OPS(double)
    INSTANTIATE_OPS(int)

    template void LaunchUnaryDevice<float>(const float*, float*, size_t, UnaryOp);
    template void LaunchUnaryDevice<double>(const double*, double*, size_t, UnaryOp);

    FOR_EACH_ANY_TYPE(INSTANTIATE_FILL)

}
