#include <cstddef>
#include <stdexcept>
#include <vector>

#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"

#define INSTANTIATE_OPS(T) \
    template void LaunchScalarDevice<T>(const T*, T*, T, size_t, BinaryOp); \
    template void LaunchBinaryDevice<T>(const T* A, const T* B, T* out, size_t size, BinaryOp op); \
    template void LaunchStridedBinaryDevice<T>(const T* A, const T* B, T* out, \
                                   const std::vector<size_t>& shape, \
                                   const std::vector<size_t>& a_stride, \
                                   const std::vector<size_t>& b_stride, \
                                   const std::vector<size_t>& out_stride, \
                                   BinaryOp op);
namespace cuda{

    struct BinaryLayout {
        size_t shape[kMaxRank];
        size_t a_stride[kMaxRank];
        size_t b_stride[kMaxRank];
        size_t out_stride[kMaxRank];
        int rank;
    };

    template <typename T>
    struct AddOp{__device__ T operator()(T a, T b)const{return a + b;}};

    template <typename T>
    struct SubOp{__device__ T operator()(T a, T b)const{return a - b;}};

    template <typename T>
    struct MulOp{__device__ T operator()(T a, T b)const{return a * b;}};

    template <typename T>
    struct DivOp{__device__ T operator()(T a, T b)const{return a / b;}};

    template <typename T, typename Op>
    __global__ void BinaryKernel(const T* A, const T* B, T* out, size_t size, Op op){
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            out[i] = op(A[i], B[i]);
        }
    }

    template <typename T, typename Op>
    __global__ void StridedBinaryKernel(const T* A, const T* B, T* out, BinaryLayout layout, size_t size, Op op){
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            size_t rem = i, idxA = 0, idxB = 0, idxOut = 0;
            for (int ax = layout.rank - 1; ax >= 0; --ax) {
                size_t idx = rem % layout.shape[ax];
                rem /= layout.shape[ax];
                idxA += idx * layout.a_stride[ax];
                idxB += idx * layout.b_stride[ax];
                idxOut += idx * layout.out_stride[ax];
            }
            out[idxOut] = op(A[idxA], B[idxB]);
        }
    }


    template <typename T, typename Op>
    __global__ void ScalarKernel(const T* in, T* out, T scalar, size_t size, Op op){

        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            out[i] = op(in[i], scalar);
        }
    }

    template <typename T>
    struct SqrtOp{__device__ T operator()(T x)const{return sqrt(x);}};

    template <typename T, typename Op>
    __global__ void UnaryKernel(const T* in, T* out, size_t size, Op op){
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            out[i] = op(in[i]);
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
        if(shape.size() > kMaxRank){
            throw std::invalid_argument("StridedBinary: Shape dimension exceed kMaxRank");
        }
        if(a_stride.size() != shape.size() || b_stride.size() != shape.size() || out_stride.size() != shape.size()){
            throw std::invalid_argument("StridedBinary: stride rank does not match shape rank");
        }

        BinaryLayout layout{};
        layout.rank = static_cast<int>(shape.size());

        size_t size = 1;
        for(int n = 0; n < layout.rank; ++n){
            layout.shape[n] = shape[n];
            layout.a_stride[n] = a_stride[n];
            layout.b_stride[n] = b_stride[n];
            layout.out_stride[n] = out_stride[n];

            size *= shape[n];
        }

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

    INSTANTIATE_OPS(float)
    INSTANTIATE_OPS(double)
    INSTANTIATE_OPS(int)

    template void LaunchUnaryDevice<float>(const float*, float*, size_t, UnaryOp);
    template void LaunchUnaryDevice<double>(const double*, double*, size_t, UnaryOp);

}
