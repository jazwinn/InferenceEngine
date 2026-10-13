#include <cstddef>
#include <stdexcept>

#include "CUDA/Kernel.cuh"
#include "CUDA/detail/common.cuh"

#define INSTANTIATE_OPS(T) \
    template void LaunchScalarDevice<T>(const T*, T*, T, size_t, BinaryOp); \
    template void LaunchBinaryDevice<T>(const T* A, const T* B, T* out, size_t size, BinaryOp op);

namespace cuda{

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
    __global__ void ScalarKernel(const T* in, T* out, T scalar, size_t size, Op op){

        size_t i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < size){
            out[i] = op(in[i], scalar);
        }
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

    INSTANTIATE_OPS(float)
    INSTANTIATE_OPS(double)
    INSTANTIATE_OPS(int)

}
