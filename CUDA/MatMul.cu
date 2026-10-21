#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "CUDA/Kernel.cuh"
#include "CUDA/Memory.h"
#include "CUDA/detail/common.cuh"
#include "CUDA/detail/Layout.cuh"

#define INSTANTIATE_MATMUL(T) \
    template void LaunchMatMulDevice<T>(const T*, const T*, T*, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t, size_t, size_t, size_t, size_t, size_t); \
    template void LaunchMatMulDevice<T>(const T*, const T*, T*, size_t, size_t, size_t); \
    template void LaunchMatMulHost<T>(const T*, const T*, T*, size_t, size_t, size_t); \
    template void LaunchMatMul<T>(const DeviceBuffer<T>&, const DeviceBuffer<T>&, DeviceBuffer<T>&, size_t, size_t, size_t);

namespace {

    struct GemmOperand {
        cublasOperation_t op;
        int ld;
        bool ok;
    };
    
    class CublasHandle {
    public:
        CublasHandle() { CUBLAS_CHECK(cublasCreate(&m_handle)); }
        ~CublasHandle() { cublasDestroy(m_handle); }
        CublasHandle(const CublasHandle&) = delete;
        CublasHandle& operator=(const CublasHandle&) = delete;
        cublasHandle_t Get() const { return m_handle; }
    private:
        cublasHandle_t m_handle = nullptr;
    };

    cublasHandle_t Cublas() {
        static CublasHandle handle;
        return handle.Get();
    }

    template <typename T>
    constexpr cudaDataType_t CudaType() {
        if constexpr (std::is_same_v<T, __half>)             return CUDA_R_16F;
        else if constexpr (std::is_same_v<T, float>)         return CUDA_R_32F;
        else if constexpr (std::is_same_v<T, double>)        return CUDA_R_64F;
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) return CUDA_R_16BF;
        else static_assert(sizeof(T) == 0, "CudaType: unsupported type");
    }

    template <typename In, typename Out = In>
    void Gemm(cublasOperation_t opB, cublasOperation_t opA, int m, int n, int k,
              const In* B, int ldb, const In* A, int lda, Out* C, int ldc)
    {
        constexpr bool f64 = std::is_same_v<In, double>;
        using Scale = std::conditional_t<f64, double, float>;
        const Scale alpha = 1, beta = 0;

        CUBLAS_CHECK(cublasGemmEx(Cublas(), opB, opA, m, n, k, &alpha,
                                  B, CudaType<In>(), ldb, A, CudaType<In>(), lda, &beta,
                                  C, CudaType<Out>(), ldc,
                                  f64 ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
    }

    GemmOperand ToGemmOperand(size_t rs, size_t cs, size_t row_count, size_t col_count) {
        constexpr size_t kMax = static_cast<size_t>(std::numeric_limits<int>::max());
        if (cs == 1 || col_count == 1) {
            const size_t ld = row_count == 1 ? col_count : rs;
            return {CUBLAS_OP_N, static_cast<int>(ld), ld >= col_count && ld >= 1 && ld <= kMax};
        }
        if (rs == 1 || row_count == 1) {
            const size_t ld = col_count == 1 ? row_count : cs;
            return {CUBLAS_OP_T, static_cast<int>(ld), ld >= row_count && ld >= 1 && ld <= kMax};
        }
        return {CUBLAS_OP_N, 0, false};
    }

}


namespace cuda{

    template<typename T>
    __global__ void MatMulKernel(const T* A, const T* B, T* C, StridedLayout<2> layout,
                                 size_t rows, size_t inner, size_t cols,
                                 size_t a_rs, size_t a_cs, size_t b_rs, size_t b_cs){

        const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t z = blockIdx.z;

        if (row >= rows || col >= cols) return;

        size_t off[2];
        ViewOffsets(layout, z, off);

        const T* a = A + off[0] + row * a_rs;
        const T* b = B + off[1] + col * b_cs;

        T sum = T(0);
        for (size_t n = 0; n < inner; ++n) {
            sum += a[n * a_cs] * b[n * b_rs];
        }

        C[z * rows * cols + row * cols + col] = sum;
    }

    template <typename T>
    void LaunchMatMulDevice(const T* A, const T* B, T* C,
                                const std::vector<size_t>& batch,
                                const std::vector<size_t>& a_batch_stride,
                                const std::vector<size_t>& b_batch_stride,
                                size_t rows, size_t inner, size_t cols,
                                size_t a_rs, size_t a_cs, size_t b_rs, size_t b_cs){

        size_t batch_count = 0;
        const StridedLayout<2> layout = MakeStridedLayout("BatchedMatMul", batch_count, batch, a_batch_stride, b_batch_stride);

        if (batch_count == 0 || rows == 0 || cols == 0) return;

        constexpr size_t kMax = static_cast<size_t>(std::numeric_limits<int>::max());
        if (batch_count == 1 && rows <= kMax && inner <= kMax && cols <= kMax) {
            const GemmOperand b = ToGemmOperand(b_rs, b_cs, inner, cols);
            const GemmOperand a = ToGemmOperand(a_rs, a_cs, rows, inner);
            if (a.ok && b.ok) {
                Gemm(b.op, a.op,
                    static_cast<int>(cols), static_cast<int>(rows), static_cast<int>(inner),
                    B, b.ld, A, a.ld, C, static_cast<int>(cols));
                return;
            }
        }

        
        static const size_t max_grid_z = []{
            int device_id = 0;
            CUDA_CHECK(cudaGetDevice(&device_id));
            int max_z = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&max_z, cudaDevAttrMaxGridDimZ, device_id));
            return static_cast<size_t>(max_z);
        }();

        if (batch_count > max_grid_z) {
            throw std::invalid_argument("BatchedMatMul: batch count exceeds the grid z limit");
        }

        dim3 block(16, 16);
        dim3 grid(CeilDiv(cols, block.x), CeilDiv(rows, block.y), static_cast<unsigned int>(batch_count));

        MatMulKernel<<<grid, block>>>(A, B, C, layout, rows, inner, cols, a_rs, a_cs, b_rs, b_cs);
        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchMatMulDevice(const T* dA, const T* dB, T* dC, size_t M, size_t N, size_t K){
        LaunchMatMulDevice(dA, dB, dC, {}, {}, {}, M, N, K, N, 1, K, 1);
    }

    template <typename T>
    void LaunchMatMul(const DeviceBuffer<T>& dA, const DeviceBuffer<T>& dB, DeviceBuffer<T>& dC, size_t M, size_t N, size_t K){
        if (dA.Size() < M * N || dB.Size() < N * K || dC.Size() < M * K)
            throw std::invalid_argument("LaunchMatMul: buffer too small for given M, N, K");

        LaunchMatMulDevice(dA.Data(), dB.Data(), dC.Data(), M, N, K);
    }

    template <typename T>
    void LaunchMatMulHost(const T* A, const T* B, T* C, size_t M, size_t N, size_t K){
        DeviceBuffer<T> dA(A, M * N), dB(B, N * K), dC(M * K);
        LaunchMatMulDevice(dA.Data(), dB.Data(), dC.Data(), M, N, K);
        dC.CopyToHost(C, M * K);
    }



    FOR_EACH_FLOAT_TYPE(INSTANTIATE_MATMUL)
}
