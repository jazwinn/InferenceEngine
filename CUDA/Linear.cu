#include <limits>
#include <stdexcept>
#include <vector>

#include "CUDA/Kernel.cuh"
#include "CUDA/Memory.h"
#include "CUDA/detail/common.cuh"
#include "CUDA/detail/Gemm.cuh"
#include "CUDA/detail/Layout.cuh"


#define INSTANTIATE_LINEAR(T) \
    template void LaunchLinearDevice<T>(const T*, const T*, const T*, T*, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t, size_t, size_t, size_t, size_t, size_t); \
    template void LaunchLinearDevice<T>(const T*, const T*, const T*, T*, size_t, size_t, size_t); \
    template void LaunchLinear<T>(const DeviceBuffer<T>&, const DeviceBuffer<T>&, const DeviceBuffer<T>&, DeviceBuffer<T>&, size_t, size_t, size_t); \
    template void LaunchLinearHost<T>(const T*, const T*, const T*, T*, size_t, size_t, size_t);

namespace cuda{

    template <typename T>
    __global__ void LinearKernel(const T* X, const T* W, const T* B, T* C, StridedLayout<2> layout,
                                 size_t rows, size_t inner, size_t cols,
                                 size_t X_rs, size_t X_cs, size_t W_rs, size_t W_cs){

        const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t z = blockIdx.z;

        if (row >= rows || col >= cols) return;

        size_t off[2];
        ViewOffsets(layout, z, off);

        const T* x = X + off[0] + row * X_rs;
        const T* w = W + off[1] + col * W_cs;

        T sum = B ? B[col] : T(0);
        for (size_t n = 0; n < inner; ++n) {
            sum += x[n * X_cs] * w[n * W_rs];
        }

        C[z * rows * cols + row * cols + col] = sum;
    }

    template <typename T>
    void LaunchLinearDevice(const T* X, const T* W, const T* B, T* C,
                            const std::vector<size_t>& batch,
                            const std::vector<size_t>& X_batch_stride,
                            const std::vector<size_t>& W_batch_stride,
                            size_t rows, size_t inner, size_t cols,
                            size_t X_rs, size_t X_cs, size_t W_rs, size_t W_cs){

        size_t batch_count = 0;
        const StridedLayout<2> layout = MakeStridedLayout("BatchedLinear", batch_count, batch, X_batch_stride, W_batch_stride);

        if (batch_count == 0 || rows == 0 || cols == 0) return;

        constexpr size_t kMax = static_cast<size_t>(std::numeric_limits<int>::max());
        if (batch_count == 1 && rows <= kMax && inner <= kMax && cols <= kMax) {
            const GemmOperand w = ToGemmOperand(W_rs, W_cs, inner, cols);
            const GemmOperand x = ToGemmOperand(X_rs, X_cs, rows, inner);
            if (w.ok && x.ok) {
                if (B) {
                    LaunchStridedCopyDevice(B, C, {rows, cols}, {0, 1}, {cols, 1});
                }

                Gemm(w.op, x.op,
                     static_cast<int>(cols), static_cast<int>(rows), static_cast<int>(inner),
                     W, w.ld, X, x.ld, C, static_cast<int>(cols),
                     B ? 1.0 : 0.0);
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
            throw std::invalid_argument("BatchedLinear: batch count exceeds the grid z limit");
        }

        dim3 block(16, 16);
        dim3 grid(CeilDiv(cols, block.x), CeilDiv(rows, block.y), static_cast<unsigned int>(batch_count));

        LinearKernel<<<grid, block>>>(X, W, B, C, layout, rows, inner, cols, X_rs, X_cs, W_rs, W_cs);
        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void LaunchLinearDevice(const T* dX, const T* dW, const T* dB, T* dY, size_t M, size_t in, size_t out){
        LaunchLinearDevice(dX, dW, dB, dY, {}, {}, {}, M, in, out, in, 1, 1, in);
    }

    template <typename T>
    void LaunchLinear(const DeviceBuffer<T>& dX,
                      const DeviceBuffer<T>& dW,
                      const DeviceBuffer<T>& dB,
                      DeviceBuffer<T>& dY,
                      size_t M, size_t in, size_t out){
        if (dX.Size() < M * in || dW.Size() < out * in || dY.Size() < M * out ||
            (dB.Size() != 0 && dB.Size() < out))
            throw std::invalid_argument("LaunchLinear: buffer too small for given M, in, out");

        LaunchLinearDevice(dX.Data(), dW.Data(), dB.Size() ? dB.Data() : nullptr, dY.Data(), M, in, out);
    }

    template <typename T>
    void LaunchLinearHost(const T* X, const T* W, const T* B, T* Y, size_t M, size_t in, size_t out){
        DeviceBuffer<T> dX(X, M * in), dW(W, out * in), dY(M * out);
        DeviceBuffer<T> dB = B ? DeviceBuffer<T>(B, out) : DeviceBuffer<T>();
        LaunchLinearDevice(dX.Data(), dW.Data(), dB.Data(), dY.Data(), M, in, out);
        dY.CopyToHost(Y, M * out);
    }

    FOR_EACH_FLOAT_TYPE(INSTANTIATE_LINEAR)

}
