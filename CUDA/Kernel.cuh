#pragma once
#include <cstddef>
#include <vector>

#include "CUDA/Memory.h"

namespace cuda{

    enum class Activation { GELU, GELUTanh, SiLU, ReLU, Sigmoid, Tanh };
    enum class BinaryOp { Add, Sub, Mul, Div };
    enum class UnaryOp { Sqrt };

    template <typename T>
    void LaunchMatMul(const DeviceBuffer<T>& dA, const DeviceBuffer<T>& dB, DeviceBuffer<T>& dC, size_t M, size_t N, size_t K);

    template <typename T>
    void LaunchMatMulDevice(const T* A, const T* B, T* C,
                                const std::vector<size_t>& batch,
                                const std::vector<size_t>& a_batch_stride,
                                const std::vector<size_t>& b_batch_stride,
                                size_t rows, size_t inner, size_t cols,
                                size_t a_rs, size_t a_cs, size_t b_rs, size_t b_cs);
                                
    template <typename T>
    void LaunchMatMulDevice(const T* dA, const T* dB, T* dC, size_t M, size_t N, size_t K);

    template <typename T>
    void LaunchMatMulHost(const T* A, const T* B, T* C, size_t M, size_t N, size_t K);

    template <typename T>
    void LaunchLinear(const DeviceBuffer<T>& dX, const DeviceBuffer<T>& dW, const DeviceBuffer<T>& dB, DeviceBuffer<T>& dY, size_t M, size_t in, size_t out);

    template <typename T>
    void LaunchLinearDevice(const T* dX, const T* dW, const T* dB, T* dY, size_t M, size_t in, size_t out);

    template <typename T>
    void LaunchLinearHost(const T* X, const T* W, const T* B, T* Y, size_t M, size_t in, size_t out);

    template <typename T>
    void LaunchStridedCopyDevice(const T* src, T* dst,
                                 const std::vector<size_t>& shape,
                                 const std::vector<size_t>& src_stride,
                                 const std::vector<size_t>& dst_stride);

    template <typename T, typename Index>
    void LaunchEmbeddingLookUpDevice(const Index* ids, T* out, const T* table, size_t num_ids, size_t rows, size_t dim);

    template <typename T>
    void LaunchActivationDevice(const T* in, T* out, size_t size, Activation act);

    template <typename T>
    void LaunchUnaryDevice(const T* in, T* out, size_t size, UnaryOp op);

    template <typename T>
    void LaunchScalarDevice(const T* in, T* out, T scalar, size_t size, BinaryOp op);

    template <typename T>
    void LaunchBinaryDevice(const T* A, const T* B, T* out, size_t size, BinaryOp op);

    template <typename T>
    void LaunchStridedBinaryDevice(const T* A, const T* B, T* out,
                                   const std::vector<size_t>& shape,
                                   const std::vector<size_t>& a_stride,
                                   const std::vector<size_t>& b_stride,
                                   const std::vector<size_t>& out_stride,
                                   BinaryOp op);

    template <typename T>
    void LaunchFillDevice(T* out, T value, size_t size);

    template <typename T>
    void LaunchStridedScalarDevice(T* out, T scalar,
                                   const std::vector<size_t>& shape,
                                   const std::vector<size_t>& stride,
                                   BinaryOp op);

    template <typename T>
    void LaunchStridedFillDevice(T* out, T value,
                                 const std::vector<size_t>& shape,
                                 const std::vector<size_t>& stride);

    template <typename T>
    void LaunchMaskedFillDevice(const T* in, T* out, const bool* mask, T value, size_t size);

    template <typename T>
    void LaunchStridedMaskedFillDevice(const T* in, T* out, const bool* mask, T value,
                                       const std::vector<size_t>& shape,
                                       const std::vector<size_t>& mask_stride);

    template <typename T>
    void LaunchSumDevice(const T* in, T* out, size_t size);

    template <typename T>
    void LaunchMeanAxisDevice(const T* in, T* out, size_t outer, size_t reduced, size_t inner);

    template <typename T>
    void LaunchSoftMaxDevice(const T* in, T* out, size_t outer, size_t reduced, size_t inner);

    template <typename T>
    void LaunchRoPEDevice(const T* in, T* out, const T* cos, const T* sin,
                          size_t rows, size_t length, size_t half, size_t start_pos);
}