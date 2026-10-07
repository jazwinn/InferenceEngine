#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Tensor/TensorCore.h"
#include "Tensor/detail/Shape.h"
#include "Tensor/detail/CudaOps.h"

namespace tensor {

    template <typename T, typename Op>
    Tensor<T> Zip(const Tensor<T>& lhs, const Tensor<T>& rhs, Op op) {
        lhs.RequireCPU("Element-wise operator");
        rhs.RequireCPU("Element-wise operator");
        const std::vector<size_t> shape = detail::BroadcastShape(lhs.Shape(), rhs.Shape());
        const std::vector<size_t> lhs_stride =
            detail::BroadcastStrides(lhs.Shape(), lhs.Strides(), shape);
        const std::vector<size_t> rhs_stride =
            detail::BroadcastStrides(rhs.Shape(), rhs.Strides(), shape);
        const size_t rank = shape.size();

        Tensor<T> result(shape);
        T* out = result.Data();
        const T* a = lhs.Data();
        const T* b = rhs.Data();

        std::vector<size_t> index(rank, 0);
        size_t lhs_offset = 0;
        size_t rhs_offset = 0;

        for (size_t i = 0; i < result.Size(); ++i) {
            out[i] = op(a[lhs_offset], b[rhs_offset]);

            for (size_t axis = rank; axis-- > 0; ) {
                ++index[axis];
                lhs_offset += lhs_stride[axis];
                rhs_offset += rhs_stride[axis];
                if (index[axis] < shape[axis]) {
                    break;
                }
                index[axis] = 0;
                lhs_offset -= lhs_stride[axis] * shape[axis];
                rhs_offset -= rhs_stride[axis] * shape[axis];
            }
        }
        return result;
    }

    template <typename T >
    Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return Zip(lhs, rhs, [](T a, T b) { return a + b; });
    }

    template <typename T >
    Tensor<T> operator-(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return Zip(lhs, rhs, [](T a, T b) { return a - b; });
    }

    template <typename T >
    Tensor<T> operator*(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return Zip(lhs, rhs, [](T a, T b) { return a * b; });
    }

    template <typename T >
    Tensor<T> operator/(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return Zip(lhs, rhs, [](T a, T b) {
            if (b == T(0)) {
                throw std::invalid_argument("Division by zero in tensor.");
            }
            return a / b;
        });
    }

    template <typename T>
    Tensor<T> operator+(const Tensor<T>& lhs, const T& scalar) {
        Tensor<T> result = lhs.Clone();
        result += scalar;
        return result;
    }

    template <typename T>
    Tensor<T> operator+(const T& scalar, const Tensor<T>& rhs) {
        return rhs + scalar;
    }

    template <typename T>
    Tensor<T> operator-(const Tensor<T>& lhs, const T& scalar) {
        Tensor<T> result = lhs.Clone();
        result -= scalar;
        return result;
    }

    template <typename T>
    Tensor<T> operator-(const T& scalar, const Tensor<T>& rhs) {
        Tensor<T> result = rhs.Clone();
        result *= static_cast<T>(-1);
        result += scalar;
        return result;
    }

    template <typename T >
    Tensor<T> operator*(const Tensor<T>& lhs, const T& scalar) {
        Tensor<T> result = lhs.Clone();
        result *= scalar;
        return result;
    }

    template <typename T >
    Tensor<T> operator*(const T& scalar, const Tensor<T>& rhs) {
        Tensor<T> result = rhs.Clone();
        result *= scalar;
        return result;
    }

    template <typename T>
    Tensor<T> operator/(const Tensor<T>& lhs, const T& scalar) {
        Tensor<T> result = lhs.Clone();
        result /= scalar;
        return result;
    }

    template <typename T>
    Tensor<T> MatMul(const Tensor<T>& A, const Tensor<T>& B) {
        if (A.Rank() < 2 || B.Rank() < 2) {
            throw std::invalid_argument("MatMul: operands must have at least two dimensions.");
        }
        if (A.GetDevice() != B.GetDevice()) {
            throw std::invalid_argument("MatMul: operands are on different devices.");
        }

        const std::vector<size_t>& a_shape = A.Shape();
        const std::vector<size_t>& b_shape = B.Shape();
        const std::vector<size_t>& a_stride = A.Strides();
        const std::vector<size_t>& b_stride = B.Strides();

        const size_t rows = a_shape[a_shape.size() - 2];
        const size_t inner = a_shape[a_shape.size() - 1];
        const size_t cols = b_shape[b_shape.size() - 1];
        if (inner != b_shape[b_shape.size() - 2]) {
            throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
        }

        const std::vector<size_t> a_batch(a_shape.begin(), a_shape.end() - 2);
        const std::vector<size_t> b_batch(b_shape.begin(), b_shape.end() - 2);
        const std::vector<size_t> batch = detail::BroadcastShape(a_batch, b_batch);

        std::vector<size_t> result_shape = batch;
        result_shape.push_back(rows);
        result_shape.push_back(cols);

        if (A.IsCUDA()) {
#ifdef IE_ENABLE_CUDA
            return detail::MatMulCUDA(A, B, result_shape, rows, inner, cols);
#endif
        }

        Tensor<T> result(result_shape);

        const std::vector<size_t> a_batch_stride = detail::BroadcastStrides(
            a_batch, std::vector<size_t>(a_stride.begin(), a_stride.end() - 2), batch);
        const std::vector<size_t> b_batch_stride = detail::BroadcastStrides(
            b_batch, std::vector<size_t>(b_stride.begin(), b_stride.end() - 2), batch);

        const size_t a_row_stride = a_stride[a_shape.size() - 2];
        const size_t a_col_stride = a_stride[a_shape.size() - 1];
        const size_t b_row_stride = b_stride[b_shape.size() - 2];
        const size_t b_col_stride = b_stride[b_shape.size() - 1];

        const T* a_data = A.Data();
        const T* b_data = B.Data();
        T* out = result.Data();

        const size_t batch_rank = batch.size();
        const size_t batch_count = detail::ElementCount(batch);

        std::vector<size_t> index(batch_rank, 0);
        size_t a_offset = 0;
        size_t b_offset = 0;

        for (size_t c = 0; c < batch_count; ++c) {
            T* block = out + c * rows * cols;

            for (size_t i = 0; i < rows; ++i) {
                for (size_t k = 0; k < inner; ++k) {
                    const T a_ik = a_data[a_offset + i * a_row_stride + k * a_col_stride];
                    for (size_t j = 0; j < cols; ++j) {
                        block[i * cols + j] += a_ik * b_data[b_offset + k * b_row_stride + j * b_col_stride];
                    }
                }
            }

            for (size_t ax = batch_rank; ax-- > 0; ) {
                ++index[ax];
                a_offset += a_batch_stride[ax];
                b_offset += b_batch_stride[ax];
                if (index[ax] < batch[ax]) {
                    break;
                }
                index[ax] = 0;
                a_offset -= a_batch_stride[ax] * batch[ax];
                b_offset -= b_batch_stride[ax] * batch[ax];
            }
        }

        return result;
    }

    template <typename T>
    void CopyInto(Tensor<T> dst, const Tensor<T>& src)
    {
        if (dst.Shape() != src.Shape()) {
            throw std::invalid_argument("CopyInto: shape mismatch.");
        }
        if (dst.Size() == 0) return;
        dst.RequireCPU("CopyInto");
        src.RequireCPU("CopyInto");

        const size_t n = dst.Rank();
        const std::vector<size_t>& shape      = dst.Shape();
        const std::vector<size_t>& dst_stride = dst.Strides();
        const std::vector<size_t>& src_stride = src.Strides();

        T* dst_data       = dst.Data();
        const T* src_data = src.Data();

        std::vector<size_t> index(n, 0);
        size_t dst_offset = 0;
        size_t src_offset = 0;

        for (size_t i = 0; i < dst.Size(); ++i) {
            dst_data[dst_offset] = src_data[src_offset];

            for (size_t axis = n; axis-- > 0; ) {
                ++index[axis];
                dst_offset += dst_stride[axis];
                src_offset += src_stride[axis];
                if (index[axis] < shape[axis]) break;
                index[axis]  = 0;
                dst_offset  -= shape[axis] * dst_stride[axis];
                src_offset  -= shape[axis] * src_stride[axis];
            }
        }
    }



    template <typename T>
    Tensor<T> Transpose(const Tensor<T>& tensor) {
        return tensor.Transpose();
    }

    template <typename T>
    Tensor<T> Transpose(const Tensor<T>& tensor, long long dim0, long long dim1) {
        return tensor.Transpose(dim0, dim1);
    }

    template <typename T>
    Tensor<T> Sqrt(const Tensor<T>& tensor) {
        return tensor.Sqrt();
    }

    template <typename T>
    Tensor<T> Mean(const Tensor<T>& tensor) {
        return tensor.Mean();
    }

    template <typename T>
    Tensor<T> Mean(const Tensor<T>& tensor, size_t axis, bool keepdim = false) {
        return tensor.Mean(axis, keepdim);
    }

}
