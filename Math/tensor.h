#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <random>

namespace math {

    namespace detail {

        inline std::vector<size_t> broadcast_shape(const std::vector<size_t>& lhs,
                                                   const std::vector<size_t>& rhs) {
            const size_t rank = std::max(lhs.size(), rhs.size());
            const size_t lhs_pad = rank - lhs.size();
            const size_t rhs_pad = rank - rhs.size();

            std::vector<size_t> shape(rank);
            for (size_t i = 0; i < rank; ++i) {
                const size_t l = i < lhs_pad ? 1 : lhs[i - lhs_pad];
                const size_t r = i < rhs_pad ? 1 : rhs[i - rhs_pad];
                if (l != r && l != 1 && r != 1) {
                    throw std::invalid_argument("Tensor shapes are not broadcast-compatible.");
                }
                shape[i] = std::max(l, r);
            }
            return shape;
        }

        inline std::vector<size_t> broadcast_strides(const std::vector<size_t>& shape,
                                                     const std::vector<size_t>& stride,
                                                     const std::vector<size_t>& target) {
            std::vector<size_t> result(target.size(), 0);
            const size_t pad = target.size() - shape.size();
            for (size_t i = 0; i < shape.size(); ++i) {
                result[pad + i] = shape[i] == 1 ? 0 : stride[i];
            }
            return result;
        }

    }

    template <typename T = float>
    class Tensor {
    public:
        Tensor() = default;

        Tensor(const std::vector<size_t>& shape, T initial_value = T(0))
            : m_shape(shape) {
            size_t total_size = 1;
            for (size_t d : m_shape) {
                if (d == 0) {
                    throw std::invalid_argument("Tensor dimensions must be greater than zero.");
                }
                total_size *= d;
            }
            m_data = std::make_shared<std::vector<T>>(total_size, initial_value);
            m_stride.resize(m_shape.size());
            size_t stride_value = 1;
            for (size_t i = m_shape.size(); i-- > 0; ) {
                m_stride[i] = stride_value;
                stride_value *= m_shape[i];
            }
        }


        void fill(const T& value) {
            if (!m_data) {
                return;
            }
            std::fill(m_data->begin(), m_data->end(), value);
        }
        void random_fill(T min_value = T(0), T max_value = T(1)) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dis(min_value, max_value);
            if (!m_data) {
                return;
            }
            std::generate(m_data->begin(), m_data->end(), [&dis]() {
                return dis(gen);
            });
        }

        const std::vector<size_t>& shape() const noexcept { return m_shape; }
        const std::vector<size_t>& strides() const noexcept { return m_stride; }
        size_t rank() const noexcept { return m_shape.size(); }
        size_t size() const noexcept { return m_data ? m_data->size() : 0; }


        template <typename... Idx>
        requires (std::is_convertible_v<Idx, size_t> && ...)
        T& operator()(Idx... indices) {
            return (*m_data)[flatten(indices...)];
        }

        template <typename... Idx>
        requires (std::is_convertible_v<Idx, size_t> && ...)
        const T& operator()(Idx... indices) const {
            return (*m_data)[flatten(indices...)];
        }

        T& operator()(std::span<const size_t> indices) {
            return (*m_data)[flatten(indices)];
        }

        const T& operator()(std::span<const size_t> indices) const {
            return (*m_data)[flatten(indices)];
        }

        T* data() noexcept { return m_data ? m_data->data() : nullptr; }
        const T* data() const noexcept { return m_data ? m_data->data() : nullptr; }

        bool shares_storage(const Tensor<T>& other) const noexcept {
            return m_data && m_data == other.m_data;
        }

        long use_count() const noexcept { return m_data ? m_data.use_count() : 0; }

        Tensor<T> clone() const {
            Tensor<T> copy;
            copy.m_shape = m_shape;
            copy.m_stride = m_stride;
            copy.m_data = m_data ? std::make_shared<std::vector<T>>(*m_data) : nullptr;
            return copy;
        }

        Tensor<T>& operator+=(const Tensor<T>& other) {
            return broadcast_into(other, [](T& a, T b) { a += b; });
        }

        Tensor<T>& operator-=(const Tensor<T>& other) {
            return broadcast_into(other, [](T& a, T b) { a -= b; });
        }

        Tensor<T>& operator*=(const Tensor<T>& other) {
            return broadcast_into(other, [](T& a, T b) { a *= b; });
        }

        Tensor<T>& operator/=(const Tensor<T>& other) {
            return broadcast_into(other, [](T& a, T b) {
                if (b == T(0)) {
                    throw std::invalid_argument("Division by zero in tensor.");
                }
                a /= b;
            });
        }

        Tensor<T>& operator*=(const T& scalar) {
            for (size_t i = 0; i < size(); ++i) {
                (*m_data)[i] *= scalar;
            }
            return *this;
        }

        Tensor<T>& operator/=(const T& scalar) {
            if (scalar == T(0)) {
                throw std::invalid_argument("Division by zero.");
            }
            for (size_t i = 0; i < size(); ++i) {
                (*m_data)[i] /= scalar;
            }
            return *this;
        }

        Tensor<T> transpose() const {
            if (m_shape.size() != 2) {
                throw std::invalid_argument("Transpose is only defined for 2D tensors.");
            }
            std::vector<size_t> new_shape = { m_shape[1], m_shape[0] };
            Tensor<T> transposed(new_shape);
            for (size_t i = 0; i < m_shape[0]; ++i) {
                for (size_t j = 0; j < m_shape[1]; ++j) {
                    transposed(j, i) = (*this)(i, j);
                }
            }
            return transposed;
        }

    private:
        template <typename... Idx>
        size_t flatten(Idx... indices) const {
            assert(sizeof...(Idx) == m_shape.size() && "wrong number of indices");
            size_t axis = 0;
            size_t offset = 0;
            ((assert(static_cast<size_t>(indices) < m_shape[axis]),
              offset += static_cast<size_t>(indices) * m_stride[axis],
              ++axis), ...);
            return offset;
        }

        size_t flatten(std::span<const size_t> indices) const {
            assert(indices.size() == m_shape.size() && "wrong number of indices");
            size_t offset = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
                assert(indices[i] < m_shape[i] && "index out of bounds");
                offset += indices[i] * m_stride[i];
            }
            return offset;
        }

        template <typename Op>
        Tensor<T>& broadcast_into(const Tensor<T>& other, Op op) {
            if (detail::broadcast_shape(m_shape, other.m_shape) != m_shape) {
                throw std::invalid_argument("In-place operation would change the tensor shape.");
            }

            const std::vector<size_t> stride =
                detail::broadcast_strides(other.m_shape, other.m_stride, m_shape);
            const size_t rank = m_shape.size();

            std::vector<size_t> index(rank, 0);
            size_t offset = 0;

            for (size_t i = 0; i < size(); ++i) {
                op((*m_data)[i], (*other.m_data)[offset]);

                for (size_t axis = rank; axis-- > 0; ) {
                    ++index[axis];
                    offset += stride[axis];
                    if (index[axis] < m_shape[axis]) {
                        break;
                    }
                    index[axis] = 0;
                    offset -= stride[axis] * m_shape[axis];
                }
            }
            return *this;
        }

        std::vector<size_t> m_shape;
        std::vector<size_t> m_stride;
        std::shared_ptr<std::vector<T>> m_data;
    };

    template <typename T, typename Op>
    Tensor<T> zip(const Tensor<T>& lhs, const Tensor<T>& rhs, Op op) {
        const std::vector<size_t> shape = detail::broadcast_shape(lhs.shape(), rhs.shape());
        const std::vector<size_t> lhs_stride =
            detail::broadcast_strides(lhs.shape(), lhs.strides(), shape);
        const std::vector<size_t> rhs_stride =
            detail::broadcast_strides(rhs.shape(), rhs.strides(), shape);
        const size_t rank = shape.size();

        Tensor<T> result(shape);
        T* out = result.data();
        const T* a = lhs.data();
        const T* b = rhs.data();

        std::vector<size_t> index(rank, 0);
        size_t lhs_offset = 0;
        size_t rhs_offset = 0;

        for (size_t i = 0; i < result.size(); ++i) {
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

    template <typename T = float>
    Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return zip(lhs, rhs, [](T a, T b) { return a + b; });
    }

    template <typename T = float>
    Tensor<T> operator-(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return zip(lhs, rhs, [](T a, T b) { return a - b; });
    }

    template <typename T = float>
    Tensor<T> operator*(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return zip(lhs, rhs, [](T a, T b) { return a * b; });
    }

    template <typename T = float>
    Tensor<T> operator/(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return zip(lhs, rhs, [](T a, T b) {
            if (b == T(0)) {
                throw std::invalid_argument("Division by zero in tensor.");
            }
            return a / b;
        });
    }

    template <typename T = float>
    Tensor<T> operator*(const Tensor<T>& lhs, const T& scalar) {
        Tensor<T> result = lhs.clone();
        result *= scalar;
        return result;
    }

    template <typename T = float>
    Tensor<T> operator*(const T& scalar, const Tensor<T>& rhs) {
        Tensor<T> result = rhs.clone();
        result *= scalar;
        return result;
    }

    template <typename T = float>
    Tensor<T> operator/(const Tensor<T>& lhs, const T& scalar) {
        Tensor<T> result = lhs.clone();
        result /= scalar;
        return result;
    }

    template <typename T = float>
    Tensor<T> MatMul(const Tensor<T>& A, const Tensor<T>& B) {
        if(A.shape().size() != 2 || B.shape().size() != 2) {
            throw std::invalid_argument("Both tensors must be 2D for matrix multiplication.");
        }
        if (A.shape()[1] != B.shape()[0]) {
            throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
        }

        std::vector<size_t> result_shape = { A.shape()[0], B.shape()[1] };
        Tensor<T> result(result_shape);

        for (size_t i = 0; i < A.shape()[0]; ++i) {
            for (size_t j = 0; j < B.shape()[1]; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < A.shape()[1]; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                result(i, j) = sum;
            }
        }

        return result;
    }

    template <typename T = float>
    Tensor<T> transpose(const Tensor<T>& tensor) {
        return tensor.transpose();
    }

}
