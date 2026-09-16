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

        inline std::vector<size_t> BroadcastShape(const std::vector<size_t>& lhs,
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

        inline std::vector<size_t> BroadcastStrides(const std::vector<size_t>& shape,
                                                     const std::vector<size_t>& stride,
                                                     const std::vector<size_t>& target) {
            std::vector<size_t> result(target.size(), 0);
            const size_t pad = target.size() - shape.size();
            for (size_t i = 0; i < shape.size(); ++i) {
                result[pad + i] = shape[i] == 1 ? 0 : stride[i];
            }
            return result;
        }

        inline size_t ElementCount(const std::vector<size_t>& shape) {
            size_t total = 1;
            for (size_t d : shape) {
                total *= d;
            }
            return total;
        }

        inline std::vector<size_t> RowMajorStrides(const std::vector<size_t>& shape) {
            std::vector<size_t> stride(shape.size());
            size_t value = 1;
            for (size_t i = shape.size(); i-- > 0; ) {
                stride[i] = value;
                value *= shape[i];
            }
            return stride;
        }

    }

    template <typename T = float>
    class Tensor {
    public:
        Tensor() = default;

        Tensor(const std::vector<size_t>& shape, T initial_value = T(0))
            : m_shape(shape) {
            for (size_t d : m_shape) {
                if (d == 0) {
                    throw std::invalid_argument("Tensor dimensions must be greater than zero.");
                }
            }
            m_size = detail::ElementCount(m_shape);
            m_stride = detail::RowMajorStrides(m_shape);
            m_data = std::make_shared<std::vector<T>>(m_size, initial_value);
        }

        const std::vector<size_t>& Shape() const noexcept { return m_shape; }
        const std::vector<size_t>& Strides() const noexcept { return m_stride; }
        size_t Rank() const noexcept { return m_shape.size(); }
        size_t Size() const noexcept { return m_data ? m_size : 0; }
        size_t Offset() const noexcept { return m_offset; }

        bool IsContiguous() const noexcept {
            size_t expected = 1;
            for (size_t i = m_shape.size(); i-- > 0; ) {
                if (m_shape[i] != 1 && m_stride[i] != expected) {
                    return false;
                }
                expected *= m_shape[i];
            }
            return true;
        }

        T* Data() noexcept { return m_data ? m_data->data() + m_offset : nullptr; }
        const T* Data() const noexcept { return m_data ? m_data->data() + m_offset : nullptr; }

        bool SharesStorage(const Tensor<T>& other) const noexcept {
            return m_data && m_data == other.m_data;
        }

        long UseCount() const noexcept { return m_data ? m_data.use_count() : 0; }

        void Fill(const T& value) {
            ForEach([&value](T& element) { element = value; });
        }

        void RandomFill(T min_value = T(0), T max_value = T(1)) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dis(min_value, max_value);
            ForEach([&dis](T& element) { element = dis(gen); });
        }

        template <typename... Idx>
        requires (std::is_convertible_v<Idx, size_t> && ...)
        T& operator()(Idx... indices) {
            return (*m_data)[Flatten(indices...)];
        }

        template <typename... Idx>
        requires (std::is_convertible_v<Idx, size_t> && ...)
        const T& operator()(Idx... indices) const {
            return (*m_data)[Flatten(indices...)];
        }

        T& operator()(std::span<const size_t> indices) {
            return (*m_data)[Flatten(indices)];
        }

        const T& operator()(std::span<const size_t> indices) const {
            return (*m_data)[Flatten(indices)];
        }

        Tensor<T> Clone() const {
            if (!m_data) {
                return Tensor<T>();
            }
            Tensor<T> result(m_shape);
            T* out = result.m_data->data();
            size_t i = 0;
            ForEach([&out, &i](const T& element) { out[i++] = element; });
            return result;
        }

        Tensor<T> Reshape(const std::vector<size_t>& new_shape) const {
            if (!m_data) {
                throw std::invalid_argument("Cannot reshape an empty tensor.");
            }
            for (size_t d : new_shape) {
                if (d == 0) {
                    throw std::invalid_argument("Tensor dimensions must be greater than zero.");
                }
            }
            if (detail::ElementCount(new_shape) != m_size) {
                throw std::invalid_argument("Reshape must preserve the total element count.");
            }

            Tensor<T> result;
            if (IsContiguous()) {
                result.m_data = m_data;
                result.m_offset = m_offset;
            } else {
                Tensor<T> compacted = Clone();
                result.m_data = std::move(compacted.m_data);
                result.m_offset = 0;
            }
            result.m_shape = new_shape;
            result.m_stride = detail::RowMajorStrides(new_shape);
            result.m_size = m_size;
            return result;
        }

        template <typename... Dims>
        requires (sizeof...(Dims) > 0 && (std::is_convertible_v<Dims, size_t> && ...))
        Tensor<T> Reshape(Dims... dims) const {
            return Reshape(std::vector<size_t>{ static_cast<size_t>(dims)... });
        }

        Tensor<T> View(const std::vector<size_t>& new_shape) const {
            if (!m_data) {
                throw std::invalid_argument("Cannot create a view of an empty tensor.");
            }
            if (!IsContiguous()) {
                throw std::invalid_argument("View() requires a contiguous tensor; Clone() it first.");
            }
            for (size_t d : new_shape) {
                if (d == 0) {
                    throw std::invalid_argument("Tensor dimensions must be greater than zero.");
                }
            }
            if (detail::ElementCount(new_shape) != m_size) {
                throw std::invalid_argument("View() must preserve the total element count.");
            }

            Tensor<T> result;
            result.m_data = m_data;
            result.m_offset = m_offset;
            result.m_shape = new_shape;
            result.m_stride = detail::RowMajorStrides(new_shape);
            result.m_size = m_size;
            return result;
        }

        template <typename... Dims>
        requires (sizeof...(Dims) > 0 && (std::is_convertible_v<Dims, size_t> && ...))
        Tensor<T> View(Dims... dims) const {
            return View(std::vector<size_t>{ static_cast<size_t>(dims)... });
        }

        Tensor<T> Slice(size_t axis, size_t start, size_t count) const {
            if (!m_data) {
                throw std::invalid_argument("Cannot slice an empty tensor.");
            }
            if (axis >= m_shape.size()) {
                throw std::invalid_argument("Slice() axis is out of range.");
            }
            if (count == 0) {
                throw std::invalid_argument("Slice() count must be greater than zero.");
            }
            if (start > m_shape[axis] || count > m_shape[axis] - start) {
                throw std::invalid_argument("Slice() range is out of bounds.");
            }

            Tensor<T> result = *this;
            result.m_offset = m_offset + start * m_stride[axis];
            result.m_shape[axis] = count;
            result.m_size = detail::ElementCount(result.m_shape);
            return result;
        }

        Tensor<T> Row(size_t index) const {
            if (!m_data) {
                throw std::invalid_argument("Cannot take a row of an empty tensor.");
            }
            if (m_shape.empty()) {
                throw std::invalid_argument("Row() requires at least one dimension.");
            }
            if (index >= m_shape[0]) {
                throw std::invalid_argument("Row() index is out of range.");
            }

            Tensor<T> result;
            result.m_data = m_data;
            result.m_offset = m_offset + index * m_stride[0];
            result.m_shape.assign(m_shape.begin() + 1, m_shape.end());
            result.m_stride.assign(m_stride.begin() + 1, m_stride.end());
            result.m_size = detail::ElementCount(result.m_shape);
            return result;
        }

        Tensor<T> Permute(const std::vector<size_t>& axes) const {
            ValidatePermutation(axes);
            if (!m_data) {
                throw std::invalid_argument("Cannot create a view of an empty tensor.");
            }

            Tensor<T> result;
            result.m_data = m_data;
            result.m_offset = m_offset;
            result.m_shape.resize(axes.size());
            result.m_stride.resize(axes.size());
            for (size_t i = 0; i < axes.size(); ++i) {
                result.m_shape[i] = m_shape[axes[i]];
                result.m_stride[i] = m_stride[axes[i]];
            }
            result.m_size = m_size;
            return result;
        }

        template <typename... Dims>
        requires (sizeof...(Dims) > 0 && (std::is_convertible_v<Dims, size_t> && ...))
        Tensor<T> Permute(Dims... dims) const {
            return Permute(std::vector<size_t>{ static_cast<size_t>(dims)... });
        }

        Tensor<T> Transpose() const {
            return Permute(ReversedAxes());
        }

        Tensor<T> Transpose(size_t dim0, size_t dim1) const {
            return Permute(SwappedAxes(dim0, dim1));
        }

        Tensor<T> Contiguous() const {
            if (!m_data || IsContiguous()) {
                return *this;
            }
            return Clone();
        }


        Tensor<T>& operator+=(const Tensor<T>& other) {
            return BroadcastInto(other, [](T& a, T b) { a += b; });
        }

        Tensor<T>& operator-=(const Tensor<T>& other) {
            return BroadcastInto(other, [](T& a, T b) { a -= b; });
        }

        Tensor<T>& operator*=(const Tensor<T>& other) {
            return BroadcastInto(other, [](T& a, T b) { a *= b; });
        }

        Tensor<T>& operator/=(const Tensor<T>& other) {
            return BroadcastInto(other, [](T& a, T b) {
                if (b == T(0)) {
                    throw std::invalid_argument("Division by zero in tensor.");
                }
                a /= b;
            });
        }

        Tensor<T>& operator*=(const T& scalar) {
            ForEach([&scalar](T& element) { element *= scalar; });
            return *this;
        }

        Tensor<T>& operator/=(const T& scalar) {
            if (scalar == T(0)) {
                throw std::invalid_argument("Division by zero.");
            }
            ForEach([&scalar](T& element) { element /= scalar; });
            return *this;
        }

    private:
        template <typename... Idx>
        size_t Flatten(Idx... indices) const {
            assert(sizeof...(Idx) == m_shape.size() && "wrong number of indices");
            size_t axis = 0;
            size_t offset = m_offset;
            ((assert(static_cast<size_t>(indices) < m_shape[axis]),
              offset += static_cast<size_t>(indices) * m_stride[axis],
              ++axis), ...);
            return offset;
        }

        size_t Flatten(std::span<const size_t> indices) const {
            assert(indices.size() == m_shape.size() && "wrong number of indices");
            size_t offset = m_offset;
            for (size_t i = 0; i < indices.size(); ++i) {
                assert(indices[i] < m_shape[i] && "index out of bounds");
                offset += indices[i] * m_stride[i];
            }
            return offset;
        }

        template <typename F>
        void ForEach(F f) const {
            if (!m_data) {
                return;
            }
            const size_t n = m_shape.size();
            std::vector<size_t> index(n, 0);
            size_t offset = m_offset;

            for (size_t i = 0; i < m_size; ++i) {
                f((*m_data)[offset]);

                for (size_t axis = n; axis-- > 0; ) {
                    ++index[axis];
                    offset += m_stride[axis];
                    if (index[axis] < m_shape[axis]) {
                        break;
                    }
                    index[axis] = 0;
                    offset -= m_stride[axis] * m_shape[axis];
                }
            }
        }

        void ValidatePermutation(const std::vector<size_t>& axes) const {
            const size_t n = m_shape.size();
            if (axes.size() != n) {
                throw std::invalid_argument("Permutation must list every axis exactly once.");
            }
            std::vector<bool> seen(n, false);
            for (size_t axis : axes) {
                if (axis >= n || seen[axis]) {
                    throw std::invalid_argument("Permutation must list every axis exactly once.");
                }
                seen[axis] = true;
            }
        }

        std::vector<size_t> ReversedAxes() const {
            std::vector<size_t> axes(m_shape.size());
            for (size_t i = 0; i < axes.size(); ++i) {
                axes[i] = axes.size() - 1 - i;
            }
            return axes;
        }

        std::vector<size_t> SwappedAxes(size_t dim0, size_t dim1) const {
            const size_t n = m_shape.size();
            if (dim0 >= n || dim1 >= n) {
                throw std::invalid_argument("Transpose dimension is out of range.");
            }
            std::vector<size_t> axes(n);
            for (size_t i = 0; i < n; ++i) {
                axes[i] = i;
            }
            std::swap(axes[dim0], axes[dim1]);
            return axes;
        }

        template <typename Op>
        Tensor<T>& BroadcastInto(const Tensor<T>& other, Op op) {
            if (detail::BroadcastShape(m_shape, other.m_shape) != m_shape) {
                throw std::invalid_argument("In-place operation would change the tensor shape.");
            }
            if (!m_data || !other.m_data) {
                return *this;
            }

            const std::vector<size_t> other_stride =
                detail::BroadcastStrides(other.m_shape, other.m_stride, m_shape);
            const size_t n = m_shape.size();

            std::vector<size_t> index(n, 0);
            size_t self_offset = m_offset;
            size_t other_offset = other.m_offset;

            for (size_t i = 0; i < m_size; ++i) {
                op((*m_data)[self_offset], (*other.m_data)[other_offset]);

                for (size_t axis = n; axis-- > 0; ) {
                    ++index[axis];
                    self_offset += m_stride[axis];
                    other_offset += other_stride[axis];
                    if (index[axis] < m_shape[axis]) {
                        break;
                    }
                    index[axis] = 0;
                    self_offset -= m_stride[axis] * m_shape[axis];
                    other_offset -= other_stride[axis] * m_shape[axis];
                }
            }
            return *this;
        }

        std::vector<size_t> m_shape;
        std::vector<size_t> m_stride;
        std::shared_ptr<std::vector<T>> m_data;
        size_t m_offset = 0;
        size_t m_size = 0;
    };

    template <typename T, typename Op>
    Tensor<T> Zip(const Tensor<T>& lhs, const Tensor<T>& rhs, Op op) {
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

    template <typename T = float>
    Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return Zip(lhs, rhs, [](T a, T b) { return a + b; });
    }

    template <typename T = float>
    Tensor<T> operator-(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return Zip(lhs, rhs, [](T a, T b) { return a - b; });
    }

    template <typename T = float>
    Tensor<T> operator*(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return Zip(lhs, rhs, [](T a, T b) { return a * b; });
    }

    template <typename T = float>
    Tensor<T> operator/(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        return Zip(lhs, rhs, [](T a, T b) {
            if (b == T(0)) {
                throw std::invalid_argument("Division by zero in tensor.");
            }
            return a / b;
        });
    }

    template <typename T = float>
    Tensor<T> operator*(const Tensor<T>& lhs, const T& scalar) {
        Tensor<T> result = lhs.Clone();
        result *= scalar;
        return result;
    }

    template <typename T = float>
    Tensor<T> operator*(const T& scalar, const Tensor<T>& rhs) {
        Tensor<T> result = rhs.Clone();
        result *= scalar;
        return result;
    }

    template <typename T = float>
    Tensor<T> operator/(const Tensor<T>& lhs, const T& scalar) {
        Tensor<T> result = lhs.Clone();
        result /= scalar;
        return result;
    }

    template <typename T = float>
    Tensor<T> MatMul(const Tensor<T>& A, const Tensor<T>& B) {
        if(A.Rank() != 2 || B.Rank() != 2) {
            throw std::invalid_argument("Both tensors must be 2D for matrix multiplication.");
        }
        if (A.Shape()[1] != B.Shape()[0]) {
            throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
        }

        std::vector<size_t> result_shape = { A.Shape()[0], B.Shape()[1] };
        Tensor<T> result(result_shape);

        for (size_t i = 0; i < A.Shape()[0]; ++i) {
            for (size_t j = 0; j < B.Shape()[1]; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < A.Shape()[1]; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                result(i, j) = sum;
            }
        }

        return result;
    }

    template <typename T = float>
    Tensor<T> Transpose(const Tensor<T>& tensor) {
        return tensor.Transpose();
    }

}
