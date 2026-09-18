#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
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

        inline size_t NormalizeAxis(long long axis, size_t rank) {
            const long long signed_rank = static_cast<long long>(rank);
            const long long normalized = axis < 0 ? axis + signed_rank : axis;
            if (normalized < 0 || normalized >= signed_rank) {
                throw std::invalid_argument("Axis is out of range for this tensor.");
            }
            return static_cast<size_t>(normalized);
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

    template <typename T >
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
            m_data = std::make_shared<T[]>(m_size, initial_value);
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

        T* Data() noexcept { return m_data ? m_data.get() + m_offset : nullptr; }
        const T* Data() const noexcept { return m_data ? m_data.get() + m_offset : nullptr; }

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
            return m_data[Flatten(indices...)];
        }

        template <typename... Idx>
        requires (std::is_convertible_v<Idx, size_t> && ...)
        const T& operator()(Idx... indices) const {
            return m_data[Flatten(indices...)];
        }

        T& operator()(std::span<const size_t> indices) {
            return m_data[Flatten(indices)];
        }

        const T& operator()(std::span<const size_t> indices) const {
            return m_data[Flatten(indices)];
        }

        Tensor<T> Clone() const {
            if (!m_data) {
                return Tensor<T>();
            }
            Tensor<T> result(m_shape);
            T* out = result.m_data.get();
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

        Tensor<T> Transpose(long long dim0, long long dim1) const {
            const size_t rank = m_shape.size();
            return Permute(SwappedAxes(detail::NormalizeAxis(dim0, rank),
                                       detail::NormalizeAxis(dim1, rank)));
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

        Tensor<T>& operator+=(const T& scalar) {
            ForEach([&scalar](T& element) { element += scalar; });
            return *this;
        }

        Tensor<T>& operator-=(const T& scalar) {
            ForEach([&scalar](T& element) { element -= scalar; });
            return *this;
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

        Tensor<T> Sqrt() const {
            Tensor<T> result = Clone();
            result.ForEach([](T& element) { element = std::sqrt(element); });
            return result;
        }

        Tensor<T> Mean() const {
            if (Size() == 0) {
                throw std::invalid_argument("Cannot compute mean of an empty tensor.");
            }
            T sum = T(0);
            ForEach([&sum](const T& element) { sum += element; });
            return Tensor<T>({1}, sum / static_cast<T>(Size()));
        }

        Tensor<T> Mean(size_t axis, bool keepdim = false) const {
            if (axis >= m_shape.size()) {
                throw std::invalid_argument("Axis is out of bounds for Mean operation.");
            }

            const size_t n = m_shape.size();
            const size_t reduced = m_shape[axis];

            std::vector<size_t> new_shape = m_shape;
            if (keepdim) {
                new_shape[axis] = 1;
            } else {
                new_shape.erase(new_shape.begin() + axis);
            }

            Tensor<T> result(new_shape);

            const std::vector<size_t> result_stride = detail::RowMajorStrides(new_shape);
            std::vector<size_t> out_stride(n, 0);
            for (size_t i = 0, k = 0; i < n; ++i) {
                if (i == axis) {
                    if (keepdim) {
                        ++k;
                    }
                } else {
                    out_stride[i] = result_stride[k++];
                }
            }

            const T* in = Data();
            T* out = result.Data();
            std::vector<size_t> index(n, 0);
            size_t in_offset = 0;
            size_t out_offset = 0;

            for (size_t c = 0; c < m_size; ++c) {
                out[out_offset] += in[in_offset];

                for (size_t ax = n; ax-- > 0; ) {
                    ++index[ax];
                    in_offset += m_stride[ax];
                    out_offset += out_stride[ax];
                    if (index[ax] < m_shape[ax]) {
                        break;
                    }
                    index[ax] = 0;
                    in_offset -= m_stride[ax] * m_shape[ax];
                    out_offset -= out_stride[ax] * m_shape[ax];
                }
            }

            for (size_t i = 0; i < result.Size(); ++i) {
                out[i] /= static_cast<T>(reduced);
            }

            return result;
        }

        Tensor<T> MaskedFill(const Tensor<bool>& mask, const T& value) const {
            if (mask.Size() == 0) {
                throw std::invalid_argument("MaskedFill requires a non-empty mask.");
            }
            if (detail::BroadcastShape(m_shape, mask.Shape()) != m_shape) {
                throw std::invalid_argument("Mask shape is not broadcast-compatible with the tensor.");
            }

            // Clone() holds this tensor compacted in row-major order, so the
            // output walks flat while the mask walks its own broadcast strides.
            Tensor<T> result = Clone();
            T* out = result.Data();
            const bool* msk = mask.Data();
            const std::vector<size_t> mask_stride =
                detail::BroadcastStrides(mask.Shape(), mask.Strides(), m_shape);

            const size_t n = m_shape.size();
            std::vector<size_t> index(n, 0);
            size_t mask_offset = 0;

            for (size_t i = 0; i < m_size; ++i) {
                if (msk[mask_offset]) {
                    out[i] = value;
                }

                for (size_t ax = n; ax-- > 0; ) {
                    ++index[ax];
                    mask_offset += mask_stride[ax];
                    if (index[ax] < m_shape[ax]) {
                        break;
                    }
                    index[ax] = 0;
                    mask_offset -= mask_stride[ax] * m_shape[ax];
                }
            }

            return result;
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
                f(m_data[offset]);

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
                op(m_data[self_offset], other.m_data[other_offset]);

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
        std::shared_ptr<T[]> m_data;
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
            throw std::invalid_argument("MatMul operands must have at least two dimensions.");
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
