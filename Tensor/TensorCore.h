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
#include <string>

#include "Tensor/detail/Shape.h"

#ifdef IE_ENABLE_CUDA
#include "CUDA/Kernel.cuh"
#include "CUDA/Memory.h"
#endif

namespace tensor {

    enum class Device{CPU, CUDA};

    template <typename T >
    class Tensor {
    public:
        Tensor() = default;

        Tensor(const std::vector<size_t>& shape, Device device):
            m_device(device),
            m_shape(shape) 
        {
            for (size_t d : m_shape) {
                if (d == 0) {
                    throw std::invalid_argument("Tensor dimensions must be greater than zero.");
                }
            }
            m_size = detail::ElementCount(m_shape);
            m_stride = detail::RowMajorStrides(m_shape);
            m_data = Allocate(m_size, m_device);
        }

        Tensor(const std::vector<size_t>& shape, T initial_value = T(0), Device device = Device::CPU):
            Tensor(shape, device == Device::CPU || initial_value == T(0) ? device : Device::CPU)
        {
            if (initial_value != T(0)) {
                std::fill_n(m_data.get(), m_size, initial_value);
            }
            if (device != m_device) {
                *this = To(device);
            }
        }

        const std::vector<size_t>& Shape() const noexcept { return m_shape; }
        const std::vector<size_t>& Strides() const noexcept { return m_stride; }
        size_t Rank() const noexcept { return m_shape.size(); }
        size_t Size() const noexcept { return m_data ? m_size : 0; }
        size_t Offset() const noexcept { return m_offset; }
        Device GetDevice() const noexcept {return m_device;}
        bool IsCUDA()const noexcept{return m_device == Device::CUDA;}

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
            RequireCPU("operator()");
            return m_data[Flatten(indices...)];
        }

        template <typename... Idx>
        requires (std::is_convertible_v<Idx, size_t> && ...)
        const T& operator()(Idx... indices) const {
            RequireCPU("operator()");
            return m_data[Flatten(indices...)];
        }

        T& operator()(std::span<const size_t> indices) {
            RequireCPU("operator()");
            return m_data[Flatten(indices)];
        }

        const T& operator()(std::span<const size_t> indices) const {
            RequireCPU("operator()");
            return m_data[Flatten(indices)];
        }

        Tensor<T> Clone() const {
            if (!m_data) {
                return Tensor<T>();
            }

            Tensor<T> result(m_shape, m_device);

        #ifdef IE_ENABLE_CUDA
            if (IsCUDA()) {
                if (IsContiguous()) {
                    cuda::CopyDeviceToDevice(result.Data(), Data(), m_size * sizeof(T));
                } else {
                    cuda::LaunchStridedCopyDevice(Data(), result.Data(), m_shape, m_stride, result.m_stride);
                }
                return result;
            }
        #endif

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
            result.m_device = m_device;
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
            result.m_device = m_device;
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
            result.m_device = m_device;
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
            result.m_device = m_device;
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

        Tensor<T> Unsqueeze(size_t axis) const {
            if (!m_data) {
                throw std::invalid_argument("Cannot unsqueeze an empty tensor.");
            }
            if (axis > m_shape.size()) {
                throw std::invalid_argument("Unsqueeze() axis is out of range.");
            }

            Tensor<T> result = *this;
            const size_t stride = axis < m_shape.size() ? m_stride[axis] * m_shape[axis] : 1;
            result.m_shape.insert(result.m_shape.begin() + axis, 1);
            result.m_stride.insert(result.m_stride.begin() + axis, stride);
            return result;        
        }

        Tensor<T> Squeeze(size_t axis) const {
            if (!m_data) {
                throw std::invalid_argument("Cannot squeeze an empty tensor.");
            }
            if (axis >= m_shape.size()) {
                throw std::invalid_argument("Squeeze() axis is out of range.");
            }
            if (m_shape[axis] != 1) {
                throw std::invalid_argument("Squeeze() axis must have size 1.");
            }
            if (m_shape.size() == 1) {
                throw std::invalid_argument("Squeeze() cannot remove the only axis.");
            }

            Tensor<T> result = *this;
            result.m_shape.erase(result.m_shape.begin() + axis);
            result.m_stride.erase(result.m_stride.begin() + axis);
            return result;
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
#ifdef IE_ENABLE_CUDA
            if (IsCUDA())
            {
                if (!IsContiguous()) {
                    throw std::logic_error("operator+= (CUDA): tensor must be contiguous.");
                }
                cuda::LaunchScalarDevice(Data(), Data(), scalar, Size(), cuda::BinaryOp::Add);
                return *this;
            }
            
#endif
            ForEach([&scalar](T& element) { element += scalar; });
            return *this;
        }

        Tensor<T>& operator-=(const T& scalar) {
#ifdef IE_ENABLE_CUDA
            if (IsCUDA())
            {
                if (!IsContiguous()) {
                    throw std::logic_error("operator-= (CUDA): tensor must be contiguous.");
                }
                cuda::LaunchScalarDevice(Data(), Data(), scalar, Size(), cuda::BinaryOp::Sub);
                return *this;
            }
            
#endif
            ForEach([&scalar](T& element) { element -= scalar; });
            return *this;
        }

        Tensor<T>& operator*=(const T& scalar) {
#ifdef IE_ENABLE_CUDA
            if (IsCUDA())
            {
                if (!IsContiguous()) {
                    throw std::logic_error("operator*= (CUDA): tensor must be contiguous.");
                }
                cuda::LaunchScalarDevice(Data(), Data(), scalar, Size(), cuda::BinaryOp::Mul);
                return *this;
            }
            
#endif
            ForEach([&scalar](T& element) { element *= scalar; });
            return *this;
        }

        Tensor<T>& operator/=(const T& scalar) {
            if (scalar == T(0)) {
                throw std::invalid_argument("Division by zero.");
            }
#ifdef IE_ENABLE_CUDA
            if (IsCUDA())
            {
                if (!IsContiguous()) {
                    throw std::logic_error("operator/= (CUDA): tensor must be contiguous.");
                }
                cuda::LaunchScalarDevice(Data(), Data(), scalar, Size(), cuda::BinaryOp::Div);
                return *this;
            }

#endif
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

            RequireCPU("Mean");
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
            RequireCPU("MaskedFill");
            mask.RequireCPU("MaskedFill");

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

        Tensor<T> To(Device target) const {
            if (!m_data || target == m_device) {
                return *this;
            }
#ifdef IE_ENABLE_CUDA
            const Tensor<T> src = Contiguous();
            Tensor<T> result(m_shape, target);
            const size_t bytes = m_size * sizeof(T);
            if (target == Device::CUDA) {
                cuda::CopyHostToDevice(result.Data(), src.Data(), bytes);
            } else {
                cuda::CopyDeviceToHost(result.Data(), src.Data(), bytes);
            }
            return result;
#else
            throw std::logic_error("To(): built without CUDA support.");
#endif
        }

        void RequireCPU(const char* op) const {
            if (IsCUDA()) {
                throw std::logic_error(std::string(op) + " is CPU-only; call To(Device::CPU) first.");
            }
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

        static std::shared_ptr<T[]> Allocate(size_t count, Device device) {
            if (device == Device::CPU) {
                return std::make_shared<T[]>(count);
            }
#ifdef IE_ENABLE_CUDA
            auto owner = std::make_shared<cuda::DeviceBuffer<T>>(count);
            owner->Memset(0, count);
            T* ptr = owner->Data();
            return std::shared_ptr<T[]>(owner, ptr);
#else
            throw std::logic_error("Tensor: built without CUDA support.");
#endif
        }

        template <typename F>
        void ForEach(F f) const {
            if (!m_data) {
                return;
            }
            RequireCPU("ForEach");
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
            RequireCPU("In-place operator");
            other.RequireCPU("In-place operator");

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

        Device m_device = Device::CPU;
        std::vector<size_t> m_shape;
        std::vector<size_t> m_stride;
        std::shared_ptr<T[]> m_data;
        size_t m_offset = 0;
        size_t m_size = 0;
    };

}
