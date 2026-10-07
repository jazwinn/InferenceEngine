#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tensor::detail {

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
