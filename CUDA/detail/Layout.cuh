#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "CUDA/detail/common.cuh"

namespace cuda{

    template <int N>
    struct StridedLayout {
        size_t shape[kMaxRank];
        size_t stride[N][kMaxRank];
        int rank;
    };

    template <int N>
    __device__ inline void ViewOffsets(const StridedLayout<N>& layout, size_t i, size_t (&out)[N]){
        #pragma unroll
        for (int k = 0; k < N; ++k) out[k] = 0;

        for (int ax = layout.rank - 1; ax >= 0; --ax) {
            const size_t coord = i % layout.shape[ax];
            i /= layout.shape[ax];
            #pragma unroll
            for (int k = 0; k < N; ++k) out[k] += coord * layout.stride[k][ax];
        }
    }

    template <typename... Strides>
    StridedLayout<sizeof...(Strides)> MakeStridedLayout(const char* name,
                                                        size_t& size,
                                                        const std::vector<size_t>& shape,
                                                        const Strides&... strides)
    {
        constexpr int N = static_cast<int>(sizeof...(Strides));

        if(shape.size() > kMaxRank){
            throw std::invalid_argument(std::string(name) + ": Shape dimension exceed kMaxRank");
        }
        if(((strides.size() != shape.size()) || ...)){
            throw std::invalid_argument(std::string(name) + ": stride rank does not match shape rank");
        }

        StridedLayout<N> layout{};
        layout.rank = static_cast<int>(shape.size());

        size = 1;
        for(int n = 0; n < layout.rank; ++n){
            layout.shape[n] = shape[n];
            size *= shape[n];
        }

        int k = 0;
        ((std::copy(strides.begin(), strides.end(), layout.stride[k++])), ...);

        return layout;
    }

}
