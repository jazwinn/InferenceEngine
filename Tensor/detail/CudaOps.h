#pragma once

#ifdef IE_ENABLE_CUDA

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Tensor/TensorCore.h"
#include "CUDA/Kernel.cuh"

namespace tensor::detail {

    template <typename T>
    Tensor<T> MatMulCUDA(const Tensor<T>& A, const Tensor<T>& B,
                         const std::vector<size_t>& result_shape,
                         size_t rows, size_t inner, size_t cols) {
        if (A.Rank() != 2 || B.Rank() != 2) {
            throw std::logic_error("MatMul (CUDA): only 2D operands are supported.");
        }
        if (!A.IsContiguous() || !B.IsContiguous()) {
            throw std::logic_error("MatMul (CUDA): operands must be contiguous.");
        }
        Tensor<T> result(result_shape, Device::CUDA);
        cuda::LaunchMatMulDevice(A.Data(), B.Data(), result.Data(), rows, inner, cols);
        return result;
    }

}

#endif
