#pragma once

#ifdef IE_ENABLE_CUDA

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Tensor/TensorCore.h"
#include "Tensor/ActivationType.h"
#include "CUDA/Kernel.cuh"

namespace tensor::detail {

    inline cuda::Activation ActivationToCuda(Activation kind) {
        switch (kind) {
        case Activation::ReLU:     return cuda::Activation::ReLU;
        case Activation::GELU:     return cuda::Activation::GELU;
        case Activation::GELUTanh: return cuda::Activation::GELUTanh;
        case Activation::Sigmoid:  return cuda::Activation::Sigmoid;
        case Activation::SiLU:     return cuda::Activation::SiLU;
        case Activation::Tanh:     return cuda::Activation::Tanh;
        }
        throw std::invalid_argument("ToCuda: unsupported activation.");
    }

    template <typename T>
    Tensor<T> MatMulCUDA(const Tensor<T>& A, const Tensor<T>& B,
                         const std::vector<size_t>& result_shape,
                         size_t rows, size_t inner, size_t cols) {
        if (!A.IsCUDA() || !B.IsCUDA()) {
            throw std::invalid_argument("MatMul (CUDA): operands must be on CUDA.");
        }
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

    template <typename T>
    Tensor<T> LinearCUDA(const Tensor<T>& X, const Tensor<T>& W, const Tensor<T>& B) {
        if (X.Size() == 0 || X.Rank() < 1) {
            throw std::invalid_argument("Linear (CUDA): input must be non-empty.");
        }
        if (W.Rank() != 2) {
            throw std::invalid_argument("Linear (CUDA): weight must be 2D (out, in).");
        }
        if (!X.IsCUDA() || !W.IsCUDA() || (B.Size() != 0 && !B.IsCUDA())) {
            throw std::invalid_argument("Linear (CUDA): tensors are on different devices.");
        }

        const size_t in = X.Shape().back();
        const size_t out = W.Shape()[0];
        if (in != W.Shape()[1]) {
            throw std::invalid_argument("Linear (CUDA): input features do not match weight.");
        }
        if (B.Size() != 0 && B.Size() != out) {
            throw std::invalid_argument("Linear (CUDA): bias size must equal out_features.");
        }
        if (!X.IsContiguous() || !W.IsContiguous() || !B.IsContiguous()) {
            throw std::logic_error("Linear (CUDA): operands must be contiguous.");
        }

        const size_t M = X.Size() / in;
        std::vector<size_t> shape = X.Shape();
        shape.back() = out;
        Tensor<T> result(shape, Device::CUDA);
        cuda::LaunchLinearDevice(X.Data(), W.Data(), B.Size() ? B.Data() : nullptr, result.Data(), M, in, out);
        return result;
    }

    template <typename T>
    void CopyIntoCUDA(const Tensor<T>& src, Tensor<T> dst){
            if(src.IsContiguous() && dst.IsContiguous()){
                cuda::CopyDeviceToDevice(dst.Data(), src.Data(), src.Size() * sizeof(T));
            }
            else{
                cuda::LaunchStridedCopyDevice(src.Data(), dst.Data(), dst.Shape(), src.Strides(), dst.Strides());
            }
    }

    template <typename T>
    Tensor<T> ActivationCUDA(const Tensor<T>& X, Activation act){
        if(!X.IsCUDA()){
            throw std::invalid_argument("Activation (CUDA): Tensor is of different device");
        }
        if(!X.IsContiguous()){
            throw std::logic_error("Activation (CUDA): operand must be contiguous.");
        }

        Tensor<T> result(X.Shape(), Device::CUDA);
        cuda::LaunchActivationDevice(X.Data(), result.Data(), X.Size(), ActivationToCuda(act));

        return result;
    }


}

#endif
