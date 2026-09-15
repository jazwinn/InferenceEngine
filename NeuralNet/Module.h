#pragma once

#include "Math/tensor.h"

namespace nn{
    
    template <typename T>
    class Module{
    public:
        virtual Tensor<T> forward(const Tensor<T>& input) = 0;;
        virtual ~Module() = default;

    };
}

