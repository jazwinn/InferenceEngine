#pragma once

#include "Tensor/Tensor.h"
#include "NeuralNet/Module.h"

namespace nn::transformer::norm
{
    using nn::Module;
    using tensor::Tensor;

    template <typename T>
    class Norm: public Module<T>
    {
    public:
        virtual Tensor<T> Forward(const Tensor<T>& input) = 0;
    };
}
