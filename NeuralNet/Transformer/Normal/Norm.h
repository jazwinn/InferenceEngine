#pragma once

#include "NeuralNet/Module.h"

namespace nn::transformer::normal
{
    using nn::Module;
    using math::Tensor;

    template <typename T>
    class Norm: public Module<T>
    {
    public:
        virtual Tensor<T> Forward(const Tensor<T>& input) = 0;
    };
}