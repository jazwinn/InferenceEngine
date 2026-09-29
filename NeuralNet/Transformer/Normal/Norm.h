#pragma once

#include "Math/Tensor.h"
#include "NeuralNet/Module.h"

namespace nn::transformer::norm
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
