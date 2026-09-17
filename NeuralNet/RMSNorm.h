#pragma once

#include "Math/Tensor.h"
#include "Module.h"

namespace nn {

    template <typename T>
    class RMSNorm : public Module<T> {
    public:
        RMSNorm(size_t size, T epsilon = static_cast<T>(1e-8)) :
            weight(std::make_shared<Tensor<T>>(std::vector<size_t>{size}, T(1))),
            epsilon(epsilon) {
            this->RegisterParameter("weight", weight);
        }

        Tensor<T> Forward(const Tensor<T>& input) override {
            Tensor<T> mean_square = Mean(input * input, input.Rank() - 1, true);
            Tensor<T> rms = Sqrt(mean_square + epsilon);
            Tensor<T> normalized = input / rms;
            return normalized * *weight;
        }

        std::shared_ptr<Tensor<T>> weight;
        T epsilon;
    };
}