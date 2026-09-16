#pragma once

#include "Math/tensor.h"
#include "Module.h"

#include <memory>

namespace nn{

    template <typename T>
    class Linear : public Module<T>{
    public:
        Linear(size_t input_size, size_t output_size, bool use_bias = true) :
        weight(std::make_shared<Tensor<T>>(std::vector<size_t>{output_size, input_size})) {
            this->register_parameter("weight", weight);

            if (use_bias) {
                bias = std::make_shared<Tensor<T>>(std::vector<size_t>{output_size});
                this->register_parameter("bias", bias);
            }
        }

        Tensor<T> forward(const Tensor<T>& input) override {
            Tensor<T> output = MatMul(input, weight->transpose());
            if (bias) {
                output += *bias;
            }
            return output;
        }
    
        std::shared_ptr<Tensor<T>> weight;
        std::shared_ptr<Tensor<T>> bias;
    };
}