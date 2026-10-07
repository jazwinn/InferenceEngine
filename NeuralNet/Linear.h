#pragma once

#include "Tensor/Tensor.h"
#include "Module.h"

#include <memory>

namespace nn{

    template <typename T = float>
    class Linear : public Module<T>{
    public:
        Linear(size_t input_size, size_t output_size, bool use_bias = true) :
        weight(std::make_shared<Tensor<T>>(std::vector<size_t>{output_size, input_size})) {
            this->RegisterParameter("weight", weight);

            if (use_bias) {
                bias = std::make_shared<Tensor<T>>(std::vector<size_t>{output_size});
                this->RegisterParameter("bias", bias);
            }
        }

        Tensor<T> Forward(const Tensor<T>& input) {
            Tensor<T> output = MatMul(input, weight->Transpose());
            if (bias) {
                output += *bias;
            }
            return output;
        }
    
        std::shared_ptr<Tensor<T>> weight;
        std::shared_ptr<Tensor<T>> bias;
    };
}