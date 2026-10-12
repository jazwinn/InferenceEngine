#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "Tensor/Tensor.h"
#include "NeuralNet/Module.h"
#include "NeuralNet/Linear.h"
#include "NeuralNet/Transformer/TransformerConfig.h"

namespace nn::transformer{

    using tensor::Tensor;
    using nn::Module;
    using nn::Linear;
    using nn::transformer::config::TransformerConfig;
    using nn::transformer::config::MLPType;
    using nn::transformer::config::Activation;

    template <typename T = float>
    class MLP: public Module<T>
    {
    public:
        explicit MLP(const TransformerConfig& cfg):
            activation(cfg.activation),
            up(std::make_shared<Linear<T>>(ValidatedSize(cfg.d_model, "d_model"), ValidatedSize(cfg.ffn_dim, "ffn_dim"), cfg.output_bias)),
            down(std::make_shared<Linear<T>>(cfg.ffn_dim, cfg.d_model, cfg.output_bias))
        {
            if (cfg.mlp_type == MLPType::Gated)
            {
                gate = std::make_shared<Linear<T>>(cfg.d_model, cfg.ffn_dim, cfg.output_bias);
                this->RegisterSubmodule("gate", gate);
            }
            this->RegisterSubmodule("up", up);
            this->RegisterSubmodule("down", down);
        }

        Tensor<T> Forward(const Tensor<T>& input){
            Tensor<T> h = up->Forward(input);
            h = gate ? tensor::Activate(gate->Forward(input), activation) * h : tensor::Activate(h, activation);
            return down->Forward(h);
        }

        Activation activation;
        std::shared_ptr<Linear<T>> up;
        std::shared_ptr<Linear<T>> down;
        std::shared_ptr<Linear<T>> gate;

    private:
        static size_t ValidatedSize(size_t value, const char* name)
        {
            if (value == 0)
            {
                throw std::invalid_argument(std::string("MLP: ") + name + " must be greater than zero.");
            }
            return value;
        }
    };
}
