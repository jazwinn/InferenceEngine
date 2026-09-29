#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "Math/Tensor.h"
#include "NeuralNet/Module.h"
#include "NeuralNet/Linear.h"
#include "NeuralNet/Activation.h"
#include "NeuralNet/Transformer/TransformerConfig.h"

namespace nn::transformer{

    using math::Tensor;
    using nn::Module;
    using nn::Linear;

    template <typename T = float>
    class MLP: public Module<T>
    {
    public:
        explicit MLP(const TransformerConfig& cfg):
            activation(ValidatedActivation(cfg.activation)),
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
            h = gate ? Activate(activation, gate->Forward(input)) * h : Activate(activation, h);
            return down->Forward(h);
        }

        Activation activation;
        std::shared_ptr<Linear<T>> up;
        std::shared_ptr<Linear<T>> down;
        std::shared_ptr<Linear<T>> gate;

    private:
        static Tensor<T> Activate(Activation kind, const Tensor<T>& x)
        {
            switch (kind)
            {
            case Activation::GELUTanh: return nn::GELU<T>().Forward(x);
            case Activation::SiLU:     return nn::SiLU<T>().Forward(x);
            case Activation::ReLU:     return nn::ReLU<T>().Forward(x);
            case Activation::GELU:     break;
            }
            throw std::invalid_argument("MLP: unsupported activation.");
        }

        static Activation ValidatedActivation(Activation kind)
        {
            if (kind == Activation::GELU)
            {
                throw std::invalid_argument("MLP: exact GELU is not implemented; use GELUTanh.");
            }
            return kind;
        }

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
