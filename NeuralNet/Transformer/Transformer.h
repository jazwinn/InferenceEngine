#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "NeuralNet/Module.h"
#include "TransformerConfig.h"
#include "MultiHeadAttention.h"
#include "NeuralNet/Sequential.h"
#include "NeuralNet/Activation.h"
#include "NeuralNet/LayerNorm.h"

namespace transformer
{

    template <typename T>
    class TransformerBlock : public Module<T>
    {
    public:
        using FeedForward = Sequential<T, Linear<T>, GELU<T>, Linear<T>>;

        TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff, T eps = T(1e-5))
        : ln_1(std::make_shared<nn::LayerNorm<T>>(d_model, eps)),
          attention(std::make_shared<MultiHeadAttention<T>>(d_model, num_heads)),
          ln_2(std::make_shared<nn::LayerNorm<T>>(d_model, eps)),
          mlp(std::make_shared<FeedForward>(nn::Linear<T>(d_model, d_ff), activation::GELU<T>(), nn::Linear<T>(d_ff, d_model)))
        {
            this->RegisterSubmodule("ln_1", ln_1);
            this->RegisterSubmodule("attn", attention);
            this->RegisterSubmodule("ln_2", ln_2);
            this->RegisterSubmodule("mlp", mlp);
        }

        Tensor<T> Forward(const Tensor<T> &input, const Tensor<bool> &mask = Tensor<bool>(), bool causal = false)
        {
            Tensor<T> output = input + attention->Forward (ln_1->Forward(input), mask, causal);
            output = output + mlp->Forward (ln_2->Forward (output));

            return output;
        }

        std::shared_ptr<nn::LayerNorm<T>> ln_1;
        std::shared_ptr<MultiHeadAttention<T>> attention;
        std::shared_ptr<nn::LayerNorm<T>> ln_2;
        std::shared_ptr<FeedForward> mlp;
    };

    template <typename T>
    class Transformer : public Module<T>
    {
    public:
        explicit Transformer(TransformerConfig _config) : 
        config(std::move(_config)),
        ln_f(std::make_shared<nn::LayerNorm<T>>(config.d_model, config.norm_eps))
        {
            blocks.reserve(config.num_layers);
            for (size_t i = 0; i < this->config.num_layers; ++i)
            {
                auto block = std::make_shared<TransformerBlock<T>>(
                    this->config.d_model, this->config.num_heads, this->config.ffn_dim, T(this->config.norm_eps));
                this->RegisterSubmodule("h." + std::to_string(i), block);
                blocks.push_back(block);
            }
            this->RegisterSubmodule("ln_f", ln_f);
        }

        Tensor<T> Forward(const Tensor<T> &input, const Tensor<bool> &mask = Tensor<bool>(), bool causal = true)
        {
            Tensor<T> output = input;
            for (auto &block : blocks)
            {
                output = block->Forward(output, mask, causal);
            }
            return ln_f->Forward(output);
        }

        TransformerConfig config;

        std::vector<std::shared_ptr<TransformerBlock<T>>> blocks;
        std::shared_ptr<nn::LayerNorm<T>> ln_f;
    };

}



