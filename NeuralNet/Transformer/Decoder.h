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
#include "NeuralNet/Transformer/Normal/LayerNorm.h"
#include "NeuralNet/Transformer/KVCache.h"

namespace nn::transformer
{
    using math::Tensor;
    using nn::Module;
    using nn::Linear;
    using nn::transformer::normal::LayerNorm;
    using nn::Sequential;
    using nn::GELU;

    template <typename T>
    class TransformerBlock : public Module<T>
    {
    public:
        using FeedForward = Sequential<T, Linear<T>, GELU<T>, Linear<T>>;

        TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff, T eps = T(1e-5))
        : ln_1(std::make_shared<nn::transformer::normal::LayerNorm<T>>(d_model, eps)),
          attention(std::make_shared<MultiHeadAttention<T>>(d_model, num_heads)),
          ln_2(std::make_shared<nn::transformer::normal::LayerNorm<T>>(d_model, eps)),
          mlp(std::make_shared<FeedForward>(nn::Linear<T>(d_model, d_ff), nn::GELU<T>(), nn::Linear<T>(d_ff, d_model)))
        {
            this->RegisterSubmodule("ln_1", ln_1);
            this->RegisterSubmodule("attn", attention);
            this->RegisterSubmodule("ln_2", ln_2);
            this->RegisterSubmodule("mlp", mlp);
        }

        Tensor<T> Forward(const Tensor<T> &input, KVLayer<T>* cache = nullptr, size_t cache_len = 0, const Tensor<bool> &mask = Tensor<bool>(), bool causal = false)
        {
            Tensor<T> output = input + attention->Forward(ln_1->Forward(input), cache, cache_len, mask, causal);
            output = output + mlp->Forward (ln_2->Forward (output));

            return output;
        }

        std::shared_ptr<nn::transformer::normal::LayerNorm<T>> ln_1;
        std::shared_ptr<MultiHeadAttention<T>> attention;
        std::shared_ptr<nn::transformer::normal::LayerNorm<T>> ln_2;
        std::shared_ptr<FeedForward> mlp;
    };

    template <typename T>
    class Transformer : public Module<T>
    {
    public:
        explicit Transformer(TransformerConfig _config) : 
        config(std::move(_config)),
        ln_f(std::make_shared<nn::transformer::normal::LayerNorm<T>>(config.d_model, config.norm_eps))
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

        Tensor<T> Forward(const Tensor<T> &input, KVCache<T>* cache = nullptr, const Tensor<bool> &mask = Tensor<bool>(), bool causal = true)
        {

            const size_t seq       = input.Shape()[1];
            const size_t cache_len = cache ? cache->Room(seq) : 0;
            
            Tensor<T> output = input;
            for (size_t i = 0; i < blocks.size(); ++i)
            {
                // Every layer appends at the SAME offset, so cache_len is read
                // once above -- never re-read from the cache inside this loop.
                KVLayer<T>* kv = cache ? &cache->Layer(i) : nullptr;

                output = blocks[i]->Forward(output, kv, cache_len, mask, causal);
            }

            if (cache) cache->Advance(seq);   // once per pass, after every layer
            return ln_f->Forward(output);
        }

        TransformerConfig config;

        std::vector<std::shared_ptr<TransformerBlock<T>>> blocks;
        std::shared_ptr<nn::transformer::normal::LayerNorm<T>> ln_f;
    };

}



