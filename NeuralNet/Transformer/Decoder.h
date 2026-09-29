#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Math/Tensor.h"
#include "NeuralNet/Module.h"
#include "NeuralNet/Transformer/MLP.h"
#include "NeuralNet/Transformer/RoPE.h"
#include "NeuralNet/Transformer/KVCache.h"
#include "NeuralNet/Transformer/Normal/Norm.h"
#include "NeuralNet/Transformer/Normal/MakeNorm.h"
#include "NeuralNet/Transformer/TransformerConfig.h"
#include "NeuralNet/Transformer/GroupQueryAttention.h"

namespace nn::transformer
{
    using math::Tensor;
    using nn::Module;
    using nn::transformer::norm::Norm;
    using nn::transformer::config::TransformerConfig;
    using nn::transformer::config::NormPlacement;
    using nn::transformer::config::PositionEncoding;

    template <typename T = float>
    class DecoderBlock : public Module<T>
    {
    public:
        explicit DecoderBlock(const TransformerConfig& cfg, std::shared_ptr<RotaryEmbedding<T>> rope = nullptr):
            placement(cfg.norm_placement),
            attention(std::make_shared<GroupQueryAttention<T>>(cfg, std::move(rope))),
            attenNorm(norm::MakeNorm<T>(cfg)),
            mlp(std::make_shared<MLP<T>>(cfg)),
            mlpNorm(norm::MakeNorm<T>(cfg))
        {
            this->RegisterSubmodule("attention", attention);
            this->RegisterSubmodule("attenNorm", attenNorm);
            this->RegisterSubmodule("mlp", mlp);
            this->RegisterSubmodule("mlpNorm", mlpNorm);
        }

        Tensor<T> Forward(const Tensor<T>& input, KVLayer<T>* cache = nullptr, size_t cache_len = 0, const Tensor<bool>& mask = Tensor<bool>(), bool causal = true)
        {
            if (placement == NormPlacement::Pre)
            {
                Tensor<T> output = input + attention->Forward(attenNorm->Forward(input), cache, cache_len, mask, causal);
                return output + mlp->Forward(mlpNorm->Forward(output));
            }

            Tensor<T> output = attenNorm->Forward(input + attention->Forward(input, cache, cache_len, mask, causal));
            return mlpNorm->Forward(output + mlp->Forward(output));
        }

        NormPlacement placement;
        std::shared_ptr<GroupQueryAttention<T>> attention;
        std::shared_ptr<Norm<T>> attenNorm;
        std::shared_ptr<MLP<T>> mlp;
        std::shared_ptr<Norm<T>> mlpNorm;
    };


    template <typename T = float>
    class Decoder : public Module<T>
    {
    public:
        explicit Decoder(TransformerConfig cfg) : 
            config(std::move(cfg))
        {
            config.Validate();
            if(config.position == PositionEncoding::Rotary){
                rope = std::make_shared<RotaryEmbedding<T>>(config.HeadDim(), config.max_seq_len, config.rope_theta, config.rope_scaling);
            }


            blocks.reserve(config.num_layers);
            for (size_t i = 0; i < this->config.num_layers; ++i)
            {
                auto block = std::make_shared<DecoderBlock<T>>(config, rope);
                this->RegisterSubmodule("h." + std::to_string(i), block);
                blocks.push_back(block);
            }

            if(config.norm_placement == NormPlacement::Pre){
                finalNorm = norm::MakeNorm<T>(config);
                this->RegisterSubmodule("finalNorm", finalNorm);    
            }

            
        }

        Tensor<T> Forward(const Tensor<T> &input, KVCache<T>* cache = nullptr, const Tensor<bool> &mask = Tensor<bool>(), bool causal = true)
        {

            const size_t seq       = input.Shape()[1];
            const size_t cache_len = cache ? cache->Room(seq) : 0;
            
            Tensor<T> output = input;
            for (size_t i = 0; i < blocks.size(); ++i)
            {
                KVLayer<T>* kv = cache ? &cache->Layer(i) : nullptr;

                output = blocks[i]->Forward(output, kv, cache_len, mask, causal);
            }

            if (cache) cache->Advance(seq);
            return finalNorm? finalNorm->Forward(output): output;
        }

        TransformerConfig config;
        std::shared_ptr<RotaryEmbedding<T>> rope;
        std::vector<std::shared_ptr<DecoderBlock<T>>> blocks;
        std::shared_ptr<Norm<T>> finalNorm;
    };
}
