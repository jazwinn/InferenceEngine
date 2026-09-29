#pragma once

#include <memory>
#include <utility>

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
}
