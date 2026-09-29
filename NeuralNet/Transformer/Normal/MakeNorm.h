#pragma once

#include <memory>
#include <stdexcept>

#include "NeuralNet/Transformer/Normal/Norm.h"
#include "NeuralNet/Transformer/Normal/LayerNorm.h"
#include "NeuralNet/Transformer/Normal/RMSNorm.h"
#include "NeuralNet/Transformer/TransformerConfig.h"

namespace nn::transformer::norm
{
    using nn::transformer::config::TransformerConfig;
    using nn::transformer::config::NormType;

    template <typename T>
    std::shared_ptr<Norm<T>> MakeNorm(const TransformerConfig& cfg)
    {
        switch (cfg.norm_type)
        {
        case NormType::RMSNorm:   return std::make_shared<RMSNorm<T>>(cfg.d_model, T(cfg.norm_eps));
        case NormType::LayerNorm: return std::make_shared<LayerNorm<T>>(cfg.d_model, T(cfg.norm_eps));
        }
        throw std::invalid_argument("MakeNorm: unsupported norm type.");
    }
}
