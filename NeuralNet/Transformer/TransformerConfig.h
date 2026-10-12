#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include "Tensor/Activations.h"

namespace nn::transformer::config
{
    using tensor::Activation;

    enum class NormType          { LayerNorm, RMSNorm };
    enum class NormPlacement     { Pre, Post };
    enum class MLPType           { Standard, Gated };          // Gated: down(act(gate(x)) * up(x))
    enum class PositionEncoding  { Learned, Rotary, None };
    enum class RopeScalingType   { Default, Linear, Dynamic, Yarn, Llama3, LongRope, MRope };

    struct RopeScaling
    {
        RopeScalingType type = RopeScalingType::Default;
        double factor = 1.0;
        double low_freq_factor = 1.0;
        double high_freq_factor = 4.0;
        size_t original_max_position = 8192;
    };

    struct TransformerConfig
    {
        // sizes
        size_t vocab_size   = 0;
        size_t d_model      = 0;
        size_t num_layers   = 0;
        size_t num_heads    = 0;
        size_t num_kv_heads = 0;   // 0 → same as num_heads (plain multi-head attention)
        size_t head_dim     = 0;   // 0 → d_model / num_heads
        size_t ffn_dim      = 0;
        size_t max_seq_len  = 0;

        // architecture choices
        NormType         norm_type      = NormType::LayerNorm;
        NormPlacement    norm_placement = NormPlacement::Pre;
        float            norm_eps       = 1e-5f;
        MLPType          mlp_type       = MLPType::Standard;
        Activation       activation     = Activation::GELU;
        PositionEncoding position       = PositionEncoding::Learned;
        RopeScaling      rope_scaling   = RopeScaling();
        float            rope_theta     = 10000.0f;
        bool             attention_bias = true;    // bias on q/k/v projections
        bool             output_bias    = true;    // bias on attention output + MLP projections
        bool             causal         = true;
        bool             tie_word_embeddings = true;

        // special tokens (optional: not every model has all of them)
        std::optional<int64_t> bos_token_id;
        std::vector<int64_t>   eos_token_ids;
        std::optional<int64_t> pad_token_id;

        size_t KVHeads() const { return num_kv_heads ? num_kv_heads : num_heads; }
        size_t HeadDim() const { return head_dim ? head_dim : d_model / num_heads; }
        bool   IsEos(int64_t id) const { return std::find(eos_token_ids.begin(), eos_token_ids.end(), id) != eos_token_ids.end(); }
        void   Validate() const;   // throws on inconsistent values
    };

    inline void TransformerConfig::Validate() const
    {
        if (vocab_size == 0 || d_model == 0 || num_layers == 0 || num_heads == 0 || ffn_dim == 0 || max_seq_len == 0)
        {
            throw std::invalid_argument("TransformerConfig: vocab_size, d_model, num_layers, num_heads, ffn_dim and max_seq_len must all be greater than zero.");
        }
        if (head_dim == 0 && d_model % num_heads != 0)
        {
            throw std::invalid_argument("TransformerConfig: d_model must be divisible by num_heads.");
        }
        if (num_heads % KVHeads() != 0)
        {
            throw std::invalid_argument("TransformerConfig: num_heads must be divisible by num_kv_heads.");
        }
        if (position == PositionEncoding::Rotary && rope_theta <= 0.0f)
        {
            throw std::invalid_argument("TransformerConfig: rope_theta must be positive for rotary positions.");
        }
        if (norm_eps <= 0.0f)
        {
            throw std::invalid_argument("TransformerConfig: norm_eps must be positive.");
        }
    }


}