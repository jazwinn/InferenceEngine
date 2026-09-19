#pragma once

#include <cstddef>

namespace transformer
{
    enum class NormType          { LayerNorm, RMSNorm };
    enum class NormPlacement     { Pre, Post };
    enum class Activation        { GELU, GELUTanh, SiLU, ReLU };
    enum class MLPType           { Standard, Gated };          // Gated: down(act(gate(x)) * up(x))
    enum class PositionEncoding  { Learned, Rotary, None };

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
        float            rope_theta     = 10000.0f;
        bool             attention_bias = true;    // bias on q/k/v projections
        bool             output_bias    = true;    // bias on attention output + MLP projections
        bool             causal         = true;
        bool             tie_word_embeddings = true;

        // special tokens (optional: not every model has all of them)
        std::optional<int64_t> bos_token_id;
        std::optional<int64_t> eos_token_id;
        std::optional<int64_t> pad_token_id;

        size_t KVHeads() const { return num_kv_heads ? num_kv_heads : num_heads; }
        size_t HeadDim() const { return head_dim ? head_dim : d_model / num_heads; }
        void   Validate() const;   // throws on inconsistent values
    };


}