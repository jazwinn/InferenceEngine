#pragma once

#include <vector>

#include "Models/WeightLoader.h"
#include "NeuralNet/Transformer/TransformerConfig.h"

namespace models::gpt2 {

    using nn::transformer::config::TransformerConfig;
    using nn::transformer::config::Activation;


    inline TransformerConfig Small() {
        TransformerConfig c;
        c.vocab_size = 50257; c.d_model = 768; c.num_layers = 12; c.num_heads = 12;
        c.ffn_dim = 3072; c.max_seq_len = 1024; c.activation = Activation::GELUTanh;
        c.eos_token_ids = {50256};
        return c;
    }
}

namespace models::llama {

    using nn::transformer::config::TransformerConfig;
    using nn::transformer::config::Activation;
    using nn::transformer::config::MLPType;
    using nn::transformer::config::NormType;
    using nn::transformer::config::PositionEncoding;
    using nn::transformer::config::RopeScalingType;

    inline TransformerConfig Llama32_1B() {
        TransformerConfig c;
        c.vocab_size = 128256; c.d_model = 2048; c.num_layers = 16;
        c.num_heads = 32; c.num_kv_heads = 8; c.head_dim = 64;
        c.ffn_dim = 8192; c.max_seq_len = 131072;

        c.norm_type = NormType::RMSNorm; c.norm_eps = 1e-5f;
        c.mlp_type = MLPType::Gated; c.activation = Activation::SiLU;
        c.position = PositionEncoding::Rotary; c.rope_theta = 500000.0f;
        c.rope_scaling.type = RopeScalingType::Llama3;
        c.rope_scaling.factor = 32.0;
        c.rope_scaling.low_freq_factor = 1.0;
        c.rope_scaling.high_freq_factor = 4.0;
        c.rope_scaling.original_max_position = 8192;
        c.attention_bias = false; c.output_bias = false;
        c.tie_word_embeddings = true;

        c.bos_token_id = 128000;
        c.eos_token_ids = {128001, 128008, 128009};
        return c;
    }

    inline const std::vector<RenameRule> &Rules()
    {
        static const std::vector<RenameRule> rules = {
            {"model.embed_tokens.weight", "wte.weight"},
            {"model.norm.weight", "transformer.finalNorm.weight"},
            {"model.layers.{i}.input_layernorm.weight", "transformer.h.{i}.attenNorm.weight"},
            {"model.layers.{i}.post_attention_layernorm.weight", "transformer.h.{i}.mlpNorm.weight"},
            {"model.layers.{i}.self_attn.q_proj.weight", "transformer.h.{i}.attention.query.weight"},
            {"model.layers.{i}.self_attn.k_proj.weight", "transformer.h.{i}.attention.key.weight"},
            {"model.layers.{i}.self_attn.v_proj.weight", "transformer.h.{i}.attention.value.weight"},
            {"model.layers.{i}.self_attn.o_proj.weight", "transformer.h.{i}.attention.output_proj.weight"},
            {"model.layers.{i}.mlp.gate_proj.weight", "transformer.h.{i}.mlp.gate.weight"},
            {"model.layers.{i}.mlp.up_proj.weight", "transformer.h.{i}.mlp.up.weight"},
            {"model.layers.{i}.mlp.down_proj.weight", "transformer.h.{i}.mlp.down.weight"},
        };
        return rules;
    }
}
