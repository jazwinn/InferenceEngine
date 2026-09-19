#pragma once

#include <cstddef>

namespace transformer
{
struct TransformerConfig
{
    //GPT-2
    size_t vocab_size  = 50257;
    size_t max_seq_len = 1024;   // n_positions
    size_t d_model     = 768;    // n_embd
    size_t num_layers  = 12;     // n_layer
    size_t num_heads   = 12;     // n_head
    size_t ffn_dim     = 3072;   // n_inner (null in the file, which means 4 * d_model)
    float  norm_eps    = 1e-5f;  // layer_norm_epsilon
    int eos_token_id = 50256;
};

}