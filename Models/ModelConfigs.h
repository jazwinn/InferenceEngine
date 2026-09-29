#pragma once

#include "NeuralNet/Transformer/TransformerConfig.h"

namespace models::gpt2 {

    using nn::transformer::config::TransformerConfig;
    using nn::transformer::config::Activation;


    inline TransformerConfig Small() {
        TransformerConfig c;
        c.vocab_size = 50257; c.d_model = 768; c.num_layers = 12; c.num_heads = 12;
        c.ffn_dim = 3072; c.max_seq_len = 1024; c.activation = Activation::GELUTanh;
        c.eos_token_id = 50256;
        return c;
    }
}