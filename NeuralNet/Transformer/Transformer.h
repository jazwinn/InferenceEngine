#pragma once

#include <nlohmann/json.hpp>

namespace transformer
{
    struct TransformerConfig
    {
        size_t d_model;
        size_t num_heads;
        size_t num_layers;
        size_t feedforward_dim;
        float dropout_rate;
    };

    template <typename T>
    class Transformer
    {
    public:
        Transformer(){

        }
    };
}