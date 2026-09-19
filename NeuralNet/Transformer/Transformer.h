#pragma once

#include <utility>

#include "TransformerConfig.h"
#include "MultiHeadAttention.h"

namespace transformer
{

    template <typename T>
    class Transformer
    {
    public:
        explicit Transformer(TransformerConfig _config):
        config(std::move(_config)),
        atten(config.d_model, config.num_heads)
        {
        }

        TransformerConfig config;
        MultiHeadAttention<T> atten;
    };
}
