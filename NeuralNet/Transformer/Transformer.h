#pragma once

#include <utility>

#include "NeuralNet/Module.h"
#include "TransformerConfig.h"
#include "MultiHeadAttention.h"

namespace transformer
{

    template <typename T> 
    class TransformerBlock: public Module<T>
    {
    public:
        TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff, size_t dropout):
        d_model(d_model),
        num_heads(num_heads),
        d_ff(d_ff),
        dropout(dropout),
        attention(d_model, num_heads),
        feed_forward(d_model, d_ff, dropout)
        {
            
        }

        Tensor<T> forward(Tensor<T> const& input, Tensor<T> const& mask)
        {
            Tensor<T> output = input;
            output = attention(output, mask);
            output = output + input;
            output = feed_forward(output);
            return output;
        }

        size_t d_model;
        size_t num_heads;
        size_t d_ff;
        size_t dropout;
        MultiHeadAttention<T> attention;
        FeedForward<T> feed_forward;
    };


    }

    template <typename T>
    class Transformer: public Module<T>
    {
    public:
        explicit Transformer(TransformerConfig _config):
        config(std::move(_config)),
        atten(config.d_model, config.num_heads)
        {
            
        }

        Tensor<T> forward(Tensor<T> const& input, Tensor<T> const& mask)
        {
            Tensor<T> output = input;
            for (int i = 0; i < config.num_layers; i++)
            {
                output = atten(output, mask);
                output = output + input;
            }
            return output;
        }

        TransformerConfig config;
        MultiHeadAttention<T> atten;
    };
}
