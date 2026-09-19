#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "Math/Tensor.h"
#include "NeuralNet/Module.h"
#include "NeuralNet/Linear.h"
#include "NeuralNet/Activation.h"

namespace transformer
{

    using namespace activation;
    using namespace nn;

    template <typename T = float>
    class MultiHeadAttention : public nn::Module<T>
    {
    public:
        MultiHeadAttention(size_t d_model, size_t num_heads, bool use_bias = true) : 
        d_model(d_model),
        num_heads(num_heads),
        head_dim(ValidatedHeadDim(d_model, num_heads)),
        scale(std::sqrt(static_cast<T>(head_dim))),
        query(std::make_shared<Linear<T>>(d_model, d_model, use_bias)),
        key(std::make_shared<Linear<T>>(d_model, d_model, use_bias)),
        value(std::make_shared<Linear<T>>(d_model, d_model, use_bias)),
        output_proj(std::make_shared<Linear<T>>(d_model, d_model, use_bias))
        {

            this->RegisterSubmodule("query", query);
            this->RegisterSubmodule("key", key);
            this->RegisterSubmodule("value", value);
            this->RegisterSubmodule("output_proj", output_proj);
        }

        static size_t ValidatedHeadDim(size_t d_model, size_t num_heads)
        {
            if (num_heads == 0)
            {
                throw std::invalid_argument("MultiHeadAttention: num_heads must be greater than zero.");
            }
            if (d_model % num_heads != 0)
            {
                throw std::invalid_argument("MultiHeadAttention: d_model must be divisible by num_heads.");
            }
            return d_model / num_heads;
        }

        Tensor<T> SplitHeads(const Tensor<T> &x)
        {
            std::vector<size_t> new_shape = {x.Shape()[0], x.Shape()[1], num_heads, head_dim};
            return x.Reshape(new_shape).Transpose(1, 2).Contiguous(); // (batch_size, num_heads, seq_length, head_dim)
        }

        Tensor<T> MergeHeads(const Tensor<T> &x)
        {
            Tensor<T> transposed = x.Transpose(1, 2);                                         // (batch_size, seq_length, num_heads, head_dim)
            return transposed.Reshape(transposed.Shape()[0], transposed.Shape()[1], d_model); // (batch_size, seq_length, d_model)
        }

        Tensor<T> Forward(const Tensor<T>& q, const Tensor<T>& k, const Tensor<T>& v, const Tensor<bool>& mask = Tensor<bool>(), bool causal = false)
        {
            Tensor<T> Q = SplitHeads(query->Forward(q));
            Tensor<T> K = SplitHeads(key->Forward(k));
            Tensor<T> V = SplitHeads(value->Forward(v));

            Tensor<T> scores = MatMul(Q, K.Transpose(-2, -1)) / scale;

            if (causal)
            {
                const size_t seq_q = scores.Shape()[scores.Rank() - 2];
                const size_t seq_k = scores.Shape()[scores.Rank() - 1];
                Tensor<bool> causal_mask({seq_q, seq_k}, false);

                for (size_t i = 0; i < seq_q; ++i)
                {
                    for (size_t j = i + 1 + (seq_k - seq_q); j < seq_k; ++j)
                    {
                        causal_mask(i, j) = true;
                    }
                }
                scores = scores.MaskedFill(causal_mask, -std::numeric_limits<T>::infinity());
            }

            if (mask.Size() > 0)
            {
                scores = scores.MaskedFill(mask, -std::numeric_limits<T>::infinity());
            }

            Tensor<T> attention_weights = Softmax<T>(-1).Forward(scores);
            Tensor<T> context = MatMul(attention_weights, V);
            return output_proj->Forward(MergeHeads(context));
        }

        Tensor<T> Forward(const Tensor<T> &input, const Tensor<bool> &mask = Tensor<bool>(), bool causal = false)
        {
            return Forward(input, input, input, mask, causal);
        }

        template <typename... Args>
        Tensor<T> operator()(const Tensor<T> &input, Args... args)
        {
            return Forward(input, args...);
        }

        // Declared before the projections so head_dim is validated first:
        // members are initialized in declaration order, not initializer-list order.
        size_t d_model;
        size_t num_heads;
        size_t head_dim;
        T scale;

        std::shared_ptr<Linear<T>> query;
        std::shared_ptr<Linear<T>> key;
        std::shared_ptr<Linear<T>> value;
        std::shared_ptr<Linear<T>> output_proj;
    };

}
