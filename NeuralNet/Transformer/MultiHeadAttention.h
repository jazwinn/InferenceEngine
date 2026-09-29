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
#include "NeuralNet/Transformer/KVCache.h"
#include "NeuralNet/Transformer/AttentionOps.h"


namespace nn::transformer
{

    using math::Tensor;
    using math::MatMul;
    using math::CopyInto;
    using nn::Module;
    using nn::Linear;
    using nn::Softmax;

    template <typename T = float>
    class MultiHeadAttention : public Module<T>
    {
    public:
        MultiHeadAttention(size_t d_model, size_t num_heads, bool use_bias = true) : 
        d_model(d_model),
        num_heads(num_heads),
        head_dim(ValidatedDivide(d_model, num_heads)),
        scale(1/std::sqrt(static_cast<T>(head_dim))),
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

        Tensor<T> Forward(const Tensor<T>& q, const Tensor<T>& k, const Tensor<T>& v, const Tensor<bool>& mask = Tensor<bool>(), bool causal = false)
        {
            Tensor<T> Q = SplitHeads(query->Forward(q), num_heads, head_dim);
            Tensor<T> K = SplitHeads(key->Forward(k), num_heads, head_dim);
            Tensor<T> V = SplitHeads(value->Forward(v), num_heads, head_dim);

            const Tensor<bool> attnMask = NormalizeMask(mask, Q.Shape()[0], num_heads, Q.Shape()[2], K.Shape()[2]);

            return output_proj->Forward(MergeHeads(SDPA(Q, K, V, attnMask, causal, scale), d_model));
        }


        Tensor<T> Forward(const Tensor<T> &input, KVLayer<T>* cache = nullptr, size_t cache_len = 0, const Tensor<bool> &mask = Tensor<bool>(), bool causal = false)
        {
            const size_t seq = input.Shape()[1];

            Tensor<T> Q = SplitHeads(query->Forward(input), num_heads, head_dim);
            Tensor<T> K = SplitHeads(key->Forward(input), num_heads, head_dim);
            Tensor<T> V = SplitHeads(value->Forward(input), num_heads, head_dim);

            if (cache)
            {
                CopyInto(cache->k.Slice(2, cache_len, seq), K);
                CopyInto(cache->v.Slice(2, cache_len, seq), V);
                K = cache->k.Slice(2, 0, cache_len + seq);
                V = cache->v.Slice(2, 0, cache_len + seq);
            }

            const Tensor<bool> attnMask = NormalizeMask(mask, Q.Shape()[0], num_heads, seq, K.Shape()[2]);

            return output_proj->Forward(MergeHeads(SDPA(Q, K, V, attnMask, causal, scale), d_model));
        }

        template <typename... Args>
        Tensor<T> operator()(const Tensor<T> &input, Args... args)
        {
            return Forward(input, args...);
        }

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
