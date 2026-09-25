#pragma once

#include <memory>

#include "NeuralNet/Module.h"
#include "Math/Tensor.h"
#include "NeuralNet/Linear.h"
#include "NeuralNet/Transformer/KVCache.h"
#include "NeuralNet/Transformer/AttentionOps.h"
#include "NeuralNet/Transformer/TransformerConfig.h"

namespace transformer{

    using nn::Linear;
    using nn::Module;
    using math::Tensor;
    using math::CopyInto;


    template <typename T = float>
    class GroupQueryAttention: public Module<T>
    {
    public:
        explicit GroupQueryAttention(const TransformerConfig& cfg):
            dModel(cfg.d_model > 0 ? cfg.d_model : throw std::invalid_argument("GroupQueryAttention: d_model must be greater than zero.")),
            numHeadsQ(cfg.num_heads),
            numHeadsKV(cfg.KVHeads()),
            numGroups(ValidatedDivide(numHeadsQ, numHeadsKV)),
            headDim(cfg.head_dim > 0 ? cfg.head_dim : ValidatedDivide(dModel, numHeadsQ)),
            scale(T(1/std::sqrt(static_cast<T>(headDim)))),
            query(std::make_shared<Linear<T>>(dModel, headDim * numHeadsQ, cfg.attention_bias)),
            key(std::make_shared<Linear<T>>(dModel, headDim * numHeadsKV, cfg.attention_bias)),
            value(std::make_shared<Linear<T>>(dModel, headDim * numHeadsKV, cfg.attention_bias)),
            output_proj(std::make_shared<Linear<T>>(headDim * numHeadsQ, dModel, cfg.output_bias))
        {
            this->RegisterSubmodule("query", query);
            this->RegisterSubmodule("key", key);
            this->RegisterSubmodule("value", value);
            this->RegisterSubmodule("output_proj", output_proj);
        }



        Tensor<T> Forward(const Tensor<T>& input, KVLayer<T>* cache = nullptr, size_t cache_len = 0, const Tensor<bool> &mask = Tensor<bool>(), bool causal = false){
            
            const size_t batch = input.Shape()[0];
            const size_t seq = input.Shape()[1];
            
            Tensor<T> Q = SplitHeads(query->Forward(input), numHeadsQ, headDim);
            Tensor<T> K = SplitHeads(key->Forward(input), numHeadsKV, headDim);
            Tensor<T> V = SplitHeads(value->Forward(input), numHeadsKV, headDim);

            Q = Q.Reshape(batch, numHeadsKV, numGroups, seq, headDim);

            if (cache)
            {
                CopyInto(cache->k.Slice(2, cache_len, seq), K);
                CopyInto(cache->v.Slice(2, cache_len, seq), V);
                K = cache->k.Slice(2, 0, cache_len + seq);
                V = cache->v.Slice(2, 0, cache_len + seq);
            }

            const size_t KVlen = K.Shape()[2];
            K = K.Unsqueeze(2);
            V = V.Unsqueeze(2);


            //ensure correct broadcast
            Tensor<bool> attnMask = NormalizeMask(mask, batch, numHeadsQ, seq, KVlen);
            if (attnMask.Size() > 0)
            {
                const auto& s = attnMask.Shape();
                attnMask = s[1] == 1 ? attnMask.Unsqueeze(2)
                                     : attnMask.Reshape(s[0], numHeadsKV, numGroups, s[2], s[3]);
            }
 
            Tensor<T> context = SDPA(Q, K, V, attnMask, causal, scale).Reshape(batch, numHeadsQ, seq, headDim);

            return output_proj->Forward(MergeHeads(context, headDim * numHeadsQ));

        }

        template <typename... Args>
        Tensor<T> operator()(const Tensor<T> &input, Args... args)
        {
            return Forward(input, args...);
        }


        size_t dModel;
        size_t numHeadsQ;
        size_t numHeadsKV;
        size_t numGroups;
        size_t headDim;
        T scale;

        std::shared_ptr<Linear<T>> query;
        std::shared_ptr<Linear<T>> key;
        std::shared_ptr<Linear<T>> value;
        std::shared_ptr<Linear<T>> output_proj;
    };

}