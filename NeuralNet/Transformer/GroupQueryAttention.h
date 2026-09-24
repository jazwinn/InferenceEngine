#pragma once

#include <memory>

#include "NeuralNet/Module.h"
#include "Math/Tensor.h"
#include "NeuralNet/Linear.h"
#include "NeuralNet/Transformer/KVCache.h"
#include "NeuralNet/Transformer/AttentionOps.h"

namespace transformer{

    using nn::Linear;
    using nn::Module;
    using math::Tensor;
    using math::CopyInto;


    template <typename T>
    class GroupQueryAttention: public Module<T>
    {
    public:
        GroupQueryAttention(size_t d_model, size_t num_heads_q, size_t num_heads_kv, bool use_bias = true):
            dModel(d_model),
            numHeadsQ(num_heads_q),
            numHeadsKV(num_heads_kv),
            numGroups(numHeadsQ/numHeadsKV),
            headDim(ValidatedHeadDim(d_model,num_heads_q)),
            scale(T(1/std::sqrt(static_cast<T>(headDim)))),
            query(std::make_shared<Linear<T>>(d_model, headDim * numHeadsQ, use_bias)),
            key(std::make_shared<Linear<T>>(d_model, headDim * numHeadsKV, use_bias)),
            value(std::make_shared<Linear<T>>(d_model, headDim * numHeadsKV, use_bias)),
            output_proj(std::make_shared<Linear<T>>(d_model, d_model, use_bias))
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

            Q = Q.Reshape(batch, numHeadsKV, numGroups, seq, headDim).Contiguous();
            //reshape KV wip

            if (cache)
            {
                CopyInto(cache->k.Slice(2, cache_len, seq), K);
                CopyInto(cache->v.Slice(2, cache_len, seq), V);
                K = cache->k.Slice(2, 0, cache_len + seq);
                V = cache->v.Slice(2, 0, cache_len + seq);
            }

            return output_proj->Forward(MergeHeads(SDPA(Q, K, V, mask, causal, scale), dModel));

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