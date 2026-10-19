#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include "Tensor/Tensor.h"
#include "NeuralNet/Module.h"
#include "NeuralNet/Linear.h"
#include "NeuralNet/Embedding.h"
#include "NeuralNet/Transformer/KVCache.h"
#include "NeuralNet/Transformer/Decoder.h"
#include "NeuralNet/Transformer/TransformerConfig.h"

namespace models{

    using tensor::Tensor;
    using nn::Module;
    using nn::Linear;
    using nn::Embedding;
    using nn::transformer::Decoder;
    using nn::transformer::KVCache;
    using nn::transformer::config::TransformerConfig;
    using nn::transformer::config::PositionEncoding;

    template <typename T = float>
    class CausalModel : public Module<T>{
    public:
        explicit CausalModel(TransformerConfig _config):
        config(Validated(std::move(_config))),
        transformer(std::make_shared<Decoder<T>>(config)),
        wte(std::make_shared<Embedding<T>>(config.vocab_size, config.d_model))
        {
            this->RegisterSubmodule("transformer", transformer);
            this->RegisterSubmodule("wte", wte);
            if(config.position == PositionEncoding::Learned){
                wpe = std::make_shared<Embedding<T>>(config.max_seq_len, config.d_model);
                this->RegisterSubmodule("wpe", wpe);
            }
            if(!config.tie_word_embeddings){
                lm_head = std::make_shared<Linear<T>>(config.d_model, config.vocab_size, false);
                this->RegisterSubmodule("lm_head", lm_head);
            }
        }

        Tensor<T> Forward(const Tensor<int64_t>& ids, KVCache<T>* cache = nullptr, size_t start_pos = 0){

            if (ids.Rank() != 2)
                throw std::invalid_argument("CausalModel: ids must be a [batch, seq] tensor.");

            const size_t seq = ids.Shape()[1];

            // A cache already knows how far the sequence has got, so it is the
            // single source of truth for absolute positions; start_pos applies
            // only to uncached calls. Otherwise every decoded token would be
            // given wpe[0] and the positional encoding would silently vanish.
            const size_t first = cache ? cache->Length() : start_pos;
            if (first + seq > config.max_seq_len)
                throw std::invalid_argument("CausalModel: sequence exceeds max_seq_len.");

            Tensor<T> x = wte->Forward(ids);
            if(wpe != nullptr){
                Tensor<int64_t> positions({seq});
                for (size_t i = 0; i < seq; ++i) positions(i) = static_cast<int64_t>(first + i);
                x = x + wpe->Forward(positions.To(wpe->weight->GetDevice()));
            }

            Tensor<T> h = transformer->Forward(x, cache, Tensor<bool>(), config.causal);

            if(lm_head != nullptr){
                return lm_head->Forward(h);
            }
            return tensor::Linear(h, *wte->weight);
        }

        TransformerConfig config;
        std::shared_ptr<Decoder<T>> transformer;
        std::shared_ptr<Embedding<T>> wte;
        std::shared_ptr<Embedding<T>> wpe;
        std::shared_ptr<Linear<T>> lm_head;

    private:
        static TransformerConfig Validated(TransformerConfig c){
            c.Validate();
            return c;
        }
    };
}