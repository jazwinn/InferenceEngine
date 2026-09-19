#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include "NeuralNet/Transformer/TransformerConfig.h"
#include "NeuralNet/Transformer/Transformer.h"
#include "NeuralNet/Embedding.h"
#include "NeuralNet/Linear.h"


namespace llm{

    using namespace transformer;
    using namespace nn;

    template <typename T = float>
    class GPT2 : public Module<T>{
    public:
        explicit GPT2(TransformerConfig _config):
        config(Validated(std::move(_config))),
        transformer(std::make_shared<Transformer<T>>(config)),
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

        Tensor<T> Forward(const Tensor<int64_t>& ids, size_t start_pos = 0){

            if (ids.Rank() != 2)
                throw std::invalid_argument("GPT2: ids must be a [batch, seq] tensor.");

            const size_t seq = ids.Shape()[1];
            if (start_pos + seq > config.max_seq_len)
                throw std::invalid_argument("GPT2: sequence exceeds max_seq_len.");

            Tensor<T> x = wte->Forward(ids);
            if(wpe != nullptr){
                Tensor<int64_t> positions({seq});
                for (size_t i = 0; i < seq; ++i) positions(i) = static_cast<int64_t>(start_pos + i);
                x = x + wpe->Forward(positions);
            }

            Tensor<T> h = transformer->Forward(x, Tensor<bool>(), config.causal);

            if(lm_head != nullptr){
                return lm_head->Forward(h);
            }
            return math::MatMul(h, wte->weight->Transpose());
        }

        TransformerConfig config;
        std::shared_ptr<Transformer<T>> transformer;
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
