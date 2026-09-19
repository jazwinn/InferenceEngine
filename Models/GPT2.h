#pragma once

#include <memory>
#include <cstdint>

#include "NeuralNet/Transformer/TransformerConfig.h"
#include "NeuralNet/Transformer/Transformer.h"
#include "NeuralNet/Embedding.h"


namespace llm{

    using namespace transformer;
    using namespace nn;

    template <typename T = float>
    class GPT2 : public Module<T>{
    public:
        GPT2(TransformerConfig _config):
        config(std::move(_config)),
        transformer(std::make_shared<Transformer<T>>(config)),
        wte(std::make_shared<Embedding<T>>(config.vocab_size, config.d_model))
        {
            this->RegisterSubmodule("transformer", transformer);
            this->RegisterSubmodule("wte", wte);
            if(config.position == PositionEncoding::learned){
                wpe = std::make_shared<Embedding<T>>(config.max_seq_len, config.d_model);
                this->RegisterSubmodule("wpe", wpe);
            }
            if(!config.tie_word_embeddings){
                lm_head = std::make_shared<Linear<T>>(config.d_model, config.vocab_size);
                this->RegisterSubmodule("lm_head", lm_head);
            }
        }

        Tensor<T> Forward(const Tensor<int64_t>& ids, size_t start_pos = 0, size_t end_pos = 0, bool causal = true){
            
            const size_t seq = ids.Shape()[1];
            if (start_pos + seq > config.max_seq_len)
                throw std::invalid_argument("CausalLM: sequence exceeds max_seq_len.");

            if(start_pos > end_pos){
                throw std::invalid_argument("LLM: start_pos must be less than or equal to end_pos.");
            }
            if(start_pos == end_pos){
                throw std::invalid_argument("LLM: start_pos must be less than end_pos.");
            }
            
            Tensor<T> x = wte->Forward(ids);
            if(wpe != nullptr){
                Tensor<int64_t> positions({seq});
                for (size_t i = 0; i < seq; ++i) positions(i) = static_cast<int64_t>(start_pos + i);                
                x = x + wpe->Forward(positions);
            }

            Tensor<T> h = transformer->Forward(x, Tensor<bool>(), causal);
            
            if(lm_head != nullptr){
                return lm_head->Forward(h);
            }
            return math::MatMul(h, wte->weight->Transpose());
        }

        

        std::shared_ptr<Transformer<T>> transformer;
        std::shared_ptr<Embedding<T>> wte;
        std::shared_ptr<Embedding<T>> wpe;
        std::shared_ptr<Linear<T>> lm_head;
        TransformerConfig config;
    };
}