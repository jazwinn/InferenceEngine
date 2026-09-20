#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <set>
#include <map>
#include <string>
#include <array>
#include <vector>

#include "NeuralNet/Transformer/TransformerConfig.h"
#include "NeuralNet/Transformer/Transformer.h"
#include "NeuralNet/Embedding.h"
#include "NeuralNet/Linear.h"
#include "NeuralNet/LayerNorm.h"
#include "Serialize/Safetensors.h"

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

    namespace detail
    {
        template <typename T>
        class GPT2WeightLoader
        {
        public:
            using Weights = std::map<std::string, math::Tensor<T>>;

            GPT2WeightLoader(GPT2<T> &model, Weights weights)
                : m_model(model), m_weights(Normalise(std::move(weights)))
            {
            }

            void Load()
            {
                LoadEmbeddings();
                for (size_t i = 0; i < m_model.transformer->blocks.size(); ++i)
                {
                    LoadBlock(*m_model.transformer->blocks[i], "h." + std::to_string(i) + ".");
                }
                LoadNorm(*m_model.transformer->ln_f, "ln_f");
                if (m_model.lm_head)
                {
                    Copy(m_model.lm_head->weight, "lm_head.weight", Take("lm_head.weight"));
                }
                VerifyNothingWasMissed();
            }

        private:
            static Weights Normalise(Weights weights)
            {
                Weights renamed;
                for (auto &[name, tensor] : weights)
                {
                    std::string key = name.starts_with("transformer.") ? name.substr(12) : name;
                    if (key.ends_with(".attn.bias") || key.ends_with(".attn.masked_bias"))
                    {
                        continue;
                    }
                    renamed.emplace(std::move(key), std::move(tensor));
                }
                return renamed;
            }

            void LoadEmbeddings()
            {
                Copy(m_model.wte->weight, "wte.weight", Take("wte.weight"));
                if (m_model.wpe)
                {
                    Copy(m_model.wpe->weight, "wpe.weight", Take("wpe.weight"));
                }
            }

            void LoadBlock(TransformerBlock<T> &block, const std::string &prefix)
            {
                LoadNorm(*block.ln_1, prefix + "ln_1");
                LoadAttention(*block.attention, prefix + "attn.");
                LoadNorm(*block.ln_2, prefix + "ln_2");
                LoadFeedForward(block, prefix + "mlp.");
            }

            void LoadNorm(LayerNorm<T> &norm, const std::string &prefix)
            {
                Copy(norm.gamma, prefix + ".weight", Take(prefix + ".weight"));
                Copy(norm.beta, prefix + ".bias", Take(prefix + ".bias"));
            }

            void LoadAttention(MultiHeadAttention<T> &attention, const std::string &prefix)
            {
                LoadFusedQKV(attention, prefix);
                LoadLinear(*attention.output_proj, prefix + "c_proj");
            }

            void LoadFusedQKV(MultiHeadAttention<T> &attention, const std::string &prefix)
            {
                const size_t d = m_model.config.d_model;
                const math::Tensor<T> weight = Take(prefix + "c_attn.weight");
                const math::Tensor<T> bias = Take(prefix + "c_attn.bias");

                if (weight.Rank() != 2 || weight.Shape()[0] != d || weight.Shape()[1] != 3 * d)
                {
                    throw Error("'" + prefix + "c_attn.weight' is " + ShapeText(weight.Shape()) + " but should be [" +
                                std::to_string(d) + ", " + std::to_string(3 * d) + "].");
                }

                const std::array<std::shared_ptr<Linear<T>>, 3> projections{
                    attention.query, attention.key, attention.value};

                for (size_t part = 0; part < projections.size(); ++part)
                {
                    const size_t begin = part * d;
                    Copy(projections[part]->weight, prefix + "c_attn.weight", weight.Slice(1, begin, d).Transpose());
                    Copy(projections[part]->bias, prefix + "c_attn.bias", bias.Slice(0, begin, d));
                }
            }

            void LoadFeedForward(TransformerBlock<T> &block, const std::string &prefix)
            {
                LoadLinear(*block.mlp->template Get<0>(), prefix + "c_fc");
                LoadLinear(*block.mlp->template Get<2>(), prefix + "c_proj");
            }

            void LoadLinear(Linear<T> &linear, const std::string &prefix)
            {
                Copy(linear.weight, prefix + ".weight", Take(prefix + ".weight").Transpose());
                Copy(linear.bias, prefix + ".bias", Take(prefix + ".bias"));
            }

            const math::Tensor<T> &Take(const std::string &name)
            {
                const auto entry = m_weights.find(name);
                if (entry == m_weights.end())
                {
                    throw Error("checkpoint has no tensor '" + name + "'.");
                }
                m_used.insert(name);
                return entry->second;
            }

            void Copy(const std::shared_ptr<math::Tensor<T>> &parameter, const std::string &name, const math::Tensor<T> &value)
            {
                if (!parameter)
                {
                    throw Error("no parameter to receive '" + name + "'.");
                }
                if (parameter->Shape() != value.Shape())
                {
                    throw Error("'" + name + "' is " + ShapeText(value.Shape()) + " but the model expects " +
                                ShapeText(parameter->Shape()) + ".");
                }
                *parameter = value.Clone();
                ++m_filled;
            }

            void VerifyNothingWasMissed() const
            {
                const size_t expected = m_model.NamedParameters().size();
                if (m_filled != expected)
                {
                    throw Error("filled " + std::to_string(m_filled) + " parameters but the model has " +
                                std::to_string(expected) + ".");
                }
                for (const auto &[name, tensor] : m_weights)
                {
                    if (!m_used.contains(name))
                    {
                        throw Error("checkpoint tensor '" + name + "' was not used.");
                    }
                }
            }

            static std::runtime_error Error(const std::string &message)
            {
                return std::runtime_error("GPT2: " + message);
            }

            static std::string ShapeText(const std::vector<size_t> &shape)
            {
                std::string text = "[";
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
                }
                return text + "]";
            }

            GPT2<T> &m_model;
            Weights m_weights;
            std::set<std::string> m_used;
            size_t m_filled = 0;
        };
    }

    template <typename T>
    void LoadGPT2Weights(GPT2<T> &model, std::map<std::string, math::Tensor<T>> weights)
    {
        detail::GPT2WeightLoader<T>(model, std::move(weights)).Load();
    }

    template <typename T>
    void LoadGPT2Weights(GPT2<T> &model, const std::string &path)
    {
        LoadGPT2Weights(model, serialize::ReadSafetensors<T>(path));
    }

}
