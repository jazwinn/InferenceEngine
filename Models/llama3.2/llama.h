#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Math/Tensor.h"
#include "NeuralNet/Linear.h"
#include "NeuralNet/Transformer/Decoder.h"
#include "NeuralNet/Transformer/Normal/Norm.h"
#include "NeuralNet/Transformer/Normal/RMSNorm.h"
#include "Serialize/Safetensors.h"
#include "Models/CausalModel.h"

namespace models::llama
{

    using math::Tensor;
    using nn::Linear;
    using nn::transformer::DecoderBlock;
    using nn::transformer::norm::Norm;
    using nn::transformer::norm::RMSNorm;

    template <typename T>
    class LlamaWeightLoader
    {
    public:
        using Weights = std::map<std::string, Tensor<T>>;

        LlamaWeightLoader(CausalModel<T> &model, Weights weights)
            : m_model(model), m_weights(std::move(weights))
        {
        }

        void Load()
        {
            Fill(m_model.wte->weight, "model.embed_tokens.weight");
            for (size_t i = 0; i < m_model.transformer->blocks.size(); ++i)
            {
                LoadBlock(*m_model.transformer->blocks[i], "model.layers." + std::to_string(i) + ".");
            }
            if (!m_model.transformer->finalNorm)
            {
                throw Error("the model has no final norm; check the config's norm_placement.");
            }
            LoadNorm(*m_model.transformer->finalNorm, "model.norm");
            if (m_model.lm_head)
            {
                Fill(m_model.lm_head->weight, "lm_head.weight");
            }
            VerifyNothingWasMissed();
        }

    private:
        void LoadBlock(DecoderBlock<T> &block, const std::string &prefix)
        {
            LoadNorm(*block.attenNorm, prefix + "input_layernorm");
            LoadLinear(*block.attention->query, prefix + "self_attn.q_proj");
            LoadLinear(*block.attention->key, prefix + "self_attn.k_proj");
            LoadLinear(*block.attention->value, prefix + "self_attn.v_proj");
            LoadLinear(*block.attention->output_proj, prefix + "self_attn.o_proj");

            LoadNorm(*block.mlpNorm, prefix + "post_attention_layernorm");
            if (!block.mlp->gate)
            {
                throw Error("'" + prefix + "mlp' must be gated; check the config's mlp_type.");
            }
            LoadLinear(*block.mlp->gate, prefix + "mlp.gate_proj");
            LoadLinear(*block.mlp->up, prefix + "mlp.up_proj");
            LoadLinear(*block.mlp->down, prefix + "mlp.down_proj");
        }

        void LoadNorm(Norm<T> &norm, const std::string &prefix)
        {
            auto *rmsNorm = dynamic_cast<RMSNorm<T> *>(&norm);
            if (!rmsNorm)
            {
                throw Error("'" + prefix + "' must be an RMSNorm; check the config's norm_type.");
            }
            Fill(rmsNorm->weight, prefix + ".weight");
        }

        void LoadLinear(Linear<T> &linear, const std::string &prefix)
        {
            Fill(linear.weight, prefix + ".weight");
            if (linear.bias)
            {
                Fill(linear.bias, prefix + ".bias");
            }
        }

        void Fill(const std::shared_ptr<Tensor<T>> &parameter, const std::string &name)
        {
            auto node = m_weights.extract(name);
            if (node.empty())
            {
                throw Error("checkpoint has no tensor '" + name + "'.");
            }
            if (!parameter)
            {
                throw Error("no parameter to receive '" + name + "'.");
            }
            if (parameter->Shape() != node.mapped().Shape())
            {
                throw Error("'" + name + "' is " + ShapeText(node.mapped().Shape()) + " but the model expects " +
                            ShapeText(parameter->Shape()) + ".");
            }
            *parameter = std::move(node.mapped());
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
            if (!m_weights.empty())
            {
                throw Error("checkpoint tensor '" + m_weights.begin()->first + "' was not used.");
            }
        }

        static std::runtime_error Error(const std::string &message)
        {
            return std::runtime_error("Llama: " + message);
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

        CausalModel<T> &m_model;
        Weights m_weights;
        size_t m_filled = 0;
    };

    template <typename T>
    void LoadLlamaWeights(CausalModel<T> &model, std::map<std::string, Tensor<T>> weights)
    {
        LlamaWeightLoader<T>(model, std::move(weights)).Load();
    }

    template <typename T>
    void LoadLlamaWeights(CausalModel<T> &model, const std::string &path)
    {
        LoadLlamaWeights(model, serialize::ReadSafetensors<T>(path));
    }

}
