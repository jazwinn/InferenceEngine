#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Tensor/Tensor.h"
#include "Models/WeightLoader.h"

namespace models::gpt2
{

    namespace detail
    {
        inline std::runtime_error CheckpointError(const std::string &message)
        {
            return std::runtime_error("GPT2 checkpoint: " + message);
        }

        inline bool IsBuffer(const std::string &name)
        {
            return name.ends_with(".attn.bias") || name.ends_with(".attn.masked_bias");
        }

        inline bool IsConv1DWeight(const std::string &name)
        {
            return name.ends_with(".c_attn.weight") || name.ends_with(".c_proj.weight") || name.ends_with(".c_fc.weight");
        }

        inline bool IsFusedQKV(const std::string &name)
        {
            return name.ends_with(".attn.c_attn.weight") || name.ends_with(".attn.c_attn.bias");
        }

        template <typename T>
        void Insert(Weights<T> &weights, std::string name, tensor::Tensor<T> tensor)
        {
            if (!weights.emplace(name, std::move(tensor)).second)
            {
                throw CheckpointError("tensor '" + name + "' appears twice.");
            }
        }

        template <typename T>
        void SplitQKV(Weights<T> &weights, const std::string &name, const tensor::Tensor<T> &fused)
        {
            const size_t width = fused.Shape()[0];
            if (width % 3 != 0 || (fused.Rank() == 2 && width != 3 * fused.Shape()[1]))
            {
                throw CheckpointError("'" + name + "' cannot be split into equal q, k and v parts.");
            }

            const size_t d = width / 3;
            const size_t stem = name.rfind("c_attn.");
            const std::string prefix = name.substr(0, stem);
            const std::string suffix = name.substr(stem + 6);
            const std::array<const char *, 3> parts{"q_proj", "k_proj", "v_proj"};

            for (size_t part = 0; part < parts.size(); ++part)
            {
                Insert(weights, prefix + parts[part] + suffix, fused.Slice(0, part * d, d).Clone());
            }
        }

        inline void AddLinear(std::vector<RenameRule> &rules, const std::string &from, const std::string &to)
        {
            rules.push_back({from + ".weight", to + ".weight"});
            rules.push_back({from + ".bias", to + ".bias"});
        }

        inline void AddLayerNorm(std::vector<RenameRule> &rules, const std::string &from, const std::string &to)
        {
            rules.push_back({from + ".weight", to + ".gamma"});
            rules.push_back({from + ".bias", to + ".beta"});
        }

        inline const std::vector<RenameRule> &Rules()
        {
            static const std::vector<RenameRule> rules = []
            {
                std::vector<RenameRule> r;
                AddLayerNorm(r, "ln_f", "transformer.finalNorm");
                AddLayerNorm(r, "h.{i}.ln_1", "transformer.h.{i}.attenNorm");
                AddLayerNorm(r, "h.{i}.ln_2", "transformer.h.{i}.mlpNorm");
                AddLinear(r, "h.{i}.attn.q_proj", "transformer.h.{i}.attention.query");
                AddLinear(r, "h.{i}.attn.k_proj", "transformer.h.{i}.attention.key");
                AddLinear(r, "h.{i}.attn.v_proj", "transformer.h.{i}.attention.value");
                AddLinear(r, "h.{i}.attn.c_proj", "transformer.h.{i}.attention.output_proj");
                AddLinear(r, "h.{i}.mlp.c_fc", "transformer.h.{i}.mlp.up");
                AddLinear(r, "h.{i}.mlp.c_proj", "transformer.h.{i}.mlp.down");
                return r;
            }();
            return rules;
        }
    }

    template <typename T>
    Weights<T> ConvertCheckpoint(Weights<T> checkpoint)
    {
        Weights<T> converted;
        while (!checkpoint.empty())
        {
            auto node = checkpoint.extract(checkpoint.begin());
            const std::string &key = node.key();
            std::string name = key.starts_with("transformer.") ? key.substr(12) : key;
            tensor::Tensor<T> tensor = std::move(node.mapped());

            if (detail::IsBuffer(name))
            {
                continue;
            }
            if (detail::IsConv1DWeight(name))
            {
                if (tensor.Rank() != 2)
                {
                    throw detail::CheckpointError("'" + name + "' must be a matrix.");
                }
                tensor = tensor.Transpose().Clone();
            }
            if (detail::IsFusedQKV(name))
            {
                detail::SplitQKV(converted, name, tensor);
                continue;
            }
            detail::Insert(converted, std::move(name), std::move(tensor));
        }
        return Rename(std::move(converted), detail::Rules());
    }

    template <typename T>
    Weights<T> ReadCheckpoint(const std::string &path)
    {
        return ConvertCheckpoint(serialize::ReadSafetensors<T>(path));
    }

}
