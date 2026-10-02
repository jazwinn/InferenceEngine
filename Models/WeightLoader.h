#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Math/Tensor.h"
#include "NeuralNet/Module.h"
#include "Serialize/Safetensors.h"

namespace models
{

    template <typename T>
    using Weights = std::map<std::string, math::Tensor<T>>;

    struct RenameRule
    {
        std::string from;
        std::string to;
    };

    namespace detail
    {
        inline std::runtime_error LoaderError(const std::string &message)
        {
            return std::runtime_error("WeightLoader: " + message);
        }

        inline std::string ShapeText(const std::vector<size_t> &shape)
        {
            std::string text = "[";
            for (size_t i = 0; i < shape.size(); ++i)
            {
                text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
            }
            return text + "]";
        }

        inline std::string ListText(const std::vector<std::string> &names, size_t limit = 5)
        {
            std::string text;
            for (size_t i = 0; i < names.size() && i < limit; ++i)
            {
                text += (i == 0 ? "'" : ", '") + names[i] + "'";
            }
            if (names.size() > limit)
            {
                text += " and " + std::to_string(names.size() - limit) + " more";
            }
            return text;
        }

        class CompiledRule
        {
        public:
            explicit CompiledRule(const RenameRule &rule)
                : m_pattern(ToRegex(rule.from)), m_target(rule.to), m_indexed(rule.from.find(kIndex) != std::string::npos)
            {
                if (!m_indexed && rule.to.find(kIndex) != std::string::npos)
                {
                    throw LoaderError("rule '" + rule.from + "' -> '" + rule.to + "' uses {i} only in its target.");
                }
            }

            bool Apply(const std::string &name, std::string &renamed) const
            {
                std::smatch match;
                if (!std::regex_match(name, match, m_pattern))
                {
                    return false;
                }
                renamed = m_indexed ? ReplaceAll(m_target, kIndex, match[1].str()) : m_target;
                return true;
            }

        private:
            static constexpr const char *kIndex = "{i}";

            static std::regex ToRegex(const std::string &pattern)
            {
                static const std::string special = R"(\^$.|?*+()[]{})";
                std::string regex;
                bool indexed = false;
                for (size_t i = 0; i < pattern.size(); ++i)
                {
                    if (pattern.compare(i, 3, kIndex) == 0)
                    {
                        if (indexed)
                        {
                            throw LoaderError("rule '" + pattern + "' may use {i} only once.");
                        }
                        regex += R"((\d+))";
                        indexed = true;
                        i += 2;
                        continue;
                    }
                    if (special.find(pattern[i]) != std::string::npos)
                    {
                        regex += '\\';
                    }
                    regex += pattern[i];
                }
                return std::regex(regex);
            }

            static std::string ReplaceAll(std::string text, const std::string &from, const std::string &to)
            {
                for (size_t at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size()))
                {
                    text.replace(at, from.size(), to);
                }
                return text;
            }

            std::regex m_pattern;
            std::string m_target;
            bool m_indexed;
        };
    }

    template <typename T>
    Weights<T> Rename(Weights<T> weights, const std::vector<RenameRule> &rules)
    {
        std::vector<detail::CompiledRule> compiled(rules.begin(), rules.end());

        Weights<T> renamed;
        while (!weights.empty())
        {
            auto node = weights.extract(weights.begin());
            std::string name = node.key();
            for (const detail::CompiledRule &rule : compiled)
            {
                if (rule.Apply(node.key(), name))
                {
                    break;
                }
            }
            if (renamed.contains(name))
            {
                throw detail::LoaderError("two checkpoint tensors are renamed to '" + name + "'.");
            }
            node.key() = std::move(name);
            renamed.insert(std::move(node));
        }
        return renamed;
    }

    template <typename T>
    void LoadWeights(nn::Module<T> &model, Weights<T> weights)
    {
        const auto parameters = model.NamedParameters();

        std::vector<std::string> missing;
        for (const auto &[name, parameter] : parameters)
        {
            const auto entry = weights.find(name);
            if (entry == weights.end())
            {
                missing.push_back(name);
                continue;
            }
            if (!parameter)
            {
                throw detail::LoaderError("parameter '" + name + "' is null.");
            }
            if (parameter->Shape() != entry->second.Shape())
            {
                throw detail::LoaderError("'" + name + "' is " + detail::ShapeText(entry->second.Shape()) +
                                          " but the model expects " + detail::ShapeText(parameter->Shape()) + ".");
            }
        }

        std::vector<std::string> unused;
        for (const auto &[name, tensor] : weights)
        {
            const bool wanted = std::any_of(parameters.begin(), parameters.end(),
                                            [&](const auto &parameter) { return parameter.name == name; });
            if (!wanted)
            {
                unused.push_back(name);
            }
        }

        if (!missing.empty() || !unused.empty())
        {
            std::string message;
            if (!missing.empty())
            {
                message += "checkpoint is missing " + detail::ListText(missing) + ".";
            }
            if (!unused.empty())
            {
                message += (message.empty() ? "" : " ") + std::string("checkpoint has unused ") + detail::ListText(unused) + ".";
            }
            throw detail::LoaderError(message);
        }

        for (const auto &[name, parameter] : parameters)
        {
            *parameter = std::move(weights.at(name));
        }
    }

    template <typename T>
    void LoadWeights(nn::Module<T> &model, Weights<T> weights, const std::vector<RenameRule> &rules)
    {
        LoadWeights(model, Rename(std::move(weights), rules));
    }

    template <typename T>
    void LoadWeights(nn::Module<T> &model, const std::string &path, const std::vector<RenameRule> &rules = {})
    {
        LoadWeights(model, Rename(serialize::ReadSafetensors<T>(path), rules));
    }

}
