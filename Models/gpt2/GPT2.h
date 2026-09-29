#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Math/Tensor.h"
#include "NeuralNet/Linear.h"
#include "NeuralNet/Transformer/Decoder.h"
#include "NeuralNet/Transformer/GroupQueryAttention.h"
#include "NeuralNet/Transformer/Normal/Norm.h"
#include "NeuralNet/Transformer/Normal/LayerNorm.h"
#include "Serialize/Safetensors.h"
#include "Models/CausalModel.h"

namespace models::gpt2
{

    using math::Tensor;
    using nn::Linear;
    using nn::transformer::DecoderBlock;
    using nn::transformer::GroupQueryAttention;
    using nn::transformer::norm::Norm;
    using nn::transformer::norm::LayerNorm;

    namespace detail
    {
        template <typename T>
        class GPT2WeightLoader
        {
        public:
            using Weights = std::map<std::string, math::Tensor<T>>;

            GPT2WeightLoader(CausalModel<T> &model, Weights weights)
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
                LoadNorm(*m_model.transformer->finalNorm, "ln_f");
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

            void LoadBlock(DecoderBlock<T> &block, const std::string &prefix)
            {
                LoadNorm(*block.attenNorm, prefix + "ln_1");
                LoadAttention(*block.attention, prefix + "attn.");
                LoadNorm(*block.mlpNorm, prefix + "ln_2");
                LoadFeedForward(block, prefix + "mlp.");
            }

            void LoadNorm(Norm<T> &norm, const std::string &prefix)
            {
                auto *layerNorm = dynamic_cast<LayerNorm<T> *>(&norm);
                if (!layerNorm)
                {
                    throw Error("'" + prefix + "' must be a LayerNorm; check the config's norm_type.");
                }
                Copy(layerNorm->gamma, prefix + ".weight", Take(prefix + ".weight"));
                Copy(layerNorm->beta, prefix + ".bias", Take(prefix + ".bias"));
            }

            void LoadAttention(GroupQueryAttention<T> &attention, const std::string &prefix)
            {
                LoadFusedQKV(attention, prefix);
                LoadLinear(*attention.output_proj, prefix + "c_proj");
            }

            void LoadFusedQKV(GroupQueryAttention<T> &attention, const std::string &prefix)
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

            void LoadFeedForward(DecoderBlock<T> &block, const std::string &prefix)
            {
                LoadLinear(*block.mlp->up, prefix + "c_fc");
                LoadLinear(*block.mlp->down, prefix + "c_proj");
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

            CausalModel<T> &m_model;
            Weights m_weights;
            std::set<std::string> m_used;
            size_t m_filled = 0;
        };
    }

    template <typename T>
    void LoadGPT2Weights(CausalModel<T> &model, std::map<std::string, math::Tensor<T>> weights)
    {
        detail::GPT2WeightLoader<T>(model, std::move(weights)).Load();
    }

    template <typename T>
    void LoadGPT2Weights(CausalModel<T> &model, const std::string &path)
    {
        LoadGPT2Weights(model, serialize::ReadSafetensors<T>(path));
    }

    class GPT2Tokenizer
    {
    public:
        GPT2Tokenizer(const std::string &vocab_json, const std::string &merges_txt)
        {
            BuildByteMaps();
            LoadVocab(vocab_json);
            LoadMerges(merges_txt);
        }

        std::vector<int64_t> Encode(std::string_view text) const
        {
            std::vector<int64_t> ids;
            for (const std::string &word : SplitWords(text))
            {
                const std::vector<int64_t> &tokens = EncodeWord(MapBytes(word));
                ids.insert(ids.end(), tokens.begin(), tokens.end());
            }
            return ids;
        }

        std::string Decode(const std::vector<int64_t> &ids) const
        {
            return UnmapBytes(Join(ids));
        }

        size_t VocabSize() const noexcept { return m_reverse.size(); }

    private:
        using Pair = std::pair<std::string, std::string>;

        std::string MapBytes(const std::string &word) const
        {
            std::string mapped;
            mapped.reserve(word.size());
            for (unsigned char byte : word)
            {
                mapped += m_byte_to_char[byte];
            }
            return mapped;
        }

        const std::vector<int64_t> &EncodeWord(const std::string &mapped) const
        {
            const auto cached = m_cache.find(mapped);
            if (cached != m_cache.end())
            {
                return cached->second;
            }
            return m_cache.emplace(mapped, Lookup(Merge(SplitCharacters(mapped)))).first->second;
        }

        std::vector<std::string> Merge(std::vector<std::string> symbols) const
        {
            while (symbols.size() > 1)
            {
                const std::optional<size_t> best = BestPair(symbols);
                if (!best)
                {
                    break;
                }
                symbols = MergePair(symbols, symbols[*best], symbols[*best + 1]);
            }
            return symbols;
        }

        std::optional<size_t> BestPair(const std::vector<std::string> &symbols) const
        {
            std::optional<size_t> best;
            int best_rank = std::numeric_limits<int>::max();

            for (size_t i = 0; i + 1 < symbols.size(); ++i)
            {
                const auto rule = m_ranks.find(Pair{symbols[i], symbols[i + 1]});
                if (rule != m_ranks.end() && rule->second < best_rank)
                {
                    best_rank = rule->second;
                    best = i;
                }
            }
            return best;
        }

        static std::vector<std::string> MergePair(const std::vector<std::string> &symbols,
                                                  const std::string &left, const std::string &right)
        {
            std::vector<std::string> merged;
            merged.reserve(symbols.size());

            for (size_t i = 0; i < symbols.size();)
            {
                const bool matches = i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right;
                merged.push_back(matches ? left + right : symbols[i]);
                i += matches ? 2 : 1;
            }
            return merged;
        }

        std::vector<int64_t> Lookup(const std::vector<std::string> &symbols) const
        {
            std::vector<int64_t> ids;
            ids.reserve(symbols.size());

            for (const std::string &symbol : symbols)
            {
                const auto entry = m_vocab.find(symbol);
                if (entry == m_vocab.end())
                {
                    throw std::runtime_error("GPT2Tokenizer: no vocabulary entry for '" + symbol + "'.");
                }
                ids.push_back(entry->second);
            }
            return ids;
        }

        std::string Join(const std::vector<int64_t> &ids) const
        {
            std::string mapped;
            for (int64_t id : ids)
            {
                if (id < 0 || static_cast<size_t>(id) >= m_reverse.size())
                {
                    throw std::out_of_range("GPT2Tokenizer: token id is outside the vocabulary.");
                }
                mapped += m_reverse[static_cast<size_t>(id)];
            }
            return mapped;
        }

        std::string UnmapBytes(const std::string &mapped) const
        {
            std::string text;
            for (size_t i = 0; i < mapped.size();)
            {
                const size_t width = Utf8Width(mapped[i]);
                const auto byte = m_char_to_byte.find(mapped.substr(i, width));
                if (byte == m_char_to_byte.end())
                {
                    throw std::runtime_error("GPT2Tokenizer: token contains a character outside the byte map.");
                }
                text += byte->second;
                i += width;
            }
            return text;
        }

        enum class CharClass
        {
            Letter,
            Number,
            Other
        };

        static size_t Utf8Width(char first)
        {
            const unsigned char lead = static_cast<unsigned char>(first);
            if (lead < 0x80) return 1;
            if ((lead & 0xE0) == 0xC0) return 2;
            if ((lead & 0xF0) == 0xE0) return 3;
            return 4;
        }

        static bool IsSpace(char c)
        {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
        }

        static CharClass ClassOf(char c)
        {
            const unsigned char value = static_cast<unsigned char>(c);
            if (value >= 0x80) return CharClass::Letter;
            if ((value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z')) return CharClass::Letter;
            if (value >= '0' && value <= '9') return CharClass::Number;
            return CharClass::Other;
        }

        static std::optional<size_t> ContractionLength(std::string_view text, size_t at)
        {
            static const std::array<std::string_view, 7> contractions{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

            for (std::string_view contraction : contractions)
            {
                if (text.size() - at >= contraction.size() && text.compare(at, contraction.size(), contraction) == 0)
                {
                    return contraction.size();
                }
            }
            return std::nullopt;
        }

        static size_t EndOfWord(std::string_view text, size_t at)
        {
            const size_t first = text[at] == ' ' ? at + 1 : at;
            if (first >= text.size() || IsSpace(text[first]))
            {
                return at;
            }

            const CharClass group = ClassOf(text[first]);
            size_t end = first;
            while (end < text.size() && !IsSpace(text[end]) && ClassOf(text[end]) == group)
            {
                end += Utf8Width(text[end]);
            }
            return end;
        }

        static size_t EndOfSpaces(std::string_view text, size_t at)
        {
            size_t end = at;
            while (end < text.size() && IsSpace(text[end]))
            {
                ++end;
            }
            if (end < text.size() && end - at > 1)
            {
                --end;
            }
            return end;
        }

        static std::vector<std::string> SplitWords(std::string_view text)
        {
            std::vector<std::string> words;

            for (size_t i = 0; i < text.size();)
            {
                size_t end = i;

                if (const std::optional<size_t> contraction = ContractionLength(text, i))
                {
                    end = i + *contraction;
                }
                else if (const size_t word = EndOfWord(text, i); word > i)
                {
                    end = word;
                }
                else
                {
                    end = EndOfSpaces(text, i);
                }

                words.emplace_back(text.substr(i, end - i));
                i = end;
            }
            return words;
        }

        static std::vector<std::string> SplitCharacters(const std::string &mapped)
        {
            std::vector<std::string> symbols;
            for (size_t i = 0; i < mapped.size();)
            {
                const size_t width = Utf8Width(mapped[i]);
                symbols.push_back(mapped.substr(i, width));
                i += width;
            }
            return symbols;
        }

        static std::string Utf8(int codepoint)
        {
            std::string out;
            if (codepoint < 0x80)
            {
                out += static_cast<char>(codepoint);
            }
            else if (codepoint < 0x800)
            {
                out += static_cast<char>(0xC0 | (codepoint >> 6));
                out += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            else
            {
                out += static_cast<char>(0xE0 | (codepoint >> 12));
                out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            return out;
        }

        static bool IsPrintable(int byte)
        {
            return (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255);
        }

        void BuildByteMaps()
        {
            int next = 0;
            for (int byte = 0; byte < 256; ++byte)
            {
                const int codepoint = IsPrintable(byte) ? byte : 256 + next++;
                const std::string character = Utf8(codepoint);

                m_byte_to_char[static_cast<size_t>(byte)] = character;
                m_char_to_byte[character] = static_cast<char>(byte);
            }
        }

        void LoadVocab(const std::string &path)
        {
            std::ifstream file(path);
            if (!file)
            {
                throw std::runtime_error("GPT2Tokenizer: failed to open " + path);
            }

            const nlohmann::json json = nlohmann::json::parse(file, nullptr, false);
            if (json.is_discarded() || !json.is_object())
            {
                throw std::runtime_error("GPT2Tokenizer: " + path + " is not a JSON object.");
            }

            m_reverse.resize(json.size());
            for (const auto &[token, id] : json.items())
            {
                if (!id.is_number_integer())
                {
                    throw std::runtime_error("GPT2Tokenizer: vocab entry '" + token + "' has a non-integer id.");
                }

                const int64_t value = id.get<int64_t>();
                if (value < 0 || static_cast<size_t>(value) >= m_reverse.size())
                {
                    throw std::runtime_error("GPT2Tokenizer: vocab id out of range for '" + token + "'.");
                }

                m_vocab.emplace(token, value);
                m_reverse[static_cast<size_t>(value)] = token;
            }
        }

        void LoadMerges(const std::string &path)
        {
            std::ifstream file(path);
            if (!file)
            {
                throw std::runtime_error("GPT2Tokenizer: failed to open " + path);
            }

            std::string line;
            int rank = 0;
            while (std::getline(file, line))
            {
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }
                if (line.empty() || line[0] == '#')
                {
                    continue;
                }

                const size_t space = line.find(' ');
                if (space == std::string::npos)
                {
                    throw std::runtime_error("GPT2Tokenizer: malformed merge line: " + line);
                }

                m_ranks.emplace(Pair{line.substr(0, space), line.substr(space + 1)}, rank++);
            }

            if (m_ranks.empty())
            {
                throw std::runtime_error("GPT2Tokenizer: " + path + " has no merges.");
            }
        }

        std::unordered_map<std::string, int64_t> m_vocab;
        std::vector<std::string> m_reverse;
        std::map<Pair, int> m_ranks;
        std::array<std::string, 256> m_byte_to_char;
        std::unordered_map<std::string, std::string> m_char_to_byte;
        mutable std::unordered_map<std::string, std::vector<int64_t>> m_cache;
    };

}
