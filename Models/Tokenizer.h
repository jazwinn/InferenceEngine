#pragma once

#ifndef PCRE2_CODE_UNIT_WIDTH
#define PCRE2_CODE_UNIT_WIDTH 8
#endif

#include <pcre2.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace models
{
    namespace detail
    {
        inline size_t Utf8Width(unsigned char lead)
        {
            if (lead < 0x80) return 1;
            if ((lead & 0xE0) == 0xC0) return 2;
            if ((lead & 0xF0) == 0xE0) return 3;
            if ((lead & 0xF8) == 0xF0) return 4;
            return 1;
        }

        inline std::string Utf8(uint32_t codepoint)
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
            else if (codepoint < 0x10000)
            {
                out += static_cast<char>(0xE0 | (codepoint >> 12));
                out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            else
            {
                out += static_cast<char>(0xF0 | (codepoint >> 18));
                out += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
                out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            return out;
        }

        class Regex
        {
        public:
            explicit Regex(const std::string& pattern)
            {
                int error = 0;
                PCRE2_SIZE offset = 0;
                pcre2_code* code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(),
                                                 PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr);
                if (!code)
                {
                    std::array<PCRE2_UCHAR, 256> message{};
                    pcre2_get_error_message(error, message.data(), message.size());
                    throw std::invalid_argument("Tokenizer: invalid regex at offset " + std::to_string(offset) + ": " +
                                                reinterpret_cast<const char*>(message.data()));
                }
                pcre2_jit_compile(code, PCRE2_JIT_COMPLETE);
                m_code.reset(code, pcre2_code_free);
            }

            std::vector<std::pair<size_t, size_t>> FindAll(std::string_view text) const
            {
                std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
                    pcre2_match_data_create_from_pattern(m_code.get(), nullptr), pcre2_match_data_free);

                std::vector<std::pair<size_t, size_t>> spans;
                size_t start = 0;
                uint32_t options = 0;
                while (start <= text.size())
                {
                    const int rc = pcre2_match(m_code.get(), reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(),
                                               start, options, match.get(), nullptr);
                    if (rc == PCRE2_ERROR_NOMATCH)
                    {
                        break;
                    }
                    if (rc < 0)
                    {
                        throw std::invalid_argument("Tokenizer: text is not valid UTF-8.");
                    }

                    const PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match.get());
                    const size_t begin = ovector[0];
                    const size_t end = ovector[1];
                    if (end > begin)
                    {
                        spans.emplace_back(begin, end);
                        start = end;
                    }
                    else
                    {
                        if (begin >= text.size())
                        {
                            break;
                        }
                        start = begin + Utf8Width(static_cast<unsigned char>(text[begin]));
                    }
                    options = PCRE2_NO_UTF_CHECK;
                }
                return spans;
            }

        private:
            std::shared_ptr<pcre2_code> m_code;
        };
    }

    class Tokenizer
    {
    public:
        using json = nlohmann::json;

        static Tokenizer FromFile(const std::string& path)
        {
            std::ifstream file(path);
            if (!file)
            {
                throw std::runtime_error("Tokenizer: failed to open " + path);
            }
            const json document = json::parse(file, nullptr, false);
            if (document.is_discarded() || !document.is_object())
            {
                throw std::runtime_error("Tokenizer: " + path + " is not a JSON object.");
            }
            return FromJson(document);
        }

        static Tokenizer FromJson(const json& document)
        {
            Tokenizer tokenizer;
            tokenizer.LoadModel(Required(document, "model"));
            tokenizer.LoadAddedTokens(document.value("added_tokens", json::array()));
            tokenizer.LoadNormalizer(document.value("normalizer", json()));
            tokenizer.LoadPreTokenizer(document.value("pre_tokenizer", json()));
            tokenizer.LoadPostProcessor(document.value("post_processor", json()));
            tokenizer.LoadDecoder(document.value("decoder", json()));
            tokenizer.BuildReverse();
            return tokenizer;
        }

        std::vector<int64_t> Encode(std::string_view text, bool add_special_tokens = true) const
        {
            std::vector<int64_t> ids;
            if (add_special_tokens)
            {
                ids.insert(ids.end(), m_prefix.begin(), m_prefix.end());
            }

            size_t segment = 0;
            size_t i = 0;
            while (i < text.size())
            {
                const std::optional<size_t> added = MatchAddedToken(text, i);
                if (!added)
                {
                    ++i;
                    continue;
                }
                EncodeSegment(text.substr(segment, i - segment), ids);
                ids.push_back(m_added[*added].id);
                i += m_added[*added].content.size();
                segment = i;
            }
            EncodeSegment(text.substr(segment), ids);

            if (add_special_tokens)
            {
                ids.insert(ids.end(), m_suffix.begin(), m_suffix.end());
            }
            return ids;
        }

        std::string Decode(const std::vector<int64_t>& ids, bool skip_special_tokens = false) const
        {
            std::string text;
            std::string pending;
            for (int64_t id : ids)
            {
                if (id < 0 || static_cast<size_t>(id) >= m_reverse.size() || m_reverse[static_cast<size_t>(id)].empty())
                {
                    throw std::out_of_range("Tokenizer: token id " + std::to_string(id) + " is outside the vocabulary.");
                }

                const auto added = m_addedById.find(id);
                if (added == m_addedById.end())
                {
                    pending += m_reverse[static_cast<size_t>(id)];
                    continue;
                }

                text += DecodeTokens(pending);
                pending.clear();
                const AddedToken& token = m_added[added->second];
                if (!(skip_special_tokens && token.special))
                {
                    text += token.content;
                }
            }
            return text + DecodeTokens(pending);
        }

        std::optional<int64_t> TokenToId(std::string_view token) const
        {
            const std::string key(token);
            for (const AddedToken& added : m_added)
            {
                if (added.content == key)
                {
                    return added.id;
                }
            }
            const auto entry = m_vocab.find(key);
            if (entry == m_vocab.end())
            {
                return std::nullopt;
            }
            return entry->second;
        }

        std::optional<std::string> IdToToken(int64_t id) const
        {
            if (id < 0 || static_cast<size_t>(id) >= m_reverse.size() || m_reverse[static_cast<size_t>(id)].empty())
            {
                return std::nullopt;
            }
            return m_reverse[static_cast<size_t>(id)];
        }

        bool IsSpecial(int64_t id) const
        {
            const auto added = m_addedById.find(id);
            return added != m_addedById.end() && m_added[added->second].special;
        }

        size_t VocabSize() const noexcept { return m_reverse.size(); }

    private:
        struct AddedToken
        {
            std::string content;
            int64_t id = 0;
            bool special = false;
        };

        struct PreTokenizerStep
        {
            std::optional<detail::Regex> split;
            bool byteLevel = false;
            bool addPrefixSpace = false;
        };

        Tokenizer() = default;

        static constexpr std::string_view kGPT2Pattern =
            R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

        static const json& Required(const json& object, const char* key)
        {
            if (!object.is_object() || !object.contains(key))
            {
                throw std::runtime_error(std::string("Tokenizer: missing '") + key + "'.");
            }
            return object.at(key);
        }

        static std::string TypeOf(const json& component, const char* section)
        {
            if (!component.is_object() || !component.contains("type") || !component.at("type").is_string())
            {
                throw std::runtime_error(std::string("Tokenizer: ") + section + " has no type.");
            }
            return component.at("type").get<std::string>();
        }

        static std::runtime_error Unsupported(const char* section, const std::string& type)
        {
            return std::runtime_error(std::string("Tokenizer: unsupported ") + section + " type '" + type + "'.");
        }

        void LoadModel(const json& model)
        {
            // Files written by older `tokenizers` versions (e.g. the original
            // openai-community/gpt2 tokenizer.json) omit the model type; a
            // model with vocab and merges is BPE.
            const bool untypedBpe = model.is_object() && !model.contains("type") && model.contains("vocab") && model.contains("merges");
            const std::string type = untypedBpe ? "BPE" : TypeOf(model, "model");
            if (type != "BPE")
            {
                throw Unsupported("model", type);
            }
            if (model.value("byte_fallback", false))
            {
                throw std::runtime_error("Tokenizer: BPE byte_fallback is not supported yet.");
            }

            for (const auto& [token, id] : Required(model, "vocab").items())
            {
                if (!id.is_number_integer() || id.get<int64_t>() < 0)
                {
                    throw std::runtime_error("Tokenizer: vocab entry '" + token + "' has an invalid id.");
                }
                if (id.get<int64_t>() > std::numeric_limits<uint32_t>::max())
                {
                    throw std::runtime_error("Tokenizer: vocab id for '" + token + "' does not fit in 32 bits.");
                }
                m_vocab.emplace(token, id.get<int64_t>());
            }

            int rank = 0;
            for (const json& merge : Required(model, "merges"))
            {
                std::string left;
                std::string right;
                if (merge.is_string())
                {
                    const std::string text = merge.get<std::string>();
                    const size_t space = text.find(' ');
                    if (space == std::string::npos)
                    {
                        throw std::runtime_error("Tokenizer: malformed merge '" + text + "'.");
                    }
                    left = text.substr(0, space);
                    right = text.substr(space + 1);
                }
                else if (merge.is_array() && merge.size() == 2 && merge[0].is_string() && merge[1].is_string())
                {
                    left = merge[0].get<std::string>();
                    right = merge[1].get<std::string>();
                }
                else
                {
                    throw std::runtime_error("Tokenizer: merges must be \"a b\" strings or [\"a\", \"b\"] pairs.");
                }
                const int64_t leftId = VocabId(left, "merge");
                const int64_t rightId = VocabId(right, "merge");
                const int64_t mergedId = VocabId(left + right, "merged token");
                m_merges.emplace(PairKey(leftId, rightId), MergeRule{rank++, mergedId});
            }

            m_ignoreMerges = model.value("ignore_merges", false);
            if (model.contains("unk_token") && model.at("unk_token").is_string())
            {
                const auto unk = m_vocab.find(model.at("unk_token").get<std::string>());
                if (unk == m_vocab.end())
                {
                    throw std::runtime_error("Tokenizer: unk_token is not in the vocabulary.");
                }
                m_unk = unk->second;
            }
        }

        void LoadAddedTokens(const json& tokens)
        {
            for (const json& token : tokens)
            {
                AddedToken added;
                added.content = Required(token, "content").get<std::string>();
                added.id = Required(token, "id").get<int64_t>();
                added.special = token.value("special", false);
                if (added.content.empty() || added.id < 0)
                {
                    throw std::runtime_error("Tokenizer: added token with empty content or negative id.");
                }
                if (token.value("lstrip", false) || token.value("rstrip", false) || token.value("single_word", false))
                {
                    throw std::runtime_error("Tokenizer: added token '" + added.content +
                                             "' uses lstrip/rstrip/single_word, which is not supported yet.");
                }
                m_addedById.emplace(added.id, m_added.size());
                m_added.push_back(std::move(added));
            }

            for (size_t i = 0; i < m_added.size(); ++i)
            {
                m_addedByFirstByte[static_cast<unsigned char>(m_added[i].content[0])].push_back(i);
            }
            for (auto& [byte, candidates] : m_addedByFirstByte)
            {
                std::sort(candidates.begin(), candidates.end(), [this](size_t a, size_t b) {
                    return m_added[a].content.size() > m_added[b].content.size();
                });
            }
        }

        void LoadNormalizer(const json& normalizer)
        {
            if (!normalizer.is_null())
            {
                throw Unsupported("normalizer", TypeOf(normalizer, "normalizer"));
            }
        }

        void LoadPreTokenizer(const json& pre)
        {
            if (pre.is_null())
            {
                return;
            }

            const std::string type = TypeOf(pre, "pre_tokenizer");
            if (type == "Sequence")
            {
                for (const json& step : Required(pre, "pretokenizers"))
                {
                    LoadPreTokenizer(step);
                }
                return;
            }

            if (!m_pre.empty() && m_pre.back().byteLevel)
            {
                throw std::runtime_error("Tokenizer: pre_tokenizer steps after ByteLevel are not supported.");
            }

            if (type == "Split")
            {
                const json& pattern = Required(pre, "pattern");
                std::string regex;
                if (pattern.contains("Regex"))
                {
                    regex = pattern.at("Regex").get<std::string>();
                }
                else if (pattern.contains("String"))
                {
                    for (char c : pattern.at("String").get<std::string>())
                    {
                        regex += std::string("\\x{") + ToHex(static_cast<unsigned char>(c)) + "}";
                    }
                }
                else
                {
                    throw std::runtime_error("Tokenizer: Split pattern must be Regex or String.");
                }

                const std::string behavior = pre.value("behavior", "Isolated");
                if (behavior != "Isolated" || pre.value("invert", false))
                {
                    throw std::runtime_error("Tokenizer: only Split behavior 'Isolated' without invert is supported.");
                }

                PreTokenizerStep step;
                step.split.emplace(regex);
                m_pre.push_back(std::move(step));
                return;
            }

            if (type == "ByteLevel")
            {
                PreTokenizerStep step;
                step.byteLevel = true;
                step.addPrefixSpace = pre.value("add_prefix_space", false);
                if (pre.value("use_regex", true))
                {
                    step.split.emplace(std::string(kGPT2Pattern));
                }
                m_pre.push_back(std::move(step));
                BuildByteMaps();
                return;
            }

            throw Unsupported("pre_tokenizer", type);
        }

        void LoadPostProcessor(const json& post)
        {
            if (post.is_null())
            {
                return;
            }

            const std::string type = TypeOf(post, "post_processor");
            if (type == "Sequence")
            {
                for (const json& processor : Required(post, "processors"))
                {
                    LoadPostProcessor(processor);
                }
                return;
            }
            if (type == "ByteLevel")
            {
                return;
            }
            if (type != "TemplateProcessing")
            {
                throw Unsupported("post_processor", type);
            }

            const json& specials = post.value("special_tokens", json::object());
            bool seenSequence = false;
            for (const json& piece : Required(post, "single"))
            {
                if (piece.contains("Sequence"))
                {
                    seenSequence = true;
                    continue;
                }
                const std::string name = Required(Required(piece, "SpecialToken"), "id").get<std::string>();
                const json& entry = Required(specials, name.c_str());
                std::vector<int64_t>& target = seenSequence ? m_suffix : m_prefix;
                for (const json& id : Required(entry, "ids"))
                {
                    target.push_back(id.get<int64_t>());
                }
            }
        }

        void LoadDecoder(const json& decoder)
        {
            if (decoder.is_null())
            {
                return;
            }
            const std::string type = TypeOf(decoder, "decoder");
            if (type != "ByteLevel")
            {
                throw Unsupported("decoder", type);
            }
            m_byteLevelDecoder = true;
            BuildByteMaps();
        }

        void BuildByteMaps()
        {
            if (!m_charToByte.empty())
            {
                return;
            }
            uint32_t next = 0;
            for (uint32_t byte = 0; byte < 256; ++byte)
            {
                const bool printable = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255);
                const std::string character = detail::Utf8(printable ? byte : 256 + next++);
                m_byteToChar[byte] = character;
                m_charToByte.emplace(character, static_cast<char>(byte));
            }
        }

        void BuildReverse()
        {
            int64_t largest = -1;
            for (const auto& [token, id] : m_vocab) largest = std::max(largest, id);
            for (const AddedToken& added : m_added) largest = std::max(largest, added.id);

            m_reverse.assign(static_cast<size_t>(largest + 1), std::string());
            for (const auto& [token, id] : m_vocab) m_reverse[static_cast<size_t>(id)] = token;
            for (const AddedToken& added : m_added) m_reverse[static_cast<size_t>(added.id)] = added.content;
        }

        std::optional<size_t> MatchAddedToken(std::string_view text, size_t at) const
        {
            const auto candidates = m_addedByFirstByte.find(static_cast<unsigned char>(text[at]));
            if (candidates == m_addedByFirstByte.end())
            {
                return std::nullopt;
            }
            for (size_t index : candidates->second)
            {
                const std::string& content = m_added[index].content;
                if (text.compare(at, content.size(), content) == 0)
                {
                    return index;
                }
            }
            return std::nullopt;
        }

        void EncodeSegment(std::string_view segment, std::vector<int64_t>& ids) const
        {
            if (segment.empty())
            {
                return;
            }
            for (const std::string& word : PreTokenize(segment))
            {
                const std::vector<int64_t>& tokens = EncodeWord(word);
                ids.insert(ids.end(), tokens.begin(), tokens.end());
            }
        }

        std::vector<std::string> PreTokenize(std::string_view segment) const
        {
            std::vector<std::string> pieces{std::string(segment)};
            bool mapped = false;

            for (const PreTokenizerStep& step : m_pre)
            {
                if (step.addPrefixSpace)
                {
                    for (std::string& piece : pieces)
                    {
                        if (piece.empty() || piece[0] != ' ') piece.insert(piece.begin(), ' ');
                    }
                }
                if (step.split)
                {
                    std::vector<std::string> next;
                    for (const std::string& piece : pieces)
                    {
                        size_t last = 0;
                        for (const auto& [begin, end] : step.split->FindAll(piece))
                        {
                            if (begin > last) next.push_back(piece.substr(last, begin - last));
                            next.push_back(piece.substr(begin, end - begin));
                            last = end;
                        }
                        if (last < piece.size()) next.push_back(piece.substr(last));
                    }
                    pieces = std::move(next);
                }
                mapped = mapped || step.byteLevel;
            }

            if (mapped)
            {
                for (std::string& piece : pieces)
                {
                    std::string out;
                    out.reserve(piece.size() * 2);
                    for (unsigned char byte : piece) out += m_byteToChar[byte];
                    piece = std::move(out);
                }
            }
            return pieces;
        }

        const std::vector<int64_t>& EncodeWord(const std::string& word) const
        {
            const auto cached = m_cache.find(word);
            if (cached != m_cache.end())
            {
                return cached->second;
            }

            std::vector<int64_t> ids;
            const auto whole = m_vocab.find(word);
            if (m_ignoreMerges && whole != m_vocab.end())
            {
                ids.push_back(whole->second);
            }
            else
            {
                ids = Merge(CharacterIds(word));
            }
            return m_cache.emplace(word, std::move(ids)).first->second;
        }

        std::vector<int64_t> CharacterIds(const std::string& word) const
        {
            std::vector<int64_t> ids;
            ids.reserve(word.size());
            for (size_t i = 0; i < word.size();)
            {
                const size_t width = std::min(detail::Utf8Width(static_cast<unsigned char>(word[i])), word.size() - i);
                const std::string character = word.substr(i, width);
                const auto entry = m_vocab.find(character);
                if (entry != m_vocab.end())
                {
                    ids.push_back(entry->second);
                }
                else if (m_unk)
                {
                    ids.push_back(*m_unk);
                }
                else
                {
                    throw std::runtime_error("Tokenizer: no vocabulary entry for '" + character + "' and no unk_token.");
                }
                i += width;
            }
            return ids;
        }

        std::vector<int64_t> Merge(std::vector<int64_t> ids) const
        {
            while (ids.size() > 1)
            {
                size_t best = ids.size();
                int bestRank = std::numeric_limits<int>::max();
                int64_t mergedId = -1;
                for (size_t i = 0; i + 1 < ids.size(); ++i)
                {
                    const auto rule = m_merges.find(PairKey(ids[i], ids[i + 1]));
                    if (rule != m_merges.end() && rule->second.rank < bestRank)
                    {
                        bestRank = rule->second.rank;
                        best = i;
                        mergedId = rule->second.merged;
                    }
                }
                if (best == ids.size())
                {
                    break;
                }
                ids[best] = mergedId;
                ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(best) + 1);
            }
            return ids;
        }

        std::string DecodeTokens(const std::string& joined) const
        {
            if (!m_byteLevelDecoder)
            {
                return joined;
            }
            std::string text;
            text.reserve(joined.size());
            for (size_t i = 0; i < joined.size();)
            {
                const size_t width = std::min(detail::Utf8Width(static_cast<unsigned char>(joined[i])), joined.size() - i);
                const std::string character = joined.substr(i, width);
                const auto byte = m_charToByte.find(character);
                text += byte != m_charToByte.end() ? std::string(1, byte->second) : character;
                i += width;
            }
            return text;
        }

        static uint64_t PairKey(int64_t left, int64_t right)
        {
            return (static_cast<uint64_t>(left) << 32) | static_cast<uint32_t>(right);
        }

        int64_t VocabId(const std::string& token, const char* role) const
        {
            const auto entry = m_vocab.find(token);
            if (entry == m_vocab.end())
            {
                throw std::runtime_error(std::string("Tokenizer: ") + role + " '" + token + "' is not in the vocabulary.");
            }
            return entry->second;
        }

        static std::string ToHex(unsigned char byte)
        {
            static constexpr char digits[] = "0123456789abcdef";
            return {digits[byte >> 4], digits[byte & 0xF]};
        }

        std::unordered_map<std::string, int64_t> m_vocab;
        struct MergeRule
        {
            int rank = 0;
            int64_t merged = 0;
        };

        std::unordered_map<uint64_t, MergeRule> m_merges;
        bool m_ignoreMerges = false;
        std::optional<int64_t> m_unk;

        std::vector<AddedToken> m_added;
        std::unordered_map<int64_t, size_t> m_addedById;
        std::unordered_map<unsigned char, std::vector<size_t>> m_addedByFirstByte;

        std::vector<PreTokenizerStep> m_pre;
        std::vector<int64_t> m_prefix;
        std::vector<int64_t> m_suffix;
        bool m_byteLevelDecoder = false;

        std::array<std::string, 256> m_byteToChar;
        std::unordered_map<std::string, char> m_charToByte;
        std::vector<std::string> m_reverse;
        mutable std::unordered_map<std::string, std::vector<int64_t>> m_cache;
    };
}
