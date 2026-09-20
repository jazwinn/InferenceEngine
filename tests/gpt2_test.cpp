#include <gtest/gtest.h>

#include <Models/GPT2.h>
#include <Models/ModelConfigs.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using llm::GPT2;
using llm::GPT2Tokenizer;
using math::Tensor;
using transformer::TransformerConfig;

namespace {

TransformerConfig TinyConfig() {
    TransformerConfig c = models::gpt2::Small();
    c.vocab_size = 11;
    c.d_model = 8;
    c.num_layers = 2;
    c.num_heads = 2;
    c.ffn_dim = 32;
    c.max_seq_len = 6;
    return c;
}

void Seed(nn::Module<float>& m) {
    int k = 0;
    for (const auto& p : m.NamedParameters()) {
        float* data = p.tensor->Data();
        for (size_t i = 0; i < p.tensor->Size(); ++i)
            data[i] = 0.3f * static_cast<float>(std::sin(0.37 * ++k));
    }
}

Tensor<int64_t> Ids(size_t batch, size_t seq) {
    Tensor<int64_t> t({batch, seq});
    for (size_t b = 0; b < batch; ++b)
        for (size_t s = 0; s < seq; ++s) t(b, s) = static_cast<int64_t>((b * 3 + s * 2) % 11);
    return t;
}

}  // namespace

TEST(GPT2Construction, SmallPresetPassesValidation) {
    EXPECT_NO_THROW(models::gpt2::Small().Validate());
}

TEST(GPT2Construction, RejectsAnInvalidConfig) {
    TransformerConfig c = TinyConfig();
    c.vocab_size = 0;
    EXPECT_THROW(GPT2<float>{c}, std::invalid_argument);
}

TEST(GPT2Construction, BuildsFromTheConfig) {
    GPT2<float> model(TinyConfig());

    EXPECT_EQ(model.config.d_model, 8u);
    EXPECT_EQ(model.wte->weight->Shape(), (std::vector<size_t>{11, 8}));
    ASSERT_TRUE(model.wpe);
    EXPECT_EQ(model.wpe->weight->Shape(), (std::vector<size_t>{6, 8}));
    EXPECT_EQ(model.transformer->blocks.size(), 2u);
    EXPECT_FALSE(model.lm_head) << "GPT-2 ties its output to wte";
}

TEST(GPT2Construction, UntiedBuildsABiasFreeHead) {
    TransformerConfig c = TinyConfig();
    c.tie_word_embeddings = false;
    GPT2<float> model(c);

    ASSERT_TRUE(model.lm_head);
    EXPECT_EQ(model.lm_head->weight->Shape(), (std::vector<size_t>{11, 8}));
    EXPECT_FALSE(model.lm_head->bias);
}

TEST(GPT2Forward, GivesLogitsForEveryPosition) {
    GPT2<float> model(TinyConfig());
    Seed(model);

    EXPECT_EQ(model.Forward(Ids(2, 4)).Shape(), (std::vector<size_t>{2, 4, 11}));
}

TEST(GPT2Forward, RejectsIdsThatAreNot2D) {
    GPT2<float> model(TinyConfig());
    EXPECT_THROW(model.Forward(Tensor<int64_t>({4}, 1)), std::invalid_argument);
}

TEST(GPT2Forward, RejectsSequencesPastMaxSeqLen) {
    GPT2<float> model(TinyConfig());
    EXPECT_NO_THROW(model.Forward(Ids(1, 6)));
    EXPECT_THROW(model.Forward(Ids(1, 7)), std::invalid_argument);
    EXPECT_THROW(model.Forward(Ids(1, 2), 5), std::invalid_argument);
}

TEST(GPT2Forward, StartPosShiftsThePositions) {
    GPT2<float> model(TinyConfig());
    Seed(model);

    const Tensor<float> a = model.Forward(Ids(1, 2), 0);
    const Tensor<float> b = model.Forward(Ids(1, 2), 3);

    float largest = 0.0f;
    for (size_t i = 0; i < a.Size(); ++i) largest = std::max(largest, std::fabs(a.Data()[i] - b.Data()[i]));
    EXPECT_GT(largest, 1e-4f);
}

TEST(GPT2Parameters, SmallMatchesTheReleasedModelSize) {
    GPT2<float> model(models::gpt2::Small());

    size_t total = 0;
    for (const auto& p : model.NamedParameters()) total += p.tensor->Size();
    EXPECT_EQ(total, 124'439'808u);
}

// --- weight loading ---------------------------------------------------------

namespace {

using Weights = std::map<std::string, Tensor<float>>;

// Fills a tensor with base + 0.01, base + 0.02, ... so every value says which
// tensor it came from and where in it.
Tensor<float> Numbered(const std::vector<size_t>& shape, float base) {
    Tensor<float> t(shape);
    for (size_t i = 0; i < t.Size(); ++i) t.Data()[i] = base + 0.01f * static_cast<float>(i + 1);
    return t;
}

// Writes weights to a real .safetensors file (F32), removed on destruction.
class TempSafetensors {
public:
    explicit TempSafetensors(const std::map<std::string, Tensor<float>>& weights)
        : m_path(std::filesystem::temp_directory_path() /
                 ("gpt2_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".safetensors")) {
        nlohmann::json header = nlohmann::json::object();
        std::string data;
        for (const auto& [name, tensor] : weights) {
            const Tensor<float> flat = tensor.Contiguous();
            const size_t begin = data.size();
            data.append(reinterpret_cast<const char*>(flat.Data()), flat.Size() * sizeof(float));
            header[name] = {{"dtype", "F32"}, {"shape", flat.Shape()}, {"data_offsets", {begin, data.size()}}};
        }

        const std::string text = header.dump();
        const uint64_t size = text.size();
        std::ofstream out(m_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out << text << data;
    }
    ~TempSafetensors() { std::filesystem::remove(m_path); }

    std::string Path() const { return m_path.string(); }

private:
    std::filesystem::path m_path;
};

// A checkpoint matching TinyConfig(), in GPT-2's own layout: Conv1D weights are
// [in, out] and q/k/v share one c_attn matrix.
Weights TinyCheckpoint(const transformer::TransformerConfig& c = TinyConfig()) {
    Weights w;
    w.emplace("wte.weight", Numbered({c.vocab_size, c.d_model}, 1.0f));
    w.emplace("wpe.weight", Numbered({c.max_seq_len, c.d_model}, 2.0f));

    for (size_t i = 0; i < c.num_layers; ++i) {
        const std::string p = "h." + std::to_string(i) + ".";
        const float layer = 10.0f * static_cast<float>(i + 1);

        w.emplace(p + "ln_1.weight", Numbered({c.d_model}, layer + 1.0f));
        w.emplace(p + "ln_1.bias", Numbered({c.d_model}, layer + 2.0f));
        w.emplace(p + "attn.c_attn.weight", Numbered({c.d_model, 3 * c.d_model}, layer + 3.0f));
        w.emplace(p + "attn.c_attn.bias", Numbered({3 * c.d_model}, layer + 4.0f));
        w.emplace(p + "attn.c_proj.weight", Numbered({c.d_model, c.d_model}, layer + 5.0f));
        w.emplace(p + "attn.c_proj.bias", Numbered({c.d_model}, layer + 6.0f));
        w.emplace(p + "ln_2.weight", Numbered({c.d_model}, layer + 7.0f));
        w.emplace(p + "ln_2.bias", Numbered({c.d_model}, layer + 8.0f));
        w.emplace(p + "mlp.c_fc.weight", Numbered({c.d_model, c.ffn_dim}, layer + 9.0f));
        w.emplace(p + "mlp.c_fc.bias", Numbered({c.ffn_dim}, layer + 10.0f));
        w.emplace(p + "mlp.c_proj.weight", Numbered({c.ffn_dim, c.d_model}, layer + 11.0f));
        w.emplace(p + "mlp.c_proj.bias", Numbered({c.d_model}, layer + 12.0f));
    }

    w.emplace("ln_f.weight", Numbered({c.d_model}, 90.0f));
    w.emplace("ln_f.bias", Numbered({c.d_model}, 91.0f));
    return w;
}

}  // namespace

TEST(GPT2Load, FillsEveryParameter) {
    GPT2<float> model(TinyConfig());

    llm::LoadGPT2Weights(model, TinyCheckpoint());

    for (const auto& p : model.NamedParameters()) {
        bool all_zero = true;
        for (size_t i = 0; i < p.tensor->Size() && all_zero; ++i) all_zero = p.tensor->Data()[i] == 0.0f;
        EXPECT_FALSE(all_zero) << p.name << " was never loaded";
    }
}

TEST(GPT2Load, CopiesTheEmbeddingsAsTheyAre) {
    GPT2<float> model(TinyConfig());
    const Weights checkpoint = TinyCheckpoint();

    llm::LoadGPT2Weights(model, checkpoint);

    const Tensor<float>& wte = checkpoint.at("wte.weight");
    EXPECT_EQ(model.wte->weight->Shape(), wte.Shape());
    EXPECT_FLOAT_EQ((*model.wte->weight)(3, 2), wte(3, 2));
    EXPECT_FLOAT_EQ((*model.wpe->weight)(4, 1), checkpoint.at("wpe.weight")(4, 1));
}

TEST(GPT2Load, TransposesConv1DWeights) {
    // c_fc is [d, 4d] in the file and [4d, d] in the model.
    GPT2<float> model(TinyConfig());
    const Weights checkpoint = TinyCheckpoint();

    llm::LoadGPT2Weights(model, checkpoint);

    const Tensor<float>& c_fc = checkpoint.at("h.0.mlp.c_fc.weight");
    const auto& loaded = *model.transformer->blocks[0]->mlp->Get<0>()->weight;
    ASSERT_EQ(loaded.Shape(), (std::vector<size_t>{32, 8}));
    for (size_t in = 0; in < 8; ++in)
        for (size_t out = 0; out < 32; ++out)
            ASSERT_FLOAT_EQ(loaded(out, in), c_fc(in, out)) << "at (" << out << ", " << in << ")";
}

TEST(GPT2Load, SplitsFusedQKVInOrder) {
    GPT2<float> model(TinyConfig());
    const Weights checkpoint = TinyCheckpoint();

    llm::LoadGPT2Weights(model, checkpoint);

    const Tensor<float>& fused = checkpoint.at("h.1.attn.c_attn.weight");
    const Tensor<float>& fused_bias = checkpoint.at("h.1.attn.c_attn.bias");
    const auto& attention = *model.transformer->blocks[1]->attention;
    const std::array<std::shared_ptr<nn::Linear<float>>, 3> parts{
        attention.query, attention.key, attention.value};

    for (size_t part = 0; part < 3; ++part) {
        const size_t begin = part * 8;
        for (size_t in = 0; in < 8; ++in)
            for (size_t out = 0; out < 8; ++out)
                ASSERT_FLOAT_EQ((*parts[part]->weight)(out, in), fused(in, begin + out))
                    << "part " << part << " at (" << out << ", " << in << ")";
        for (size_t out = 0; out < 8; ++out)
            ASSERT_FLOAT_EQ((*parts[part]->bias)(out), fused_bias(begin + out)) << "part " << part;
    }
}

TEST(GPT2Load, RenamesLayerNormParameters) {
    GPT2<float> model(TinyConfig());
    const Weights checkpoint = TinyCheckpoint();

    llm::LoadGPT2Weights(model, checkpoint);

    EXPECT_FLOAT_EQ((*model.transformer->blocks[0]->ln_1->gamma)(0), checkpoint.at("h.0.ln_1.weight")(0));
    EXPECT_FLOAT_EQ((*model.transformer->blocks[0]->ln_1->beta)(1), checkpoint.at("h.0.ln_1.bias")(1));
    EXPECT_FLOAT_EQ((*model.transformer->ln_f->gamma)(2), checkpoint.at("ln_f.weight")(2));
    EXPECT_FLOAT_EQ((*model.transformer->ln_f->beta)(3), checkpoint.at("ln_f.bias")(3));
}

TEST(GPT2Load, LoadedParametersDoNotShareStorageWithTheCheckpoint) {
    GPT2<float> model(TinyConfig());
    Weights checkpoint = TinyCheckpoint();

    llm::LoadGPT2Weights(model, checkpoint);
    checkpoint.at("wte.weight").Fill(-1.0f);

    EXPECT_NE((*model.wte->weight)(0, 0), -1.0f);
}

TEST(GPT2Load, AcceptsTheTransformerPrefixAndSkipsMaskBuffers) {
    GPT2<float> model(TinyConfig());
    Weights prefixed;
    for (const auto& [name, tensor] : TinyCheckpoint()) prefixed.emplace("transformer." + name, tensor);
    prefixed.emplace("transformer.h.0.attn.bias", Numbered({6, 6}, 0.5f));
    prefixed.emplace("transformer.h.1.attn.masked_bias", Numbered({1}, 0.5f));

    EXPECT_NO_THROW(llm::LoadGPT2Weights(model, prefixed));
    EXPECT_FLOAT_EQ((*model.wte->weight)(0, 0), 1.01f);
}

TEST(GPT2Load, LoadsAnUntiedHead) {
    transformer::TransformerConfig c = TinyConfig();
    c.tie_word_embeddings = false;
    GPT2<float> model(c);
    Weights checkpoint = TinyCheckpoint(c);
    checkpoint.emplace("lm_head.weight", Numbered({c.vocab_size, c.d_model}, 5.0f));

    llm::LoadGPT2Weights(model, checkpoint);

    EXPECT_FLOAT_EQ((*model.lm_head->weight)(0, 0), 5.01f);
}

TEST(GPT2Load, ChangesTheOutput) {
    GPT2<float> model(TinyConfig());
    const Tensor<float> before = model.Forward(Ids(1, 3));

    llm::LoadGPT2Weights(model, TinyCheckpoint());
    const Tensor<float> after = model.Forward(Ids(1, 3));

    float largest = 0.0f;
    for (size_t i = 0; i < before.Size(); ++i)
        largest = std::max(largest, std::fabs(before.Data()[i] - after.Data()[i]));
    EXPECT_GT(largest, 1e-3f);
}

TEST(GPT2Load, RejectsAMissingTensor) {
    GPT2<float> model(TinyConfig());
    Weights checkpoint = TinyCheckpoint();
    checkpoint.erase("h.1.mlp.c_proj.bias");

    EXPECT_THROW(llm::LoadGPT2Weights(model, checkpoint), std::runtime_error);
}

TEST(GPT2Load, RejectsAWrongShape) {
    GPT2<float> model(TinyConfig());
    Weights checkpoint = TinyCheckpoint();
    checkpoint.at("h.0.mlp.c_fc.weight") = Numbered({8, 16}, 1.0f);   // should be [8, 32]

    EXPECT_THROW(llm::LoadGPT2Weights(model, checkpoint), std::runtime_error);
}

TEST(GPT2Load, RejectsAWrongSizedFusedQKV) {
    GPT2<float> model(TinyConfig());
    Weights checkpoint = TinyCheckpoint();
    checkpoint.at("h.0.attn.c_attn.weight") = Numbered({8, 16}, 1.0f);   // should be [8, 24]

    EXPECT_THROW(llm::LoadGPT2Weights(model, checkpoint), std::runtime_error);
}

TEST(GPT2Load, RejectsAnUnusedTensor) {
    GPT2<float> model(TinyConfig());
    Weights checkpoint = TinyCheckpoint();
    checkpoint.emplace("h.0.attn.c_attn.extra", Numbered({4}, 1.0f));

    EXPECT_THROW(llm::LoadGPT2Weights(model, checkpoint), std::runtime_error);
}

TEST(GPT2Load, RejectsACheckpointForADifferentLayerCount) {
    GPT2<float> model(TinyConfig());        // 2 layers
    transformer::TransformerConfig deeper = TinyConfig();
    deeper.num_layers = 3;

    EXPECT_THROW(llm::LoadGPT2Weights(model, TinyCheckpoint(deeper)), std::runtime_error);
}

TEST(GPT2Load, ReadsFromASafetensorsFile) {
    GPT2<float> model(TinyConfig());
    const TempSafetensors file(TinyCheckpoint());

    llm::LoadGPT2Weights(model, file.Path());

    EXPECT_FLOAT_EQ((*model.wte->weight)(0, 0), 1.01f);
    EXPECT_FLOAT_EQ((*model.transformer->ln_f->beta)(0), 91.01f);
}

// --- tokenizer --------------------------------------------------------------

namespace {

// Writes a vocab.json / merges.txt pair, removed on destruction.
class TempTokenizerFiles {
public:
    TempTokenizerFiles(const std::vector<std::string>& tokens, const std::vector<std::string>& merges) {
        const std::string stem = "tokenizer_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
        m_vocab = std::filesystem::temp_directory_path() / (stem + "_vocab.json");
        m_merges = std::filesystem::temp_directory_path() / (stem + "_merges.txt");

        nlohmann::json vocab = nlohmann::json::object();
        for (size_t i = 0; i < tokens.size(); ++i) vocab[tokens[i]] = i;
        std::ofstream(m_vocab) << vocab.dump();

        std::ofstream merge_file(m_merges);
        merge_file << "#version: 0.2\n";
        for (const std::string& merge : merges) merge_file << merge << "\n";
    }
    ~TempTokenizerFiles() {
        std::filesystem::remove(m_vocab);
        std::filesystem::remove(m_merges);
    }

    std::string Vocab() const { return m_vocab.string(); }
    std::string Merges() const { return m_merges.string(); }

private:
    std::filesystem::path m_vocab;
    std::filesystem::path m_merges;
};

// Every byte as a single token, so any text can be encoded, plus the merged
// pieces the merge rules below produce. "Ġ" is the stand-in for a space.
std::vector<std::string> ByteTokens() {
    std::vector<std::string> tokens;
    for (int b = 0; b < 256; ++b) {
        int codepoint = b;
        if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))) {
            static int next = 0;
            static std::map<int, int> assigned;
            if (!assigned.count(b)) assigned[b] = 256 + next++;
            codepoint = assigned[b];
        }
        std::string utf8;
        if (codepoint < 0x80) {
            utf8 += static_cast<char>(codepoint);
        } else {
            utf8 += static_cast<char>(0xC0 | (codepoint >> 6));
            utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
        tokens.push_back(utf8);
    }
    return tokens;
}

const std::string kSpace = "\xC4\xA0";   // U+0120 'Ġ'

// A tokenizer whose merges build "og" -> "dog" -> "Ġdog", like GPT-2 does.
TempTokenizerFiles DogFiles() {
    std::vector<std::string> tokens = ByteTokens();
    tokens.push_back("og");
    tokens.push_back("dog");
    tokens.push_back(kSpace + "dog");
    tokens.push_back("<|endoftext|>");

    return TempTokenizerFiles(tokens, {"o g", "d og", kSpace + " dog"});
}

GPT2Tokenizer Dog(const TempTokenizerFiles& files) {
    return GPT2Tokenizer(files.Vocab(), files.Merges());
}

}  // namespace

// --- loading ----------------------------------------------------------------

TEST(TokenizerLoad, ReadsVocabAndMerges) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    EXPECT_EQ(tokenizer.VocabSize(), 260u);   // 256 bytes + og + dog + Ġdog + <|endoftext|>
}

TEST(TokenizerLoad, RejectsMissingFiles) {
    const TempTokenizerFiles files = DogFiles();

    EXPECT_THROW(GPT2Tokenizer("no/such/vocab.json", files.Merges()), std::runtime_error);
    EXPECT_THROW(GPT2Tokenizer(files.Vocab(), "no/such/merges.txt"), std::runtime_error);
}

// --- encoding ---------------------------------------------------------------

TEST(TokenizerEncode, MergesToTheLongestLearnedToken) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    // "dog" merges o+g, then d+og; " dog" merges once more with the space.
    EXPECT_EQ(tokenizer.Encode("dog").size(), 1u);
    EXPECT_EQ(tokenizer.Encode(" dog").size(), 1u);
    EXPECT_NE(tokenizer.Encode("dog"), tokenizer.Encode(" dog")) << "a leading space is part of the token";
}

TEST(TokenizerEncode, FallsBackToBytesForUnknownText) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    EXPECT_EQ(tokenizer.Encode("cat").size(), 3u) << "no merge rules cover c, a, t";
}

TEST(TokenizerEncode, SplitsOnWordBoundaries) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    // "dog dog" -> "dog" + " dog"; merges never cross the boundary.
    const std::vector<int64_t> ids = tokenizer.Encode("dog dog");
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], tokenizer.Encode("dog")[0]);
    EXPECT_EQ(ids[1], tokenizer.Encode(" dog")[0]);
}

TEST(TokenizerEncode, KeepsLettersDigitsAndPunctuationApart) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    // "ab12!?" is three runs: letters, digits, punctuation -- 6 byte tokens here.
    EXPECT_EQ(tokenizer.Encode("ab12!?").size(), 6u);
    EXPECT_EQ(tokenizer.Decode(tokenizer.Encode("ab12!?")), "ab12!?");
}

TEST(TokenizerEncode, IsEmptyForEmptyInput) {
    const TempTokenizerFiles files = DogFiles();
    EXPECT_TRUE(Dog(files).Encode("").empty());
}

// --- decoding ---------------------------------------------------------------

TEST(TokenizerDecode, UndoesTheByteMapping) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    EXPECT_EQ(tokenizer.Decode(tokenizer.Encode(" dog")), " dog") << "not the stand-in character";
}

TEST(TokenizerDecode, RejectsAnIdOutsideTheVocabulary) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    EXPECT_THROW(tokenizer.Decode({999999}), std::out_of_range);
    EXPECT_THROW(tokenizer.Decode({-1}), std::out_of_range);
}

// --- round trips ------------------------------------------------------------

TEST(TokenizerRoundTrip, RecoversTextExactly) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    for (const std::string& text : {"dog", " dog", "dog dog dog", "The dog is here.",
                                    "a  b", "  leading", "trailing  ", "line\nbreak",
                                    "tabs\tand\nnewlines\n", "don't", "1234567890",
                                    "mixed123text!", "!@#$%^&*()"}) {
        EXPECT_EQ(tokenizer.Decode(tokenizer.Encode(text)), text) << "text: [" << text << "]";
    }
}

TEST(TokenizerRoundTrip, HandlesUtf8AndEmoji) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    for (const std::string& text : {"caf\xC3\xA9", "\xE4\xBD\xA0\xE5\xA5\xBD",
                                    "\xF0\x9F\x90\xB6", "emoji \xF0\x9F\x8E\x89 party"}) {
        EXPECT_EQ(tokenizer.Decode(tokenizer.Encode(text)), text);
    }
}

TEST(TokenizerRoundTrip, HandlesEveryByteValue) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    std::string all;
    for (int b = 1; b < 256; ++b) all += static_cast<char>(b);

    EXPECT_EQ(tokenizer.Decode(tokenizer.Encode(all)), all);
}

TEST(TokenizerRoundTrip, RepeatedTextUsesTheCacheAndStaysCorrect) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    const std::vector<int64_t> once = tokenizer.Encode("dog dog dog dog");
    const std::vector<int64_t> again = tokenizer.Encode("dog dog dog dog");

    EXPECT_EQ(once, again);
    EXPECT_EQ(tokenizer.Decode(once), "dog dog dog dog");
}

TEST(TokenizerEncode, KeepsContractionsAsTheirOwnWord) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    for (const std::string& suffix : {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}) {
        const std::string text = "dog" + suffix;
        const std::vector<int64_t> ids = tokenizer.Encode(text);
        ASSERT_FALSE(ids.empty());
        EXPECT_EQ(ids[0], tokenizer.Encode("dog")[0]) << text << " must start with the whole word";
        EXPECT_EQ(tokenizer.Decode(ids), text);
    }
}

TEST(TokenizerEncode, AttachesOnlyOneLeadingSpaceToAWord) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    // "  dog" is a lone space followed by " dog", not two spaces then "dog".
    const std::vector<int64_t> ids = tokenizer.Encode("  dog");
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[1], tokenizer.Encode(" dog")[0]);
    EXPECT_EQ(tokenizer.Decode(ids), "  dog");
}

TEST(TokenizerEncode, TrailingWhitespaceStaysTogether) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    EXPECT_EQ(tokenizer.Encode("dog   ").size(), 4u) << "one word token plus three spaces";
    EXPECT_EQ(tokenizer.Decode(tokenizer.Encode("dog   ")), "dog   ");
}

TEST(TokenizerEncode, DigitsFormTheirOwnRun) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    EXPECT_EQ(tokenizer.Encode("dog2024").size(), 5u) << "dog + 2 + 0 + 2 + 4";
}

TEST(TokenizerEncode, ConcatenationMatchesEncodingTheParts) {
    const TempTokenizerFiles files = DogFiles();
    const GPT2Tokenizer tokenizer = Dog(files);

    std::vector<int64_t> parts = tokenizer.Encode("The");
    for (const std::vector<int64_t>& piece : {tokenizer.Encode(" dog"), tokenizer.Encode(" runs")})
        parts.insert(parts.end(), piece.begin(), piece.end());

    EXPECT_EQ(tokenizer.Encode("The dog runs"), parts);
}

TEST(TokenizerEncode, AppliesMergesInRankOrder) {
    // "ab" and "bc" both merge, but "b c" ranks first, so "bc" wins and "a"
    // is left alone -- left-to-right merging would produce "ab" + "c".
    std::vector<std::string> tokens = ByteTokens();
    tokens.push_back("bc");
    tokens.push_back("ab");
    const TempTokenizerFiles files(tokens, {"b c", "a b"});
    const GPT2Tokenizer tokenizer(files.Vocab(), files.Merges());

    const std::vector<int64_t> ids = tokenizer.Encode("abc");
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], tokenizer.Encode("a")[0]);
    EXPECT_EQ(tokenizer.Decode(ids), "abc");
}

TEST(TokenizerEncode, MergesEveryOccurrenceOfAPair) {
    std::vector<std::string> tokens = ByteTokens();
    tokens.push_back("aa");
    const TempTokenizerFiles files(tokens, {"a a"});
    const GPT2Tokenizer tokenizer(files.Vocab(), files.Merges());

    EXPECT_EQ(tokenizer.Encode("aaaa").size(), 2u);
    EXPECT_EQ(tokenizer.Encode("aaa").size(), 2u) << "aa + a; the pair cannot overlap itself";
}

TEST(TokenizerLoad, RejectsAMalformedMergeLine) {
    const TempTokenizerFiles files(ByteTokens(), {"nospacehere"});
    EXPECT_THROW(GPT2Tokenizer(files.Vocab(), files.Merges()), std::runtime_error);
}

TEST(TokenizerLoad, RejectsAnEmptyMergesFile) {
    const TempTokenizerFiles files(ByteTokens(), {});
    EXPECT_THROW(GPT2Tokenizer(files.Vocab(), files.Merges()), std::runtime_error);
}

TEST(TokenizerLoad, RejectsAVocabThatIsNotJson) {
    const TempTokenizerFiles files = DogFiles();
    std::ofstream(files.Vocab()) << "{not json";

    EXPECT_THROW(GPT2Tokenizer(files.Vocab(), files.Merges()), std::runtime_error);
}

TEST(TokenizerEncode, ThrowsWhenAMergeProducesAnUnknownToken) {
    // The merge rule builds "aa", but the vocabulary has no such entry.
    const TempTokenizerFiles files(ByteTokens(), {"a a"});
    const GPT2Tokenizer tokenizer(files.Vocab(), files.Merges());

    EXPECT_THROW(tokenizer.Encode("aa"), std::runtime_error);
}
