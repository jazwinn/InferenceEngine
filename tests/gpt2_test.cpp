#include <gtest/gtest.h>

#include <Models/GPT2.h>
#include <Models/ModelConfigs.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using llm::GPT2;
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
