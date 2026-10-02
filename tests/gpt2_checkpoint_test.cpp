#include <gtest/gtest.h>

#include <Models/CausalModel.h>
#include <Models/ModelConfigs.h>
#include <Models/WeightLoader.h>
#include <Models/gpt2/Checkpoint.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using math::Tensor;
using models::CausalModel;
using models::LoadWeights;
using models::Weights;
using models::gpt2::ConvertCheckpoint;
using models::gpt2::ReadCheckpoint;
using nn::transformer::config::TransformerConfig;
using nn::transformer::norm::LayerNorm;

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

Tensor<float> Numbered(const std::vector<size_t>& shape, float base) {
    Tensor<float> t(shape);
    for (size_t i = 0; i < t.Size(); ++i) t.Data()[i] = base + 0.01f * static_cast<float>(i);
    return t;
}

void Seed(nn::Module<float>& m) {
    int k = 0;
    for (const auto& p : m.NamedParameters()) {
        float* data = p.tensor->Data();
        for (size_t i = 0; i < p.tensor->Size(); ++i) data[i] = 0.3f * static_cast<float>(std::sin(0.37 * ++k));
    }
}

void ExpectSameValues(const Tensor<float>& actual, const Tensor<float>& expected) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    const Tensor<float> a = actual.Clone();
    const Tensor<float> e = expected.Clone();
    for (size_t i = 0; i < a.Size(); ++i) EXPECT_FLOAT_EQ(a.Data()[i], e.Data()[i]) << "at " << i;
}

Tensor<float> Transposed(const Tensor<float>& t) { return t.Transpose().Clone(); }

const LayerNorm<float>& AsLayerNorm(const nn::transformer::norm::Norm<float>& norm) {
    return dynamic_cast<const LayerNorm<float>&>(norm);
}

Tensor<float> FuseQKV(const nn::transformer::GroupQueryAttention<float>& attention) {
    const size_t d = attention.query->weight->Shape()[0];
    Tensor<float> fused({d, 3 * d});
    const std::vector<const Tensor<float>*> parts{attention.query->weight.get(), attention.key->weight.get(),
                                                  attention.value->weight.get()};
    for (size_t p = 0; p < parts.size(); ++p)
        for (size_t out = 0; out < d; ++out)
            for (size_t in = 0; in < d; ++in) fused(in, p * d + out) = (*parts[p])(out, in);
    return fused;
}

Tensor<float> FuseQKVBias(const nn::transformer::GroupQueryAttention<float>& attention) {
    const size_t d = attention.query->bias->Shape()[0];
    Tensor<float> fused({3 * d});
    const std::vector<const Tensor<float>*> parts{attention.query->bias.get(), attention.key->bias.get(),
                                                  attention.value->bias.get()};
    for (size_t p = 0; p < parts.size(); ++p)
        for (size_t i = 0; i < d; ++i) fused(p * d + i) = (*parts[p])(i);
    return fused;
}

Weights<float> GPT2Format(const CausalModel<float>& model) {
    Weights<float> w;
    w.emplace("wte.weight", model.wte->weight->Clone());
    w.emplace("wpe.weight", model.wpe->weight->Clone());
    w.emplace("ln_f.weight", AsLayerNorm(*model.transformer->finalNorm).gamma->Clone());
    w.emplace("ln_f.bias", AsLayerNorm(*model.transformer->finalNorm).beta->Clone());

    const size_t seq = model.config.max_seq_len;
    for (size_t i = 0; i < model.transformer->blocks.size(); ++i) {
        const auto& block = *model.transformer->blocks[i];
        const std::string h = "h." + std::to_string(i) + ".";
        w.emplace(h + "ln_1.weight", AsLayerNorm(*block.attenNorm).gamma->Clone());
        w.emplace(h + "ln_1.bias", AsLayerNorm(*block.attenNorm).beta->Clone());
        w.emplace(h + "ln_2.weight", AsLayerNorm(*block.mlpNorm).gamma->Clone());
        w.emplace(h + "ln_2.bias", AsLayerNorm(*block.mlpNorm).beta->Clone());
        w.emplace(h + "attn.c_attn.weight", FuseQKV(*block.attention));
        w.emplace(h + "attn.c_attn.bias", FuseQKVBias(*block.attention));
        w.emplace(h + "attn.c_proj.weight", Transposed(*block.attention->output_proj->weight));
        w.emplace(h + "attn.c_proj.bias", block.attention->output_proj->bias->Clone());
        w.emplace(h + "attn.bias", Tensor<float>({1, 1, seq, seq}, 1.0f));
        w.emplace(h + "mlp.c_fc.weight", Transposed(*block.mlp->up->weight));
        w.emplace(h + "mlp.c_fc.bias", block.mlp->up->bias->Clone());
        w.emplace(h + "mlp.c_proj.weight", Transposed(*block.mlp->down->weight));
        w.emplace(h + "mlp.c_proj.bias", block.mlp->down->bias->Clone());
    }
    return w;
}

Weights<float> OneBlock(size_t d = 4) {
    Weights<float> w;
    w.emplace("h.0.attn.c_attn.weight", Numbered({d, 3 * d}, 1.0f));
    w.emplace("h.0.attn.c_attn.bias", Numbered({3 * d}, 2.0f));
    return w;
}

Tensor<int64_t> Ids(size_t seq) {
    Tensor<int64_t> t({1, seq});
    for (size_t s = 0; s < seq; ++s) t(0, s) = static_cast<int64_t>((s * 2 + 1) % 11);
    return t;
}

class TempFile {
public:
    explicit TempFile(const std::string& contents)
        : m_path(std::filesystem::temp_directory_path() /
                 ("gpt2_checkpoint_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".safetensors")) {
        std::ofstream(m_path, std::ios::binary) << contents;
    }
    ~TempFile() { std::filesystem::remove(m_path); }

    std::string Path() const { return m_path.string(); }

private:
    std::filesystem::path m_path;
};

std::string SafetensorsF32(const Weights<float>& tensors) {
    nlohmann::json header = nlohmann::json::object();
    std::string data;
    for (const auto& [name, tensor] : tensors) {
        const size_t begin = data.size();
        data.append(reinterpret_cast<const char*>(tensor.Data()), tensor.Size() * sizeof(float));
        header[name] = {{"dtype", "F32"}, {"shape", tensor.Shape()}, {"data_offsets", {begin, data.size()}}};
    }
    const std::string text = header.dump();
    const uint64_t size = text.size();
    return std::string(reinterpret_cast<const char*>(&size), sizeof(size)) + text + data;
}

}  // namespace

TEST(GPT2Checkpoint, SplitsFusedQKVWeightsInOrder) {
    const Weights<float> checkpoint = OneBlock();
    const Tensor<float> fused = checkpoint.at("h.0.attn.c_attn.weight");

    const Weights<float> converted = ConvertCheckpoint(checkpoint);

    const std::vector<std::string> parts{"query", "key", "value"};
    for (size_t p = 0; p < parts.size(); ++p) {
        SCOPED_TRACE(parts[p]);
        const Tensor<float>& w = converted.at("transformer.h.0.attention." + parts[p] + ".weight");
        ASSERT_EQ(w.Shape(), (std::vector<size_t>{4, 4}));
        for (size_t out = 0; out < 4; ++out)
            for (size_t in = 0; in < 4; ++in) EXPECT_FLOAT_EQ(w(out, in), fused(in, p * 4 + out));
    }
}

TEST(GPT2Checkpoint, SplitsFusedQKVBiasesInOrder) {
    const Weights<float> converted = ConvertCheckpoint(OneBlock());

    EXPECT_FLOAT_EQ(converted.at("transformer.h.0.attention.query.bias")(0), 2.00f);
    EXPECT_FLOAT_EQ(converted.at("transformer.h.0.attention.key.bias")(0), 2.04f);
    EXPECT_FLOAT_EQ(converted.at("transformer.h.0.attention.value.bias")(3), 2.11f);
}

TEST(GPT2Checkpoint, TransposesConv1DWeights) {
    Weights<float> checkpoint;
    checkpoint.emplace("h.3.mlp.c_fc.weight", Numbered({2, 5}, 1.0f));
    checkpoint.emplace("h.3.mlp.c_proj.weight", Numbered({5, 2}, 2.0f));
    checkpoint.emplace("h.3.attn.c_proj.weight", Numbered({2, 2}, 3.0f));

    const Weights<float> converted = ConvertCheckpoint(checkpoint);

    ExpectSameValues(converted.at("transformer.h.3.mlp.up.weight"), Transposed(checkpoint.at("h.3.mlp.c_fc.weight")));
    ExpectSameValues(converted.at("transformer.h.3.mlp.down.weight"), Transposed(checkpoint.at("h.3.mlp.c_proj.weight")));
    ExpectSameValues(converted.at("transformer.h.3.attention.output_proj.weight"),
                     Transposed(checkpoint.at("h.3.attn.c_proj.weight")));
}

TEST(GPT2Checkpoint, ProducesContiguousTensors) {
    Weights<float> checkpoint = OneBlock();
    checkpoint.emplace("h.0.mlp.c_fc.weight", Numbered({2, 5}, 1.0f));

    for (const auto& [name, tensor] : ConvertCheckpoint(checkpoint)) {
        SCOPED_TRACE(name);
        EXPECT_TRUE(tensor.IsContiguous());
    }
}

TEST(GPT2Checkpoint, LeavesBiasesAndEmbeddingsAsTheyAre) {
    Weights<float> checkpoint;
    checkpoint.emplace("wte.weight", Numbered({3, 2}, 1.0f));
    checkpoint.emplace("wpe.weight", Numbered({4, 2}, 2.0f));
    checkpoint.emplace("h.0.mlp.c_fc.bias", Numbered({5}, 3.0f));

    const Weights<float> converted = ConvertCheckpoint(checkpoint);

    ExpectSameValues(converted.at("wte.weight"), checkpoint.at("wte.weight"));
    ExpectSameValues(converted.at("wpe.weight"), checkpoint.at("wpe.weight"));
    ExpectSameValues(converted.at("transformer.h.0.mlp.up.bias"), checkpoint.at("h.0.mlp.c_fc.bias"));
}

TEST(GPT2Checkpoint, RenamesLayerNormsToGammaAndBeta) {
    Weights<float> checkpoint;
    for (const std::string name : {"ln_f", "h.0.ln_1", "h.0.ln_2"}) {
        checkpoint.emplace(name + ".weight", Tensor<float>({2}));
        checkpoint.emplace(name + ".bias", Tensor<float>({2}));
    }

    const Weights<float> converted = ConvertCheckpoint(checkpoint);

    for (const std::string name : {"transformer.finalNorm", "transformer.h.0.attenNorm", "transformer.h.0.mlpNorm"}) {
        EXPECT_TRUE(converted.contains(name + ".gamma")) << name;
        EXPECT_TRUE(converted.contains(name + ".beta")) << name;
    }
    EXPECT_EQ(converted.size(), 6u);
}

TEST(GPT2Checkpoint, DropsMaskBuffers) {
    Weights<float> checkpoint;
    checkpoint.emplace("h.0.attn.bias", Tensor<float>({1, 1, 2, 2}));
    checkpoint.emplace("h.0.attn.masked_bias", Tensor<float>({1}));

    EXPECT_TRUE(ConvertCheckpoint(checkpoint).empty());
}

TEST(GPT2Checkpoint, StripsTheTransformerPrefix) {
    Weights<float> checkpoint;
    checkpoint.emplace("transformer.wte.weight", Tensor<float>({3, 2}));
    checkpoint.emplace("transformer.h.0.ln_1.weight", Tensor<float>({2}));

    const Weights<float> converted = ConvertCheckpoint(checkpoint);

    EXPECT_TRUE(converted.contains("wte.weight"));
    EXPECT_TRUE(converted.contains("transformer.h.0.attenNorm.gamma"));
}

TEST(GPT2Checkpoint, RejectsATensorThatAppearsTwice) {
    Weights<float> checkpoint;
    checkpoint.emplace("wte.weight", Tensor<float>({3, 2}));
    checkpoint.emplace("transformer.wte.weight", Tensor<float>({3, 2}));

    EXPECT_THROW(ConvertCheckpoint(checkpoint), std::runtime_error);
}

TEST(GPT2Checkpoint, RejectsAFusedQKVThatCannotBeSplit) {
    Weights<float> notThreeParts;
    notThreeParts.emplace("h.0.attn.c_attn.weight", Tensor<float>({4, 10}));
    Weights<float> notSquareParts;
    notSquareParts.emplace("h.0.attn.c_attn.weight", Tensor<float>({4, 6}));
    Weights<float> bias;
    bias.emplace("h.0.attn.c_attn.bias", Tensor<float>({10}));

    EXPECT_THROW(ConvertCheckpoint(notThreeParts), std::runtime_error);
    EXPECT_THROW(ConvertCheckpoint(notSquareParts), std::runtime_error);
    EXPECT_THROW(ConvertCheckpoint(bias), std::runtime_error);
}

TEST(GPT2Checkpoint, RejectsAConv1DWeightThatIsNotAMatrix) {
    Weights<float> checkpoint;
    checkpoint.emplace("h.0.mlp.c_fc.weight", Tensor<float>({2, 3, 4}));

    EXPECT_THROW(ConvertCheckpoint(checkpoint), std::runtime_error);
}

TEST(GPT2Checkpoint, RoundTripsIntoTheModel) {
    CausalModel<float> original(TinyConfig());
    Seed(original);
    CausalModel<float> loaded(TinyConfig());

    LoadWeights(loaded, ConvertCheckpoint(GPT2Format(original)));

    const auto expected = original.NamedParameters();
    const auto actual = loaded.NamedParameters();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        SCOPED_TRACE(actual[i].name);
        ExpectSameValues(*actual[i].tensor, *expected[i].tensor);
    }
    ExpectSameValues(loaded.Forward(Ids(5)), original.Forward(Ids(5)));
}

TEST(GPT2Checkpoint, ReadsAndConvertsASafetensorsFile) {
    Weights<float> checkpoint = OneBlock();
    checkpoint.emplace("transformer.h.0.mlp.c_fc.weight", Numbered({2, 5}, 1.0f));
    checkpoint.emplace("h.0.attn.bias", Tensor<float>({1, 1, 2, 2}));
    const TempFile file(SafetensorsF32(checkpoint));

    const Weights<float> converted = ReadCheckpoint<float>(file.Path());

    EXPECT_EQ(converted.size(), 7u);
    ExpectSameValues(converted.at("transformer.h.0.mlp.up.weight"),
                     Transposed(checkpoint.at("transformer.h.0.mlp.c_fc.weight")));
}

TEST(GPT2Checkpoint, ReleasedSmallModelMatchesHuggingFace) {
    const std::string path = std::string(IE_SOURCE_DIR) + "/Models/gpt2/model.safetensors";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " is not downloaded.";

    CausalModel<float> model(models::gpt2::Small());
    LoadWeights(model, ReadCheckpoint<float>(path));

    const std::vector<int64_t> ids{15496, 11, 616, 1438, 318};
    Tensor<int64_t> prompt({1, ids.size()});
    for (size_t i = 0; i < ids.size(); ++i) prompt(0, i) = ids[i];
    const Tensor<float> logits = model.Forward(prompt);

    const std::vector<std::pair<size_t, float>> reference{
        {0, -68.42887f}, {13, -67.08368f}, {262, -65.94913f}, {1757, -64.44328f}, {50256, -70.00941f}};
    for (const auto& [token, expected] : reference) EXPECT_NEAR(logits(0, ids.size() - 1, token), expected, 1e-3f) << token;
}
