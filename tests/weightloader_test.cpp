#include <gtest/gtest.h>

#include <Models/CausalModel.h>
#include <Models/ModelConfigs.h>
#include <Models/WeightLoader.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using tensor::Tensor;
using models::LoadWeights;
using models::Rename;
using models::RenameRule;
using models::Weights;

namespace {

Tensor<float> Numbered(const std::vector<size_t>& shape, float base) {
    Tensor<float> t(shape);
    for (size_t i = 0; i < t.Size(); ++i) t.Data()[i] = base + 0.01f * static_cast<float>(i);
    return t;
}

void ExpectSameValues(const Tensor<float>& actual, const Tensor<float>& expected) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    for (size_t i = 0; i < actual.Size(); ++i) EXPECT_FLOAT_EQ(actual.Data()[i], expected.Data()[i]) << "at " << i;
}

Weights<float> LinearCheckpoint() {
    Weights<float> w;
    w.emplace("weight", Numbered({3, 2}, 1.0f));
    w.emplace("bias", Numbered({3}, 2.0f));
    return w;
}

std::string ErrorOf(nn::Linear<float>& model, Weights<float> weights) {
    try {
        LoadWeights(model, std::move(weights));
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return "";
}

std::vector<std::string> Names(const Weights<float>& weights) {
    std::vector<std::string> names;
    for (const auto& [name, tensor] : weights) names.push_back(name);
    return names;
}

Weights<float> Single(const std::string& name) {
    Weights<float> w;
    w.emplace(name, Tensor<float>({1}));
    return w;
}

class TempFile {
public:
    explicit TempFile(const std::string& contents)
        : m_path(std::filesystem::temp_directory_path() /
                 ("weightloader_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".safetensors")) {
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

TEST(WeightLoaderLoad, FillsEveryParameterByName) {
    nn::Linear<float> linear(2, 3);
    const Weights<float> checkpoint = LinearCheckpoint();

    LoadWeights(linear, checkpoint);

    ExpectSameValues(*linear.weight, checkpoint.at("weight"));
    ExpectSameValues(*linear.bias, checkpoint.at("bias"));
}

TEST(WeightLoaderLoad, FollowsNestedModuleNames) {
    nn::transformer::config::TransformerConfig config = models::gpt2::Small();
    config.vocab_size = 11;
    config.d_model = 8;
    config.num_layers = 2;
    config.num_heads = 2;
    config.ffn_dim = 32;
    config.max_seq_len = 6;
    models::CausalModel<float> model(config);

    Weights<float> checkpoint;
    float base = 0.0f;
    for (const auto& [name, tensor] : model.NamedParameters()) checkpoint.emplace(name, Numbered(tensor->Shape(), base += 1.0f));

    LoadWeights(model, checkpoint);

    for (const auto& [name, tensor] : model.NamedParameters()) {
        SCOPED_TRACE(name);
        ExpectSameValues(*tensor, checkpoint.at(name));
    }
}

TEST(WeightLoaderLoad, ReportsMissingTensors) {
    nn::Linear<float> linear(2, 3);
    Weights<float> checkpoint = LinearCheckpoint();
    checkpoint.erase("bias");

    EXPECT_EQ(ErrorOf(linear, checkpoint), "WeightLoader: checkpoint is missing 'bias'.");
}

TEST(WeightLoaderLoad, ReportsUnusedTensors) {
    nn::Linear<float> linear(2, 3);
    Weights<float> checkpoint = LinearCheckpoint();
    checkpoint.emplace("extra", Tensor<float>({1}));

    EXPECT_EQ(ErrorOf(linear, checkpoint), "WeightLoader: checkpoint has unused 'extra'.");
}

TEST(WeightLoaderLoad, ReportsMissingAndUnusedTogether) {
    nn::Linear<float> linear(2, 3);
    Weights<float> checkpoint = LinearCheckpoint();
    checkpoint.erase("bias");
    checkpoint.emplace("bias_typo", Numbered({3}, 2.0f));

    EXPECT_EQ(ErrorOf(linear, checkpoint),
              "WeightLoader: checkpoint is missing 'bias'. checkpoint has unused 'bias_typo'.");
}

TEST(WeightLoaderLoad, ShortensLongListsOfNames) {
    nn::Linear<float> linear(2, 3);
    Weights<float> checkpoint = LinearCheckpoint();
    for (int i = 0; i < 7; ++i) checkpoint.emplace("extra" + std::to_string(i), Tensor<float>({1}));

    const std::string error = ErrorOf(linear, checkpoint);

    EXPECT_NE(error.find("'extra4'"), std::string::npos);
    EXPECT_EQ(error.find("'extra5'"), std::string::npos);
    EXPECT_NE(error.find("and 2 more"), std::string::npos);
}

TEST(WeightLoaderLoad, ReportsAWrongShape) {
    nn::Linear<float> linear(2, 3);
    Weights<float> checkpoint = LinearCheckpoint();
    checkpoint.at("weight") = Numbered({2, 3}, 1.0f);

    EXPECT_EQ(ErrorOf(linear, checkpoint), "WeightLoader: 'weight' is [2, 3] but the model expects [3, 2].");
}

TEST(WeightLoaderLoad, AFailedLoadLeavesTheModelUntouched) {
    nn::Linear<float> linear(2, 3);
    const Tensor<float> before = linear.weight->Clone();
    Weights<float> checkpoint = LinearCheckpoint();
    checkpoint.emplace("extra", Tensor<float>({1}));

    EXPECT_THROW(LoadWeights(linear, checkpoint), std::runtime_error);

    ExpectSameValues(*linear.weight, before);
}

TEST(WeightLoaderLoad, AppliesRenameRulesFirst) {
    nn::Linear<float> linear(2, 3);
    Weights<float> checkpoint;
    checkpoint.emplace("proj.w", Numbered({3, 2}, 1.0f));
    checkpoint.emplace("proj.b", Numbered({3}, 2.0f));

    LoadWeights(linear, checkpoint, {{"proj.w", "weight"}, {"proj.b", "bias"}});

    ExpectSameValues(*linear.weight, checkpoint.at("proj.w"));
    ExpectSameValues(*linear.bias, checkpoint.at("proj.b"));
}

TEST(WeightLoaderLoad, ReadsASafetensorsFile) {
    nn::Linear<float> linear(2, 3);
    Weights<float> checkpoint;
    checkpoint.emplace("layer.weight", Numbered({3, 2}, 1.0f));
    checkpoint.emplace("layer.bias", Numbered({3}, 2.0f));
    const TempFile file(SafetensorsF32(checkpoint));

    LoadWeights(linear, file.Path(), {{"layer.weight", "weight"}, {"layer.bias", "bias"}});

    ExpectSameValues(*linear.weight, checkpoint.at("layer.weight"));
    ExpectSameValues(*linear.bias, checkpoint.at("layer.bias"));
}

TEST(WeightLoaderLoad, MissingFileThrows) {
    nn::Linear<float> linear(2, 3);

    EXPECT_THROW(LoadWeights(linear, std::string("does/not/exist.safetensors")), std::runtime_error);
}

TEST(WeightLoaderRename, SubstitutesTheLayerIndex) {
    Weights<float> checkpoint = Single("model.layers.12.mlp.up_proj.weight");

    const Weights<float> renamed =
        Rename(checkpoint, {{"model.layers.{i}.mlp.up_proj.weight", "transformer.h.{i}.mlp.up.weight"}});

    EXPECT_EQ(Names(renamed), (std::vector<std::string>{"transformer.h.12.mlp.up.weight"}));
}

TEST(WeightLoaderRename, KeepsTheTensorValues) {
    Weights<float> checkpoint;
    checkpoint.emplace("a", Numbered({2, 2}, 5.0f));

    const Weights<float> renamed = Rename(checkpoint, {{"a", "b"}});

    ExpectSameValues(renamed.at("b"), checkpoint.at("a"));
}

TEST(WeightLoaderRename, PassesUnmatchedNamesThrough) {
    const Weights<float> renamed = Rename(Single("wte.weight"), {{"h.{i}.x", "y.{i}"}});

    EXPECT_EQ(Names(renamed), (std::vector<std::string>{"wte.weight"}));
}

TEST(WeightLoaderRename, TheFirstMatchingRuleWins) {
    const Weights<float> renamed = Rename(Single("h.0.x"), {{"h.{i}.x", "first"}, {"h.0.x", "second"}});

    EXPECT_EQ(Names(renamed), (std::vector<std::string>{"first"}));
}

TEST(WeightLoaderRename, MatchesTheWholeNameOnly) {
    const Weights<float> renamed = Rename(Single("h.1.xy"), {{"h.{i}.x", "y"}});

    EXPECT_EQ(Names(renamed), (std::vector<std::string>{"h.1.xy"}));
}

TEST(WeightLoaderRename, TreatsRegexCharactersLiterally) {
    EXPECT_EQ(Names(Rename(Single("aXb"), {{"a.b", "c"}})), (std::vector<std::string>{"aXb"}));
    EXPECT_EQ(Names(Rename(Single("a+(b)"), {{"a+(b)", "c"}})), (std::vector<std::string>{"c"}));
}

TEST(WeightLoaderRename, IndexMatchesDigitsOnly) {
    const Weights<float> renamed = Rename(Single("h.first.x"), {{"h.{i}.x", "y"}});

    EXPECT_EQ(Names(renamed), (std::vector<std::string>{"h.first.x"}));
}

TEST(WeightLoaderRename, RejectsTwoTensorsWithTheSameNewName) {
    Weights<float> checkpoint = Single("a.1");
    checkpoint.emplace("a.2", Tensor<float>({1}));

    EXPECT_THROW(Rename(checkpoint, {{"a.{i}", "b"}}), std::runtime_error);
}

TEST(WeightLoaderRename, RejectsMalformedRules) {
    EXPECT_THROW(Rename(Weights<float>{}, std::vector<RenameRule>{{"a.{i}.{i}", "b.{i}"}}), std::runtime_error);
    EXPECT_THROW(Rename(Weights<float>{}, std::vector<RenameRule>{{"a", "b.{i}"}}), std::runtime_error);
}
