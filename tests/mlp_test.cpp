#include <gtest/gtest.h>

#include <NeuralNet/Transformer/MLP.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using math::Tensor;
using transformer::Activation;
using transformer::MLP;
using transformer::MLPType;
using transformer::TransformerConfig;

namespace {

constexpr float kTol = 1e-5f;
constexpr size_t kModel = 6;
constexpr size_t kFfn = 10;

TransformerConfig Config(MLPType type, Activation act, bool bias = true, size_t d_model = kModel, size_t ffn = kFfn) {
    TransformerConfig cfg;
    cfg.d_model = d_model;
    cfg.ffn_dim = ffn;
    cfg.mlp_type = type;
    cfg.activation = act;
    cfg.output_bias = bias;
    return cfg;
}

void WaveFill(Tensor<float>& t, double phase) {
    float* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i)
        data[i] = static_cast<float>(0.5 * std::sin(phase + 0.37 * static_cast<double>(i)));
}

void Seed(MLP<float>& mlp) {
    double phase = 0.2;
    for (const auto& linear : {mlp.gate, mlp.up, mlp.down}) {
        if (!linear) continue;
        WaveFill(*linear->weight, phase);
        if (linear->bias) WaveFill(*linear->bias, phase + 0.1);
        phase += 0.6;
    }
}

Tensor<float> Input() {
    Tensor<float> x({2, 3, kModel});
    WaveFill(x, 1.7);
    return x;
}

Tensor<float> Apply(Activation act, const Tensor<float>& x) {
    if (act == Activation::GELUTanh) return activation::GELU<float>().Forward(x);
    if (act == Activation::SiLU) return activation::SiLU<float>().Forward(x);
    return activation::ReLU<float>().Forward(x);
}

void ExpectNear(const Tensor<float>& actual, const Tensor<float>& expected) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    Tensor<float> a = actual.Contiguous();
    Tensor<float> e = expected.Contiguous();
    for (size_t i = 0; i < a.Size(); ++i) EXPECT_NEAR(a.Data()[i], e.Data()[i], kTol) << "at flat index " << i;
}

}  // namespace

TEST(MLPConstruction, StandardHasUpAndDownOnly) {
    MLP<float> mlp(Config(MLPType::Standard, Activation::GELUTanh));

    EXPECT_FALSE(mlp.gate);
    EXPECT_EQ(mlp.up->weight->Shape(), (std::vector<size_t>{kFfn, kModel}));
    EXPECT_EQ(mlp.down->weight->Shape(), (std::vector<size_t>{kModel, kFfn}));
    EXPECT_EQ(mlp.NamedParameters().size(), 4u);
}

TEST(MLPConstruction, GatedAddsGateProjection) {
    MLP<float> mlp(Config(MLPType::Gated, Activation::SiLU));

    ASSERT_TRUE(mlp.gate);
    EXPECT_EQ(mlp.gate->weight->Shape(), (std::vector<size_t>{kFfn, kModel}));
    EXPECT_EQ(mlp.up->weight->Shape(), (std::vector<size_t>{kFfn, kModel}));
    EXPECT_EQ(mlp.down->weight->Shape(), (std::vector<size_t>{kModel, kFfn}));
    EXPECT_EQ(mlp.NamedParameters().size(), 6u);
}

TEST(MLPConstruction, BiasFlagControlsEveryProjection) {
    MLP<float> mlp(Config(MLPType::Gated, Activation::SiLU, false));

    EXPECT_FALSE(mlp.gate->bias);
    EXPECT_FALSE(mlp.up->bias);
    EXPECT_FALSE(mlp.down->bias);
    EXPECT_EQ(mlp.NamedParameters().size(), 3u);
}

TEST(MLPConstruction, RegistersSubmodulesUnderStableNames) {
    MLP<float> mlp(Config(MLPType::Gated, Activation::SiLU));

    std::vector<std::string> names;
    for (const auto& p : mlp.NamedParameters()) names.push_back(p.name);

    EXPECT_EQ(names, (std::vector<std::string>{"gate.weight", "gate.bias", "up.weight", "up.bias", "down.weight", "down.bias"}));
}

TEST(MLPConstruction, RejectsExactGELU) {
    EXPECT_THROW(MLP<float>(Config(MLPType::Standard, Activation::GELU)), std::invalid_argument);
}

TEST(MLPConstruction, RejectsZeroSizes) {
    EXPECT_THROW(MLP<float>(Config(MLPType::Standard, Activation::ReLU, true, 0, kFfn)), std::invalid_argument);
    EXPECT_THROW(MLP<float>(Config(MLPType::Standard, Activation::ReLU, true, kModel, 0)), std::invalid_argument);
}

TEST(MLPForward, KeepsInputShape) {
    MLP<float> mlp(Config(MLPType::Gated, Activation::SiLU));
    Seed(mlp);

    EXPECT_EQ(mlp.Forward(Input()).Shape(), (std::vector<size_t>{2, 3, kModel}));
}

class MLPActivations : public ::testing::TestWithParam<Activation> {};

TEST_P(MLPActivations, StandardComputesDownOfActivatedUp) {
    MLP<float> mlp(Config(MLPType::Standard, GetParam()));
    Seed(mlp);
    const Tensor<float> x = Input();

    const Tensor<float> expected = mlp.down->Forward(Apply(GetParam(), mlp.up->Forward(x)));

    ExpectNear(mlp.Forward(x), expected);
}

TEST_P(MLPActivations, GatedComputesDownOfActivatedGateTimesUp) {
    MLP<float> mlp(Config(MLPType::Gated, GetParam()));
    Seed(mlp);
    const Tensor<float> x = Input();

    const Tensor<float> expected = mlp.down->Forward(Apply(GetParam(), mlp.gate->Forward(x)) * mlp.up->Forward(x));

    ExpectNear(mlp.Forward(x), expected);
}

INSTANTIATE_TEST_SUITE_P(All, MLPActivations,
                         ::testing::Values(Activation::GELUTanh, Activation::SiLU, Activation::ReLU),
                         [](const ::testing::TestParamInfo<Activation>& info) {
                             switch (info.param) {
                                 case Activation::GELUTanh: return std::string("GELUTanh");
                                 case Activation::SiLU: return std::string("SiLU");
                                 default: return std::string("ReLU");
                             }
                         });

TEST(MLPForward, GatedSiLUMatchesHandComputedValue) {
    MLP<float> mlp(Config(MLPType::Gated, Activation::SiLU, false, 1, 1));
    (*mlp.gate->weight)(0, 0) = 1.0f;
    (*mlp.up->weight)(0, 0) = 3.0f;
    (*mlp.down->weight)(0, 0) = 2.0f;
    Tensor<float> x({1, 1, 1}, 1.0f);

    const float silu_one = 1.0f / (1.0f + std::exp(-1.0f));

    EXPECT_NEAR(mlp.Forward(x)(0, 0, 0), silu_one * 3.0f * 2.0f, kTol);
}

TEST(MLPForward, StandardReLUZeroesNegativeHiddenUnits) {
    MLP<float> mlp(Config(MLPType::Standard, Activation::ReLU, true, 1, 2));
    (*mlp.up->weight)(0, 0) = 1.0f;
    (*mlp.up->weight)(1, 0) = -1.0f;
    (*mlp.down->weight)(0, 0) = 1.0f;
    (*mlp.down->weight)(0, 1) = 10.0f;
    (*mlp.down->bias)(0) = 0.5f;
    Tensor<float> x({1, 1, 1}, 2.0f);

    EXPECT_NEAR(mlp.Forward(x)(0, 0, 0), 2.0f + 0.5f, kTol);
}

TEST(MLPForward, GateChangesTheResult) {
    MLP<float> standard(Config(MLPType::Standard, Activation::SiLU));
    MLP<float> gated(Config(MLPType::Gated, Activation::SiLU));
    Seed(standard);
    Seed(gated);
    *gated.up->weight = standard.up->weight->Clone();
    *gated.up->bias = standard.up->bias->Clone();
    *gated.down->weight = standard.down->weight->Clone();
    *gated.down->bias = standard.down->bias->Clone();
    const Tensor<float> x = Input();

    const Tensor<float> a = standard.Forward(x).Contiguous();
    const Tensor<float> b = gated.Forward(x).Contiguous();
    float diff = 0.0f;
    for (size_t i = 0; i < a.Size(); ++i) diff = std::fmax(diff, std::fabs(a.Data()[i] - b.Data()[i]));

    EXPECT_GT(diff, 1e-3f);
}
