#include <gtest/gtest.h>

#include <NeuralNet/Transformer/Normal/MakeNorm.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using math::Tensor;
using nn::transformer::config::NormType;
using nn::transformer::config::TransformerConfig;
using nn::transformer::norm::LayerNorm;
using nn::transformer::norm::MakeNorm;
using nn::transformer::norm::Norm;
using nn::transformer::norm::RMSNorm;

namespace {

TransformerConfig Config(NormType type, float eps = 1e-5f) {
    TransformerConfig cfg;
    cfg.d_model = 6;
    cfg.norm_type = type;
    cfg.norm_eps = eps;
    return cfg;
}

Tensor<float> Input() {
    Tensor<float> x({2, 3, 6});
    float* data = x.Data();
    for (size_t i = 0; i < x.Size(); ++i) data[i] = static_cast<float>(std::sin(0.7 + 0.43 * static_cast<double>(i)));
    return x;
}

void ExpectNear(const Tensor<float>& actual, const Tensor<float>& expected) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    Tensor<float> a = actual.Contiguous();
    Tensor<float> e = expected.Contiguous();
    for (size_t i = 0; i < a.Size(); ++i) EXPECT_NEAR(a.Data()[i], e.Data()[i], 1e-6f) << "at flat index " << i;
}

}  // namespace

TEST(MakeNorm, BuildsRMSNormFromConfig) {
    std::shared_ptr<Norm<float>> norm = MakeNorm<float>(Config(NormType::RMSNorm, 1e-6f));

    auto* rms = dynamic_cast<RMSNorm<float>*>(norm.get());
    ASSERT_NE(rms, nullptr);
    EXPECT_EQ(rms->weight->Shape(), (std::vector<size_t>{6}));
    EXPECT_FLOAT_EQ(rms->epsilon, 1e-6f);
}

TEST(MakeNorm, BuildsLayerNormFromConfig) {
    std::shared_ptr<Norm<float>> norm = MakeNorm<float>(Config(NormType::LayerNorm, 1e-3f));

    auto* ln = dynamic_cast<LayerNorm<float>*>(norm.get());
    ASSERT_NE(ln, nullptr);
    EXPECT_EQ(ln->NormalizedShape(), (std::vector<size_t>{6}));
    EXPECT_FLOAT_EQ(ln->Epsilon(), 1e-3f);
}

TEST(MakeNorm, RMSNormUsesConfigEpsilonNotItsDefault) {
    std::shared_ptr<Norm<float>> norm = MakeNorm<float>(Config(NormType::RMSNorm, 1e-5f));

    EXPECT_FLOAT_EQ(dynamic_cast<RMSNorm<float>&>(*norm).epsilon, 1e-5f);
}

TEST(MakeNorm, ForwardThroughBaseMatchesConcreteClass) {
    const Tensor<float> x = Input();

    std::shared_ptr<Norm<float>> rms = MakeNorm<float>(Config(NormType::RMSNorm));
    ExpectNear(rms->Forward(x), RMSNorm<float>(6, 1e-5f).Forward(x));

    std::shared_ptr<Norm<float>> ln = MakeNorm<float>(Config(NormType::LayerNorm));
    ExpectNear(ln->Forward(x), LayerNorm<float>(6, 1e-5f).Forward(x));
}

TEST(MakeNorm, ParametersAreVisibleThroughTheBase) {
    EXPECT_EQ(MakeNorm<float>(Config(NormType::RMSNorm))->NamedParameters().size(), 1u);
    EXPECT_EQ(MakeNorm<float>(Config(NormType::LayerNorm))->NamedParameters().size(), 2u);
}

TEST(MakeNorm, EachCallReturnsAnIndependentNorm) {
    auto first = MakeNorm<float>(Config(NormType::RMSNorm));
    auto second = MakeNorm<float>(Config(NormType::RMSNorm));

    EXPECT_NE(first.get(), second.get());
    EXPECT_NE(first->NamedParameters()[0].tensor.get(), second->NamedParameters()[0].tensor.get());
}

TEST(MakeNorm, RejectsUnknownNormType) {
    EXPECT_THROW(MakeNorm<float>(Config(static_cast<NormType>(99))), std::invalid_argument);
}
