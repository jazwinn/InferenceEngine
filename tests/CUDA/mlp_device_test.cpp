#include <gtest/gtest.h>

#include <NeuralNet/Activation.h>
#include <NeuralNet/Transformer/MLP.h>
#include <Tensor/Tensor.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

using nn::transformer::MLP;
using nn::transformer::config::Activation;
using nn::transformer::config::MLPType;
using nn::transformer::config::TransformerConfig;
using tensor::Device;
using tensor::Tensor;

namespace {

TransformerConfig Config(MLPType type, Activation act, bool bias) {
    TransformerConfig cfg;
    cfg.d_model = 8;
    cfg.ffn_dim = 20;
    cfg.mlp_type = type;
    cfg.activation = act;
    cfg.output_bias = bias;
    return cfg;
}

void WaveFill(Tensor<float>& t, double phase) {
    float* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i) {
        data[i] = static_cast<float>(0.5 * std::sin(phase + 0.37 * static_cast<double>(i)));
    }
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

std::vector<float> Values(const Tensor<float>& t) {
    Tensor<float> cpu = t.To(Device::CPU).Contiguous();
    return std::vector<float>(cpu.Data(), cpu.Data() + cpu.Size());
}

void ExpectNear(const Tensor<float>& actual, const Tensor<float>& expected, float tol) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    const auto a = Values(actual);
    const auto e = Values(expected);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], e[i], tol) << "at index " << i;
    }
}

std::string Name(Activation act) {
    switch (act) {
    case Activation::GELU:     return "GELU";
    case Activation::GELUTanh: return "GELUTanh";
    case Activation::SiLU:     return "SiLU";
    case Activation::ReLU:     return "ReLU";
    case Activation::Sigmoid:  return "Sigmoid";
    case Activation::Tanh:     return "Tanh";
    }
    return "Unknown";
}

}

class MLPOnCUDA : public ::testing::TestWithParam<std::tuple<MLPType, Activation, bool>> {};

TEST_P(MLPOnCUDA, PromptMatchesCPU) {
    const auto [type, act, bias] = GetParam();
    MLP<float> mlp(Config(type, act, bias));
    Seed(mlp);
    Tensor<float> x({2, 5, 8});
    WaveFill(x, 1.7);
    const Tensor<float> expected = mlp.Forward(x);

    mlp.To(Device::CUDA);
    Tensor<float> y = mlp.Forward(x.To(Device::CUDA));

    EXPECT_TRUE(y.IsCUDA());
    ExpectNear(y, expected, 1e-5f);
}

TEST_P(MLPOnCUDA, SingleTokenMatchesCPU) {
    const auto [type, act, bias] = GetParam();
    MLP<float> mlp(Config(type, act, bias));
    Seed(mlp);
    Tensor<float> x({1, 1, 8});
    WaveFill(x, 0.4);
    const Tensor<float> expected = mlp.Forward(x);

    mlp.To(Device::CUDA);

    ExpectNear(mlp.Forward(x.To(Device::CUDA)), expected, 1e-5f);
}

INSTANTIATE_TEST_SUITE_P(
    AllVariants, MLPOnCUDA,
    ::testing::Combine(::testing::Values(MLPType::Standard, MLPType::Gated),
                       ::testing::Values(Activation::GELU, Activation::GELUTanh, Activation::SiLU, Activation::ReLU),
                       ::testing::Bool()),
    [](const auto& info) {
        const MLPType type = std::get<0>(info.param);
        const Activation act = std::get<1>(info.param);
        const bool bias = std::get<2>(info.param);
        return std::string(type == MLPType::Gated ? "Gated" : "Standard") + "_" + Name(act) +
               (bias ? "_Bias" : "_NoBias");
    });

TEST(SwiGLUOnCUDA, MatchesCPU) {
    nn::SwiGLU<float> swiglu(8, 16);
    double phase = 0.3;
    for (const auto& [name, param] : swiglu.NamedParameters()) {
        WaveFill(*param, phase);
        phase += 0.5;
    }
    Tensor<float> x({3, 8});
    WaveFill(x, 2.1);
    const Tensor<float> expected = swiglu.Forward(x);

    swiglu.To(Device::CUDA);
    Tensor<float> y = swiglu.Forward(x.To(Device::CUDA));

    EXPECT_TRUE(y.IsCUDA());
    ExpectNear(y, expected, 1e-5f);
}

TEST(MLPOnCUDAErrors, CPUInputThrowsAfterMove) {
    MLP<float> mlp(Config(MLPType::Gated, Activation::SiLU, false));
    mlp.To(Device::CUDA);
    Tensor<float> x({1, 2, 8}, 1.0f);
    EXPECT_THROW(mlp.Forward(x), std::invalid_argument);
}
