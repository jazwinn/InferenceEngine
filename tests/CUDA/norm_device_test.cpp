#include <gtest/gtest.h>

#include <NeuralNet/Transformer/Normal/LayerNorm.h>
#include <NeuralNet/Transformer/Normal/RMSNorm.h>
#include <Tensor/Tensor.h>

#include <cmath>
#include <vector>

using nn::transformer::norm::LayerNorm;
using nn::transformer::norm::RMSNorm;
using tensor::Device;
using tensor::Tensor;

namespace {

template <typename T>
Tensor<T> Wave(const std::vector<size_t>& shape, double phase) {
    Tensor<T> t(shape);
    T* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i) {
        data[i] = static_cast<T>(std::sin(0.37 * static_cast<double>(i) + phase) * 2.0);
    }
    return t;
}

template <typename T>
std::vector<T> Values(const Tensor<T>& t) {
    Tensor<T> cpu = t.To(Device::CPU).Contiguous();
    return std::vector<T>(cpu.Data(), cpu.Data() + cpu.Size());
}

template <typename T>
void ExpectNear(const Tensor<T>& actual, const Tensor<T>& expected, T tol) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    const auto a = Values(actual);
    const auto e = Values(expected);
    for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], e[i], tol) << "at index " << i;
}

template <typename T>
void ExpectGPUMatchesCPU(const std::vector<size_t>& shape, T epsilon, T tol) {
    const size_t dim = shape.back();
    RMSNorm<T> cpu_norm(dim, epsilon);
    *cpu_norm.weight = Wave<T>({dim}, 0.5);
    RMSNorm<T> gpu_norm(dim, epsilon);
    *gpu_norm.weight = *cpu_norm.weight;
    gpu_norm.To(Device::CUDA);

    auto x = Wave<T>(shape, 1.0);
    Tensor<T> expected = cpu_norm.Forward(x);
    Tensor<T> actual = gpu_norm.Forward(x.To(Device::CUDA));

    EXPECT_TRUE(actual.IsCUDA());
    ExpectNear(actual, expected, tol);
}

}

TEST(RMSNormCUDA, TwoDimensionalMatchesCPU) {
    ExpectGPUMatchesCPU<float>({4, 16}, 1e-5f, 1e-5f);
}

TEST(RMSNormCUDA, PromptBatchMatchesCPU) {
    ExpectGPUMatchesCPU<float>({2, 7, 32}, 1e-5f, 1e-5f);
}

TEST(RMSNormCUDA, SingleTokenMatchesCPU) {
    ExpectGPUMatchesCPU<float>({1, 1, 64}, 1e-5f, 1e-5f);
}

TEST(RMSNormCUDA, ModelSizedHiddenDimMatchesCPU) {
    ExpectGPUMatchesCPU<float>({3, 2048}, 1e-5f, 1e-4f);
}

TEST(RMSNormCUDA, DefaultEpsilonMatchesCPU) {
    ExpectGPUMatchesCPU<float>({3, 8}, 1e-8f, 1e-5f);
}

TEST(RMSNormCUDA, LargeEpsilonMatchesCPU) {
    ExpectGPUMatchesCPU<float>({3, 8}, 0.5f, 1e-5f);
}

TEST(RMSNormCUDA, DoubleMatchesCPU) {
    ExpectGPUMatchesCPU<double>({2, 5, 24}, 1e-6, 1e-12);
}

TEST(RMSNormCUDA, KnownValues) {
    RMSNorm<float> norm(2, 0.0f);
    norm.To(Device::CUDA);
    Tensor<float> x({1, 2});
    x(0, 0) = 3.0f;
    x(0, 1) = 4.0f;
    const auto out = Values(norm.Forward(x.To(Device::CUDA)));
    const float rms = std::sqrt((9.0f + 16.0f) / 2.0f);
    EXPECT_NEAR(out[0], 3.0f / rms, 1e-6f);
    EXPECT_NEAR(out[1], 4.0f / rms, 1e-6f);
}

TEST(RMSNormCUDA, WeightScalesEachFeature) {
    RMSNorm<float> norm(3, 0.0f);
    Tensor<float> w({3});
    w(0) = 1.0f; w(1) = 2.0f; w(2) = -1.0f;
    *norm.weight = w;
    norm.To(Device::CUDA);
    Tensor<float> x({1, 3}, 2.0f);
    const auto out = Values(norm.Forward(x.To(Device::CUDA)));
    EXPECT_NEAR(out[0], 1.0f, 1e-6f);
    EXPECT_NEAR(out[1], 2.0f, 1e-6f);
    EXPECT_NEAR(out[2], -1.0f, 1e-6f);
}

TEST(RMSNormCUDA, ZeroRowStaysFinite) {
    RMSNorm<float> norm(4, 1e-5f);
    norm.To(Device::CUDA);
    Tensor<float> x({2, 4}, Device::CUDA);
    for (float v : Values(norm.Forward(x))) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_EQ(v, 0.0f);
    }
}

TEST(RMSNormCUDA, NonContiguousInputMatchesCPU) {
    RMSNorm<float> cpu_norm(4, 1e-5f);
    *cpu_norm.weight = Wave<float>({4}, 0.2);
    RMSNorm<float> gpu_norm(4, 1e-5f);
    *gpu_norm.weight = *cpu_norm.weight;
    gpu_norm.To(Device::CUDA);

    auto x = Wave<float>({4, 3}, 0.9);
    Tensor<float> gx = x.To(Device::CUDA).Transpose();
    ASSERT_FALSE(gx.IsContiguous());
    ExpectNear(gpu_norm.Forward(gx), cpu_norm.Forward(x.Transpose()), 1e-5f);
}

TEST(RMSNormCUDA, DoesNotMutateInput) {
    RMSNorm<float> norm(8, 1e-5f);
    norm.To(Device::CUDA);
    auto x = Wave<float>({3, 8}, 0.1);
    Tensor<float> gx = x.To(Device::CUDA);
    norm.Forward(gx);
    EXPECT_EQ(Values(gx), Values(x));
}

TEST(RMSNormCUDA, WeightMovesToCUDA) {
    RMSNorm<float> norm(8);
    norm.To(Device::CUDA);
    EXPECT_TRUE(norm.weight->IsCUDA());
}

TEST(RMSNormCUDA, RoundTripBackToCPUStillWorks) {
    RMSNorm<float> norm(8, 1e-5f);
    auto x = Wave<float>({2, 8}, 0.3);
    Tensor<float> before = norm.Forward(x);
    norm.To(Device::CUDA);
    norm.To(Device::CPU);
    ExpectNear(norm.Forward(x), before, 0.0f);
}

TEST(RMSNormCUDA, CPUInputWithCUDAWeightThrows) {
    RMSNorm<float> norm(4);
    norm.To(Device::CUDA);
    Tensor<float> x({2, 4}, 1.0f);
    EXPECT_ANY_THROW(norm.Forward(x));
}

TEST(RMSNormCUDA, CUDAInputWithCPUWeightThrows) {
    RMSNorm<float> norm(4);
    Tensor<float> x({2, 4}, Device::CUDA);
    EXPECT_ANY_THROW(norm.Forward(x));
}

// --- LayerNorm -----------------------------------------------------------------

namespace {

template <typename T>
void ExpectLayerNormGPUMatchesCPU(const std::vector<size_t>& input_shape,
                                  const std::vector<size_t>& normalized_shape,
                                  T epsilon, T tol) {
    LayerNorm<T> cpu_norm(normalized_shape, epsilon);
    *cpu_norm.gamma = Wave<T>(normalized_shape, 0.5);
    *cpu_norm.beta = Wave<T>(normalized_shape, 2.0);
    LayerNorm<T> gpu_norm(normalized_shape, epsilon);
    *gpu_norm.gamma = *cpu_norm.gamma;
    *gpu_norm.beta = *cpu_norm.beta;
    gpu_norm.To(Device::CUDA);

    auto x = Wave<T>(input_shape, 1.0);
    Tensor<T> expected = cpu_norm.Forward(x);
    Tensor<T> actual = gpu_norm.Forward(x.To(Device::CUDA));

    EXPECT_TRUE(actual.IsCUDA());
    ExpectNear(actual, expected, tol);
}

}

TEST(LayerNormCUDA, TwoDimensionalMatchesCPU) {
    ExpectLayerNormGPUMatchesCPU<float>({4, 16}, {16}, 1e-5f, 1e-5f);
}

TEST(LayerNormCUDA, PromptBatchMatchesCPU) {
    ExpectLayerNormGPUMatchesCPU<float>({2, 7, 32}, {32}, 1e-5f, 1e-5f);
}

TEST(LayerNormCUDA, SingleTokenMatchesCPU) {
    ExpectLayerNormGPUMatchesCPU<float>({1, 1, 64}, {64}, 1e-5f, 1e-5f);
}

TEST(LayerNormCUDA, GPT2SizedHiddenDimMatchesCPU) {
    ExpectLayerNormGPUMatchesCPU<float>({3, 768}, {768}, 1e-5f, 1e-4f);
}

TEST(LayerNormCUDA, MultiDimNormalizedShapeMatchesCPU) {
    ExpectLayerNormGPUMatchesCPU<float>({2, 3, 4, 5}, {4, 5}, 1e-5f, 1e-5f);
}

TEST(LayerNormCUDA, NormalizeWholeInputMatchesCPU) {
    ExpectLayerNormGPUMatchesCPU<float>({3, 4}, {3, 4}, 1e-5f, 1e-5f);
}

TEST(LayerNormCUDA, LargeEpsilonMatchesCPU) {
    ExpectLayerNormGPUMatchesCPU<float>({3, 8}, {8}, 0.5f, 1e-5f);
}

TEST(LayerNormCUDA, DoubleMatchesCPU) {
    ExpectLayerNormGPUMatchesCPU<double>({2, 5, 24}, {24}, 1e-6, 1e-12);
}

TEST(LayerNormCUDA, KnownValues) {
    LayerNorm<float> norm(2, 0.0f);
    norm.To(Device::CUDA);
    Tensor<float> x({1, 2});
    x(0, 0) = 1.0f;
    x(0, 1) = 3.0f;
    const auto out = Values(norm.Forward(x.To(Device::CUDA)));
    EXPECT_NEAR(out[0], -1.0f, 1e-6f);
    EXPECT_NEAR(out[1], 1.0f, 1e-6f);
}

TEST(LayerNormCUDA, OutputHasZeroMeanAndUnitVariance) {
    LayerNorm<float> norm(256, 1e-5f);
    norm.To(Device::CUDA);
    const auto out = Values(norm.Forward(Wave<float>({1, 256}, 0.7).To(Device::CUDA)));
    float mean = 0.0f;
    for (float v : out) mean += v;
    mean /= 256.0f;
    float var = 0.0f;
    for (float v : out) var += (v - mean) * (v - mean);
    var /= 256.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-5f);
    EXPECT_NEAR(var, 1.0f, 1e-3f);
}

TEST(LayerNormCUDA, GammaAndBetaApplyPerFeature) {
    LayerNorm<float> norm(2, 0.0f);
    Tensor<float> g({2}), b({2});
    g(0) = 2.0f; g(1) = 3.0f;
    b(0) = 10.0f; b(1) = -10.0f;
    *norm.gamma = g;
    *norm.beta = b;
    norm.To(Device::CUDA);
    Tensor<float> x({1, 2});
    x(0, 0) = 1.0f;
    x(0, 1) = 3.0f;
    const auto out = Values(norm.Forward(x.To(Device::CUDA)));
    EXPECT_NEAR(out[0], 8.0f, 1e-5f);
    EXPECT_NEAR(out[1], -7.0f, 1e-5f);
}

TEST(LayerNormCUDA, ConstantRowGivesBeta) {
    LayerNorm<float> norm(4, 1e-5f);
    *norm.beta = Wave<float>({4}, 0.3);
    const auto beta = Values(*norm.beta);
    norm.To(Device::CUDA);
    const auto out = Values(norm.Forward(Tensor<float>({2, 4}, 5.0f).To(Device::CUDA)));
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_TRUE(std::isfinite(out[i]));
        EXPECT_NEAR(out[i], beta[i % 4], 1e-5f) << "at index " << i;
    }
}

TEST(LayerNormCUDA, NonContiguousInputMatchesCPU) {
    LayerNorm<float> cpu_norm(4, 1e-5f);
    *cpu_norm.gamma = Wave<float>({4}, 0.2);
    *cpu_norm.beta = Wave<float>({4}, 1.1);
    LayerNorm<float> gpu_norm(4, 1e-5f);
    *gpu_norm.gamma = *cpu_norm.gamma;
    *gpu_norm.beta = *cpu_norm.beta;
    gpu_norm.To(Device::CUDA);

    auto x = Wave<float>({4, 3}, 0.9);
    Tensor<float> gx = x.To(Device::CUDA).Transpose();
    ASSERT_FALSE(gx.IsContiguous());
    ExpectNear(gpu_norm.Forward(gx), cpu_norm.Forward(x.Transpose()), 1e-5f);
}

TEST(LayerNormCUDA, DoesNotMutateInput) {
    LayerNorm<float> norm(8, 1e-5f);
    norm.To(Device::CUDA);
    auto x = Wave<float>({3, 8}, 0.1);
    Tensor<float> gx = x.To(Device::CUDA);
    norm.Forward(gx);
    EXPECT_EQ(Values(gx), Values(x));
}

TEST(LayerNormCUDA, ParametersMoveToCUDA) {
    LayerNorm<float> norm(8);
    norm.To(Device::CUDA);
    EXPECT_TRUE(norm.gamma->IsCUDA());
    EXPECT_TRUE(norm.beta->IsCUDA());
}

TEST(LayerNormCUDA, TrailingShapeMismatchThrows) {
    LayerNorm<float> norm(8);
    norm.To(Device::CUDA);
    EXPECT_THROW(norm.Forward(Tensor<float>({2, 4}, Device::CUDA)), std::invalid_argument);
}

TEST(LayerNormCUDA, CPUInputWithCUDAParametersThrows) {
    LayerNorm<float> norm(3);
    norm.To(Device::CUDA);
    EXPECT_ANY_THROW(norm.Forward(Tensor<float>({2, 3}, 1.0f)));
}

TEST(LayerNormCUDA, CUDAInputWithCPUParametersThrows) {
    LayerNorm<float> norm(3);
    EXPECT_ANY_THROW(norm.Forward(Tensor<float>({2, 3}, Device::CUDA)));
}
