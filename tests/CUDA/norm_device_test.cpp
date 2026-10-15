#include <gtest/gtest.h>

#include <NeuralNet/Transformer/Normal/RMSNorm.h>
#include <Tensor/Tensor.h>

#include <cmath>
#include <vector>

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
