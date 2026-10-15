#include <gtest/gtest.h>

#include <NeuralNet/Linear.h>
#include <NeuralNet/Sequential.h>
#include <Tensor/Tensor.h>

#include <stdexcept>
#include <vector>

using nn::Sequential;
using tensor::Device;
using tensor::Tensor;

namespace {

template <typename T>
Tensor<T> Pattern(const std::vector<size_t>& shape, int seed) {
    Tensor<T> t(shape);
    T* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i)
        data[i] = T(static_cast<int>((i * 7 + seed * 13) % 11) - 5) * T(0.25);
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
void FillParameters(nn::Module<T>& m, int seed) {
    for (const auto& [name, param] : m.NamedParameters()) *param = Pattern<T>(param->Shape(), seed++);
}

}

// --- tensor::Linear on CUDA -------------------------------------------------

TEST(TensorLinearCUDA, MatchesCPUWithBias) {
    auto X = Pattern<float>({6, 8}, 1);
    auto W = Pattern<float>({5, 8}, 2);
    auto B = Pattern<float>({5}, 3);

    Tensor<float> Y = tensor::Linear(X.To(Device::CUDA), W.To(Device::CUDA), B.To(Device::CUDA));

    EXPECT_TRUE(Y.IsCUDA());
    ExpectNear(Y, tensor::Linear(X, W, B), 1e-5f);
}

TEST(TensorLinearCUDA, MatchesCPUWithoutBias) {
    auto X = Pattern<float>({4, 3}, 4);
    auto W = Pattern<float>({7, 3}, 5);
    ExpectNear(tensor::Linear(X.To(Device::CUDA), W.To(Device::CUDA)), tensor::Linear(X, W), 1e-5f);
}

TEST(TensorLinearCUDA, KeepsLeadingDimensionsOf3DInput) {
    auto X = Pattern<float>({2, 3, 8}, 6);
    auto W = Pattern<float>({4, 8}, 7);
    auto B = Pattern<float>({4}, 8);

    Tensor<float> Y = tensor::Linear(X.To(Device::CUDA), W.To(Device::CUDA), B.To(Device::CUDA));

    EXPECT_EQ(Y.Shape(), (std::vector<size_t>{2, 3, 4}));
    ExpectNear(Y, tensor::Linear(X, W, B), 1e-5f);
}

TEST(TensorLinearCUDA, SingleTokenRow) {
    auto X = Pattern<float>({1, 1, 64}, 9);
    auto W = Pattern<float>({48, 64}, 10);
    auto B = Pattern<float>({48}, 11);
    ExpectNear(tensor::Linear(X.To(Device::CUDA), W.To(Device::CUDA), B.To(Device::CUDA)),
               tensor::Linear(X, W, B), 1e-4f);
}

TEST(TensorLinearCUDA, WorksForDouble) {
    auto X = Pattern<double>({5, 9}, 1);
    auto W = Pattern<double>({3, 9}, 2);
    auto B = Pattern<double>({3}, 3);
    ExpectNear(tensor::Linear(X.To(Device::CUDA), W.To(Device::CUDA), B.To(Device::CUDA)),
               tensor::Linear(X, W, B), 1e-12);
}

TEST(TensorLinearCUDA, LargerShapeMatchesCPU) {
    auto X = Pattern<float>({37, 130}, 1);
    auto W = Pattern<float>({70, 130}, 2);
    auto B = Pattern<float>({70}, 3);
    ExpectNear(tensor::Linear(X.To(Device::CUDA), W.To(Device::CUDA), B.To(Device::CUDA)),
               tensor::Linear(X, W, B), 1e-3f);
}

TEST(TensorLinearCUDA, InputOnCPUThrows) {
    auto X = Pattern<float>({2, 3}, 1);
    Tensor<float> W({4, 3}, Device::CUDA);
    EXPECT_THROW(tensor::Linear(X, W), std::invalid_argument);
}

TEST(TensorLinearCUDA, BiasOnCPUThrows) {
    Tensor<float> X({2, 3}, Device::CUDA);
    Tensor<float> W({4, 3}, Device::CUDA);
    Tensor<float> B({4}, 1.0f);
    EXPECT_THROW(tensor::Linear(X, W, B), std::invalid_argument);
}

TEST(TensorLinearCUDA, FeatureMismatchThrows) {
    Tensor<float> X({2, 5}, Device::CUDA);
    Tensor<float> W({4, 3}, Device::CUDA);
    EXPECT_THROW(tensor::Linear(X, W), std::invalid_argument);
}

TEST(TensorLinearCUDA, WrongBiasSizeThrows) {
    Tensor<float> X({2, 3}, Device::CUDA);
    Tensor<float> W({4, 3}, Device::CUDA);
    Tensor<float> B({3}, Device::CUDA);
    EXPECT_THROW(tensor::Linear(X, W, B), std::invalid_argument);
}

TEST(TensorLinearCUDA, TransposedInputMatchesCPU) {
    auto X = Pattern<float>({3, 2}, 1);
    auto W = Pattern<float>({4, 3}, 2);
    auto B = Pattern<float>({4}, 3);
    Tensor<float> gx = X.To(Device::CUDA).Transpose();
    ASSERT_FALSE(gx.IsContiguous());
    ExpectNear(tensor::Linear(gx, W.To(Device::CUDA), B.To(Device::CUDA)),
               tensor::Linear(X.Transpose(), W, B), 1e-5f);
}

TEST(TensorLinearCUDA, TransposedWeightMatchesCPU) {
    auto X = Pattern<float>({5, 6}, 4);
    auto Wt = Pattern<float>({6, 3}, 5);
    Tensor<float> gw = Wt.To(Device::CUDA).Transpose();
    ASSERT_FALSE(gw.IsContiguous());
    ExpectNear(tensor::Linear(X.To(Device::CUDA), gw), tensor::Linear(X, Wt.Transpose()), 1e-5f);
}

TEST(TensorLinearCUDA, StridedBiasMatchesCPU) {
    auto X = Pattern<float>({2, 3}, 6);
    auto W = Pattern<float>({4, 3}, 7);
    auto Bsrc = Pattern<float>({4, 2}, 8);
    Tensor<float> gb = Bsrc.To(Device::CUDA).Slice(1, 1, 1).Squeeze(1);
    ASSERT_EQ(gb.Shape(), (std::vector<size_t>{4}));
    ASSERT_FALSE(gb.IsContiguous());
    ExpectNear(tensor::Linear(X.To(Device::CUDA), W.To(Device::CUDA), gb),
               tensor::Linear(X, W, Bsrc.Slice(1, 1, 1).Squeeze(1)), 1e-5f);
}

TEST(TensorLinearCUDA, ColumnSliceOfInputMatchesCPU) {
    auto X = Pattern<float>({4, 10}, 9);
    auto W = Pattern<float>({3, 6}, 10);
    Tensor<float> gx = X.To(Device::CUDA).Slice(1, 2, 6);
    ASSERT_FALSE(gx.IsContiguous());
    ExpectNear(tensor::Linear(gx, W.To(Device::CUDA)), tensor::Linear(X.Slice(1, 2, 6), W), 1e-5f);
}

TEST(TensorLinearCUDA, NonContiguous3DInputMatchesCPU) {
    auto X = Pattern<float>({3, 2, 8}, 11);
    auto W = Pattern<float>({5, 8}, 12);
    auto B = Pattern<float>({5}, 13);
    Tensor<float> gx = X.To(Device::CUDA).Transpose(0, 1);
    ASSERT_FALSE(gx.IsContiguous());
    Tensor<float> Y = tensor::Linear(gx, W.To(Device::CUDA), B.To(Device::CUDA));
    EXPECT_EQ(Y.Shape(), (std::vector<size_t>{2, 3, 5}));
    ExpectNear(Y, tensor::Linear(X.Transpose(0, 1), W, B), 1e-5f);
}

// --- nn::Linear after Module::To --------------------------------------------

TEST(LinearModuleCUDA, ForwardMatchesCPU) {
    nn::Linear<float> layer(8, 5);
    FillParameters(layer, 1);
    auto X = Pattern<float>({3, 8}, 20);
    const Tensor<float> expected = layer.Forward(X);

    layer.To(Device::CUDA);
    Tensor<float> Y = layer.Forward(X.To(Device::CUDA));

    EXPECT_TRUE(Y.IsCUDA());
    ExpectNear(Y, expected, 1e-5f);
}

TEST(LinearModuleCUDA, ForwardWithoutBiasMatchesCPU) {
    nn::Linear<float> layer(6, 4, false);
    FillParameters(layer, 2);
    auto X = Pattern<float>({2, 3, 6}, 21);
    const Tensor<float> expected = layer.Forward(X);

    layer.To(Device::CUDA);

    ExpectNear(layer.Forward(X.To(Device::CUDA)), expected, 1e-5f);
}

TEST(LinearModuleCUDA, CPUInputThrows) {
    nn::Linear<float> layer(3, 2);
    layer.To(Device::CUDA);
    EXPECT_THROW(layer.Forward(Pattern<float>({1, 3}, 1)), std::invalid_argument);
}

TEST(LinearModuleCUDA, SequentialStaysOnGPU) {
    Sequential model(nn::Linear<float>(8, 6), nn::Linear<float>(6, 3));
    FillParameters(model, 3);
    auto X = Pattern<float>({4, 8}, 22);
    const Tensor<float> expected = model.Forward(X);

    model.To(Device::CUDA);
    Tensor<float> Y = model.Forward(X.To(Device::CUDA));

    EXPECT_TRUE(Y.IsCUDA());
    ExpectNear(Y, expected, 1e-4f);
}

TEST(LinearModuleCUDA, BackToCPUGivesSameResult) {
    nn::Linear<float> layer(5, 3);
    FillParameters(layer, 4);
    auto X = Pattern<float>({2, 5}, 23);
    const Tensor<float> expected = layer.Forward(X);

    layer.To(Device::CUDA);
    layer.To(Device::CPU);

    ExpectNear(layer.Forward(X), expected, 0.0f);
}
