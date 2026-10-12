#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>
#include <NeuralNet/Activation.h>
#include <Tensor/Tensor.h>

#include <string>
#include <vector>

using tensor::Activation;
using tensor::Device;
using tensor::Tensor;

namespace {

template <typename T>
Tensor<T> Ramp(const std::vector<size_t>& shape) {
    Tensor<T> t(shape);
    T* data = t.Data();
    const size_t n = t.Size();
    for (size_t i = 0; i < n; ++i) {
        data[i] = n == 1 ? T(0.5) : T(-4) + T(8) * T(i) / T(n - 1);
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
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], e[i], tol) << "at index " << i;
    }
}

std::string Name(Activation kind) {
    switch (kind) {
    case Activation::ReLU:     return "ReLU";
    case Activation::GELU:     return "GELU";
    case Activation::GELUTanh: return "GELUTanh";
    case Activation::Sigmoid:  return "Sigmoid";
    case Activation::SiLU:     return "SiLU";
    case Activation::Tanh:     return "Tanh";
    }
    return "Unknown";
}

}

// --- tensor::Activate on CUDA ------------------------------------------------

class ActivateCUDA : public ::testing::TestWithParam<Activation> {};

TEST_P(ActivateCUDA, FloatMatchesCPU) {
    auto x = Ramp<float>({4, 33});
    Tensor<float> y = tensor::Activate(x.To(Device::CUDA), GetParam());
    EXPECT_TRUE(y.IsCUDA());
    EXPECT_TRUE(y.IsContiguous());
    ExpectNear(y, tensor::Activate(x, GetParam()), 1e-5f);
}

TEST_P(ActivateCUDA, DoubleMatchesCPU) {
    auto x = Ramp<double>({3, 17});
    ExpectNear(tensor::Activate(x.To(Device::CUDA), GetParam()), tensor::Activate(x, GetParam()), 1e-12);
}

TEST_P(ActivateCUDA, NonContiguousInputMatchesCPU) {
    auto x = Ramp<float>({5, 7});
    Tensor<float> y = tensor::Activate(x.To(Device::CUDA).Transpose(), GetParam());
    EXPECT_EQ(y.Shape(), (std::vector<size_t>{7, 5}));
    ExpectNear(y, tensor::Activate(x.Transpose(), GetParam()), 1e-5f);
}

TEST_P(ActivateCUDA, SpansManyBlocks) {
    auto x = Ramp<float>({10007});
    ExpectNear(tensor::Activate(x.To(Device::CUDA), GetParam()), tensor::Activate(x, GetParam()), 1e-5f);
}

TEST_P(ActivateCUDA, KeepsFourDimensionalShape) {
    auto x = Ramp<float>({2, 3, 4, 5});
    Tensor<float> y = tensor::Activate(x.To(Device::CUDA), GetParam());
    EXPECT_EQ(y.Shape(), x.Shape());
    ExpectNear(y, tensor::Activate(x, GetParam()), 1e-5f);
}

TEST_P(ActivateCUDA, LeavesInputUnchanged) {
    auto x = Ramp<float>({3, 4});
    Tensor<float> gpu = x.To(Device::CUDA);
    Tensor<float> y = tensor::Activate(gpu, GetParam());
    EXPECT_FALSE(y.SharesStorage(gpu));
    EXPECT_EQ(Values(gpu), Values(x));
}

TEST_P(ActivateCUDA, SingleElement) {
    auto x = Ramp<float>({1});
    ExpectNear(tensor::Activate(x.To(Device::CUDA), GetParam()), tensor::Activate(x, GetParam()), 1e-6f);
}

INSTANTIATE_TEST_SUITE_P(
    AllActivations, ActivateCUDA,
    ::testing::Values(Activation::ReLU, Activation::GELU, Activation::GELUTanh,
                      Activation::Sigmoid, Activation::SiLU, Activation::Tanh),
    [](const ::testing::TestParamInfo<Activation>& info) { return Name(info.param); });

// --- known values ------------------------------------------------------------

TEST(ActivateCUDAValues, ReLUZeroesNegativesOnly) {
    Tensor<float> x({4});
    float* d = x.Data();
    d[0] = -2.0f; d[1] = -0.0f; d[2] = 0.0f; d[3] = 3.5f;
    EXPECT_EQ(Values(tensor::Activate(x.To(Device::CUDA), Activation::ReLU)),
              (std::vector<float>{0.0f, 0.0f, 0.0f, 3.5f}));
}

TEST(ActivateCUDAValues, SigmoidOfZeroIsHalf) {
    Tensor<float> x({1}, Device::CUDA);
    EXPECT_FLOAT_EQ(Values(tensor::Activate(x, Activation::Sigmoid))[0], 0.5f);
}

TEST(ActivateCUDAValues, SiLUAndGELUOfZeroAreZero) {
    Tensor<float> x({2}, Device::CUDA);
    EXPECT_EQ(Values(tensor::Activate(x, Activation::SiLU)), (std::vector<float>{0.0f, 0.0f}));
    EXPECT_EQ(Values(tensor::Activate(x, Activation::GELU)), (std::vector<float>{0.0f, 0.0f}));
}

// --- nn activation modules on CUDA -------------------------------------------

TEST(ActivationModulesCUDA, EveryModuleMatchesCPU) {
    auto x = Ramp<float>({3, 8});
    Tensor<float> gpu = x.To(Device::CUDA);
    ExpectNear(nn::ReLU<float>().Forward(gpu), nn::ReLU<float>().Forward(x), 1e-5f);
    ExpectNear(nn::GELU<float>().Forward(gpu), nn::GELU<float>().Forward(x), 1e-5f);
    ExpectNear(nn::GELUTanh<float>().Forward(gpu), nn::GELUTanh<float>().Forward(x), 1e-5f);
    ExpectNear(nn::Sigmoid<float>().Forward(gpu), nn::Sigmoid<float>().Forward(x), 1e-5f);
    ExpectNear(nn::SiLU<float>().Forward(gpu), nn::SiLU<float>().Forward(x), 1e-5f);
    ExpectNear(nn::Tanh<float>().Forward(gpu), nn::Tanh<float>().Forward(x), 1e-5f);
}

TEST(ActivationModulesCUDA, OutputStaysOnGPU) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    EXPECT_TRUE(nn::SiLU<float>().Forward(gpu).IsCUDA());
}

// --- launcher ----------------------------------------------------------------

TEST(ActivationLauncher, ZeroSizeIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchActivationDevice<float>(nullptr, nullptr, 0, cuda::Activation::ReLU));
}

TEST(ActivationLauncher, WorksInPlace) {
    auto x = Ramp<float>({16});
    Tensor<float> gpu = x.To(Device::CUDA);
    cuda::LaunchActivationDevice(gpu.Data(), gpu.Data(), gpu.Size(), cuda::Activation::ReLU);
    ExpectNear(gpu, tensor::Activate(x, Activation::ReLU), 0.0f);
}
