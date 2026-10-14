#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>
#include <Tensor/Tensor.h>

#include <vector>

using tensor::Device;
using tensor::Tensor;

namespace {

template <typename T>
Tensor<T> Ramp(const std::vector<size_t>& shape) {
    Tensor<T> t(shape);
    T* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i) {
        data[i] = T(static_cast<int>(i % 13) - 6) * T(0.5);
    }
    return t;
}

template <typename T>
T Scalar(const Tensor<T>& t) {
    return t.To(Device::CPU).Data()[0];
}

}

// --- mean over the whole tensor ------------------------------------------------

TEST(MeanAllCUDA, MatchesCPU) {
    auto cpu = Ramp<float>({3, 5});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> out = gpu.Mean();
    EXPECT_TRUE(out.IsCUDA());
    EXPECT_EQ(out.Shape(), (std::vector<size_t>{1}));
    EXPECT_NEAR(Scalar(out), Scalar(cpu.Mean()), 1e-6f);
}

TEST(MeanAllCUDA, ReturnsMeanNotSum) {
    Tensor<float> gpu = Tensor<float>({4}, 3.0f).To(Device::CUDA);
    EXPECT_FLOAT_EQ(Scalar(gpu.Mean()), 3.0f);
}

TEST(MeanAllCUDA, KnownValues) {
    Tensor<float> cpu({4});
    cpu(0) = 1.0f; cpu(1) = 2.0f; cpu(2) = 3.0f; cpu(3) = 10.0f;
    EXPECT_FLOAT_EQ(Scalar(cpu.To(Device::CUDA).Mean()), 4.0f);
}

TEST(MeanAllCUDA, Double) {
    auto cpu = Ramp<double>({7, 9});
    EXPECT_NEAR(Scalar(cpu.To(Device::CUDA).Mean()), Scalar(cpu.Mean()), 1e-12);
}

TEST(MeanAllCUDA, SingleElement) {
    Tensor<float> gpu = Tensor<float>({1}, -2.5f).To(Device::CUDA);
    EXPECT_FLOAT_EQ(Scalar(gpu.Mean()), -2.5f);
}

TEST(MeanAllCUDA, PartialLastBlockIsPaddedWithZero) {
    auto cpu = Ramp<float>({257});
    EXPECT_NEAR(Scalar(cpu.To(Device::CUDA).Mean()), Scalar(cpu.Mean()), 1e-5f);
}

TEST(MeanAllCUDA, ManyBlocksAccumulate) {
    Tensor<float> gpu = Tensor<float>({1000, 37}, 1.5f).To(Device::CUDA);
    EXPECT_NEAR(Scalar(gpu.Mean()), 1.5f, 1e-5f);
}

TEST(MeanAllCUDA, LargeRampMatchesCPU) {
    auto cpu = Ramp<float>({513, 129});
    EXPECT_NEAR(Scalar(cpu.To(Device::CUDA).Mean()), Scalar(cpu.Mean()), 1e-4f);
}

TEST(MeanAllCUDA, CalledTwiceGivesSameResult) {
    Tensor<float> gpu = Ramp<float>({6, 6}).To(Device::CUDA);
    EXPECT_NEAR(Scalar(gpu.Mean()), Scalar(gpu.Mean()), 1e-6f);
}

TEST(MeanAllCUDA, DoesNotMutateSource) {
    auto cpu = Ramp<float>({2, 5});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu.Mean();
    Tensor<float> back = gpu.To(Device::CPU);
    for (size_t i = 0; i < cpu.Size(); ++i) EXPECT_EQ(back.Data()[i], cpu.Data()[i]);
}

TEST(MeanAllCUDA, TransposedView) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> t = cpu.To(Device::CUDA).Transpose();
    ASSERT_FALSE(t.IsContiguous());
    EXPECT_NEAR(Scalar(t.Mean()), Scalar(cpu.Mean()), 1e-6f);
}

TEST(MeanAllCUDA, NonContiguousColumnSliceOnlyAveragesItsElements) {
    Tensor<float> cpu({3, 4});
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c) cpu(r, c) = (c == 1 || c == 2) ? 2.0f : 100.0f;
    Tensor<float> col = cpu.To(Device::CUDA).Slice(1, 1, 2);
    ASSERT_FALSE(col.IsContiguous());
    EXPECT_FLOAT_EQ(Scalar(col.Mean()), 2.0f);
}

TEST(MeanAllCUDA, OffsetRowSlice) {
    auto cpu = Ramp<float>({4, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    EXPECT_NEAR(Scalar(gpu.Slice(0, 1, 2).Mean()), Scalar(cpu.Slice(0, 1, 2).Mean()), 1e-6f);
}

TEST(MeanAllCUDA, FreeFormForwardsToTheMember) {
    auto cpu = Ramp<float>({2, 3});
    EXPECT_NEAR(Scalar(tensor::Mean(cpu.To(Device::CUDA))), Scalar(cpu.Mean()), 1e-6f);
}

// --- sum launcher --------------------------------------------------------------

TEST(SumLauncher, ZeroSizeIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchSumDevice<float>(nullptr, nullptr, 0));
}

TEST(SumLauncher, AccumulatesIntoExistingOutput) {
    Tensor<float> in = Tensor<float>({10}, 1.0f).To(Device::CUDA);
    Tensor<float> out = Tensor<float>({1}, 5.0f).To(Device::CUDA);
    cuda::LaunchSumDevice(in.Data(), out.Data(), in.Size());
    EXPECT_FLOAT_EQ(Scalar(out), 15.0f);
}

TEST(SumLauncher, SumsExactlySizeElements) {
    Tensor<float> in = Tensor<float>({600}, 2.0f).To(Device::CUDA);
    Tensor<float> out({1}, Device::CUDA);
    cuda::LaunchSumDevice(in.Data(), out.Data(), 300);
    EXPECT_FLOAT_EQ(Scalar(out), 600.0f);
}
