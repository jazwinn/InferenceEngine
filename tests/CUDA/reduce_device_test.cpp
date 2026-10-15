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

// --- mean along an axis --------------------------------------------------------

namespace {

template <typename T>
void ExpectMeanAxisMatchesCPU(const std::vector<size_t>& shape, size_t axis, bool keepdim, T tol) {
    auto cpu = Ramp<T>(shape);
    Tensor<T> gpu = cpu.To(Device::CUDA).Mean(axis, keepdim);
    Tensor<T> expected = cpu.Mean(axis, keepdim);
    ASSERT_TRUE(gpu.IsCUDA());
    ASSERT_EQ(gpu.Shape(), expected.Shape());
    Tensor<T> back = gpu.To(Device::CPU);
    for (size_t i = 0; i < expected.Size(); ++i) {
        EXPECT_NEAR(back.Data()[i], expected.Data()[i], tol) << "at index " << i;
    }
}

}

TEST(MeanAxisCUDA, LastAxisKeepdimMatchesCPU) {
    ExpectMeanAxisMatchesCPU<float>({4, 6}, 1, true, 1e-6f);
}

TEST(MeanAxisCUDA, LastAxisNoKeepdimMatchesCPU) {
    ExpectMeanAxisMatchesCPU<float>({4, 6}, 1, false, 1e-6f);
}

TEST(MeanAxisCUDA, FirstAxisMatchesCPU) {
    ExpectMeanAxisMatchesCPU<float>({5, 7}, 0, false, 1e-6f);
    ExpectMeanAxisMatchesCPU<float>({5, 7}, 0, true, 1e-6f);
}

TEST(MeanAxisCUDA, MiddleAxisOf3DMatchesCPU) {
    ExpectMeanAxisMatchesCPU<float>({2, 3, 4}, 1, false, 1e-6f);
    ExpectMeanAxisMatchesCPU<float>({2, 3, 4}, 1, true, 1e-6f);
}

TEST(MeanAxisCUDA, EveryAxisOf4DMatchesCPU) {
    for (size_t axis = 0; axis < 4; ++axis) {
        ExpectMeanAxisMatchesCPU<float>({2, 3, 4, 5}, axis, false, 1e-5f);
        ExpectMeanAxisMatchesCPU<float>({2, 3, 4, 5}, axis, true, 1e-5f);
    }
}

TEST(MeanAxisCUDA, KnownValuesMiddleAxis) {
    Tensor<float> cpu({2, 3, 4});
    for (size_t i = 0; i < cpu.Size(); ++i) cpu.Data()[i] = static_cast<float>(i);
    Tensor<float> out = cpu.To(Device::CUDA).Mean(1).To(Device::CPU);
    const std::vector<float> expected{4, 5, 6, 7, 16, 17, 18, 19};
    ASSERT_EQ(out.Shape(), (std::vector<size_t>{2, 4}));
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_FLOAT_EQ(out.Data()[i], expected[i]) << "at index " << i;
}

TEST(MeanAxisCUDA, RowLongerThanBlockLoops) {
    ExpectMeanAxisMatchesCPU<float>({3, 1000}, 1, true, 1e-5f);
}

TEST(MeanAxisCUDA, RowShorterThanBlockPadsWithZero) {
    Tensor<float> cpu({2, 3}, 4.0f);
    Tensor<float> out = cpu.To(Device::CUDA).Mean(1).To(Device::CPU);
    EXPECT_FLOAT_EQ(out.Data()[0], 4.0f);
    EXPECT_FLOAT_EQ(out.Data()[1], 4.0f);
}

TEST(MeanAxisCUDA, ReducedLengthOne) {
    ExpectMeanAxisMatchesCPU<float>({4, 1, 3}, 1, false, 0.0f);
}

TEST(MeanAxisCUDA, Rank1) {
    ExpectMeanAxisMatchesCPU<float>({9}, 0, true, 1e-6f);
}

TEST(MeanAxisCUDA, ManyRowsManyBlocks) {
    ExpectMeanAxisMatchesCPU<float>({700, 64}, 1, true, 1e-5f);
}

TEST(MeanAxisCUDA, Double) {
    ExpectMeanAxisMatchesCPU<double>({3, 5, 7}, 2, true, 1e-12);
    ExpectMeanAxisMatchesCPU<double>({3, 5, 7}, 0, false, 1e-12);
}

TEST(MeanAxisCUDA, RMSNormPattern) {
    auto cpu = Ramp<float>({2, 5, 16});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> g = (gpu * gpu).Mean(2, true);
    Tensor<float> c = (cpu * cpu).Mean(2, true);
    ASSERT_EQ(g.Shape(), (std::vector<size_t>{2, 5, 1}));
    Tensor<float> back = g.To(Device::CPU);
    for (size_t i = 0; i < c.Size(); ++i) EXPECT_NEAR(back.Data()[i], c.Data()[i], 1e-5f) << "at index " << i;
}

TEST(MeanAxisCUDA, TransposedViewMatchesCPU) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> t = cpu.To(Device::CUDA).Transpose();
    ASSERT_FALSE(t.IsContiguous());
    Tensor<float> out = t.Mean(1).To(Device::CPU);
    Tensor<float> expected = cpu.Transpose().Mean(1);
    ASSERT_EQ(out.Shape(), expected.Shape());
    for (size_t i = 0; i < expected.Size(); ++i) EXPECT_NEAR(out.Data()[i], expected.Data()[i], 1e-6f);
}

TEST(MeanAxisCUDA, ColumnSliceMatchesCPU) {
    auto cpu = Ramp<float>({4, 6});
    Tensor<float> s = cpu.To(Device::CUDA).Slice(1, 1, 3);
    ASSERT_FALSE(s.IsContiguous());
    Tensor<float> out = s.Mean(1).To(Device::CPU);
    Tensor<float> expected = cpu.Slice(1, 1, 3).Mean(1);
    for (size_t i = 0; i < expected.Size(); ++i) EXPECT_NEAR(out.Data()[i], expected.Data()[i], 1e-6f);
}

TEST(MeanAxisCUDA, DoesNotMutateSource) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu.Mean(1);
    Tensor<float> back = gpu.To(Device::CPU);
    for (size_t i = 0; i < cpu.Size(); ++i) EXPECT_EQ(back.Data()[i], cpu.Data()[i]);
}

TEST(MeanAxisCUDA, AxisOutOfRangeThrows) {
    Tensor<float> gpu({2, 3}, Device::CUDA);
    EXPECT_THROW(gpu.Mean(2), std::invalid_argument);
}

TEST(MeanAxisCUDA, FreeFormForwardsToTheMember) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> a = tensor::Mean(cpu.To(Device::CUDA), 1, true).To(Device::CPU);
    Tensor<float> b = cpu.Mean(1, true);
    for (size_t i = 0; i < b.Size(); ++i) EXPECT_NEAR(a.Data()[i], b.Data()[i], 1e-6f);
}

// --- mean axis launcher --------------------------------------------------------

TEST(MeanAxisLauncher, ZeroOutputsIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchMeanAxisDevice<float>(nullptr, nullptr, 0, 4, 1));
}

TEST(MeanAxisLauncher, ZeroReducedIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchMeanAxisDevice<float>(nullptr, nullptr, 2, 0, 1));
}

TEST(MeanAxisLauncher, OverwritesOutputWithoutZeroing) {
    Tensor<float> in = Tensor<float>({2, 4}, 2.0f).To(Device::CUDA);
    Tensor<float> out = Tensor<float>({2}, 99.0f).To(Device::CUDA);
    cuda::LaunchMeanAxisDevice(in.Data(), out.Data(), 2, 4, 1);
    Tensor<float> back = out.To(Device::CPU);
    EXPECT_FLOAT_EQ(back.Data()[0], 2.0f);
    EXPECT_FLOAT_EQ(back.Data()[1], 2.0f);
}
