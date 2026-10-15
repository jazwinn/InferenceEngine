#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>
#include <Tensor/Tensor.h>

#include <cmath>
#include <limits>
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

// --- softmax -------------------------------------------------------------------

namespace {

template <typename T>
void ExpectSoftmaxMatchesCPU(const Tensor<T>& cpu, long long axis, T tol) {
    Tensor<T> gpu = tensor::Softmax(cpu.To(Device::CUDA), axis);
    Tensor<T> expected = tensor::Softmax(cpu, axis);
    ASSERT_TRUE(gpu.IsCUDA());
    ASSERT_EQ(gpu.Shape(), expected.Shape());
    Tensor<T> back = gpu.To(Device::CPU);
    for (size_t i = 0; i < expected.Size(); ++i) {
        EXPECT_NEAR(back.Data()[i], expected.Data()[i], tol) << "at index " << i;
    }
}

template <typename T>
Tensor<T> Scaled(const std::vector<size_t>& shape, T scale) {
    Tensor<T> t = Ramp<T>(shape);
    for (size_t i = 0; i < t.Size(); ++i) t.Data()[i] *= scale;
    return t;
}

}

TEST(SoftmaxCUDA, LastAxisMatchesCPU) {
    ExpectSoftmaxMatchesCPU<float>(Ramp<float>({4, 9}), -1, 1e-6f);
}

TEST(SoftmaxCUDA, FirstAxisMatchesCPU) {
    ExpectSoftmaxMatchesCPU<float>(Ramp<float>({5, 7}), 0, 1e-6f);
}

TEST(SoftmaxCUDA, MiddleAxisOf3DMatchesCPU) {
    ExpectSoftmaxMatchesCPU<float>(Ramp<float>({2, 6, 4}), 1, 1e-6f);
}

TEST(SoftmaxCUDA, EveryAxisOf4DMatchesCPU) {
    for (long long axis = 0; axis < 4; ++axis) {
        ExpectSoftmaxMatchesCPU<float>(Ramp<float>({2, 3, 4, 5}), axis, 1e-6f);
    }
}

TEST(SoftmaxCUDA, NegativeAxisMatchesPositive) {
    auto cpu = Ramp<float>({2, 3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> a = tensor::Softmax(gpu, -2).To(Device::CPU);
    Tensor<float> b = tensor::Softmax(gpu, 1).To(Device::CPU);
    for (size_t i = 0; i < a.Size(); ++i) EXPECT_EQ(a.Data()[i], b.Data()[i]);
}

TEST(SoftmaxCUDA, AttentionScoresShapeMatchesCPU) {
    ExpectSoftmaxMatchesCPU<float>(Ramp<float>({1, 4, 7, 7}), -1, 1e-6f);
}

TEST(SoftmaxCUDA, RowLongerThanBlockLoops) {
    ExpectSoftmaxMatchesCPU<float>(Ramp<float>({3, 1000}), -1, 1e-6f);
}

TEST(SoftmaxCUDA, VocabularySizedRowMatchesCPU) {
    ExpectSoftmaxMatchesCPU<float>(Scaled<float>({1, 32000}, 4.0f), -1, 1e-6f);
}

TEST(SoftmaxCUDA, ManyRowsManyBlocks) {
    ExpectSoftmaxMatchesCPU<float>(Ramp<float>({700, 64}), -1, 1e-6f);
}

TEST(SoftmaxCUDA, RowsSumToOne) {
    Tensor<float> out = tensor::Softmax(Ramp<float>({6, 300}).To(Device::CUDA), -1).To(Device::CPU);
    for (size_t r = 0; r < 6; ++r) {
        float sum = 0.0f;
        for (size_t c = 0; c < 300; ++c) sum += out.Data()[r * 300 + c];
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "row " << r;
    }
}

TEST(SoftmaxCUDA, LargeValuesDoNotOverflow) {
    Tensor<float> cpu({2, 4});
    const float vals[] = {1000.0f, 1001.0f, 1002.0f, 1003.0f, -1000.0f, -999.0f, -998.0f, -997.0f};
    for (size_t i = 0; i < 8; ++i) cpu.Data()[i] = vals[i];
    Tensor<float> out = tensor::Softmax(cpu.To(Device::CUDA), -1).To(Device::CPU);
    for (size_t i = 0; i < 8; ++i) EXPECT_TRUE(std::isfinite(out.Data()[i])) << "at index " << i;
    ExpectSoftmaxMatchesCPU<float>(cpu, -1, 1e-6f);
}

TEST(SoftmaxCUDA, ConstantRowIsUniform) {
    Tensor<float> cpu({2, 5}, 3.0f);
    Tensor<float> out = tensor::Softmax(cpu.To(Device::CUDA), -1).To(Device::CPU);
    for (size_t i = 0; i < out.Size(); ++i) EXPECT_FLOAT_EQ(out.Data()[i], 0.2f);
}

TEST(SoftmaxCUDA, MaskedPositionsBecomeZero) {
    Tensor<float> cpu({1, 4});
    cpu(0, 0) = 1.0f; cpu(0, 1) = 2.0f;
    cpu(0, 2) = -std::numeric_limits<float>::infinity();
    cpu(0, 3) = -std::numeric_limits<float>::infinity();
    Tensor<float> out = tensor::Softmax(cpu.To(Device::CUDA), -1).To(Device::CPU);
    EXPECT_EQ(out.Data()[2], 0.0f);
    EXPECT_EQ(out.Data()[3], 0.0f);
    EXPECT_NEAR(out.Data()[0] + out.Data()[1], 1.0f, 1e-6f);
}

TEST(SoftmaxCUDA, MaxNotInFirstThreadSlice) {
    Tensor<float> cpu({1, 600}, -5.0f);
    cpu(0, 599) = 10.0f;
    ExpectSoftmaxMatchesCPU<float>(cpu, -1, 1e-6f);
}

TEST(SoftmaxCUDA, ReducedLengthOneIsAllOnes) {
    Tensor<float> out = tensor::Softmax(Ramp<float>({4, 1}).To(Device::CUDA), -1).To(Device::CPU);
    for (size_t i = 0; i < out.Size(); ++i) EXPECT_FLOAT_EQ(out.Data()[i], 1.0f);
}

TEST(SoftmaxCUDA, Double) {
    ExpectSoftmaxMatchesCPU<double>(Ramp<double>({3, 5, 7}), -1, 1e-12);
    ExpectSoftmaxMatchesCPU<double>(Ramp<double>({3, 5, 7}), 1, 1e-12);
}

TEST(SoftmaxCUDA, TransposedViewMatchesCPU) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> t = cpu.To(Device::CUDA).Transpose();
    ASSERT_FALSE(t.IsContiguous());
    Tensor<float> out = tensor::Softmax(t, -1).To(Device::CPU);
    Tensor<float> expected = tensor::Softmax(cpu.Transpose(), -1);
    ASSERT_EQ(out.Shape(), expected.Shape());
    for (size_t i = 0; i < expected.Size(); ++i) EXPECT_NEAR(out.Data()[i], expected.Data()[i], 1e-6f);
}

TEST(SoftmaxCUDA, ColumnSliceMatchesCPU) {
    auto cpu = Ramp<float>({4, 6});
    Tensor<float> s = cpu.To(Device::CUDA).Slice(1, 1, 4);
    ASSERT_FALSE(s.IsContiguous());
    Tensor<float> out = tensor::Softmax(s, -1).To(Device::CPU);
    Tensor<float> expected = tensor::Softmax(cpu.Slice(1, 1, 4), -1);
    for (size_t i = 0; i < expected.Size(); ++i) EXPECT_NEAR(out.Data()[i], expected.Data()[i], 1e-6f);
}

TEST(SoftmaxCUDA, DoesNotMutateSource) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    tensor::Softmax(gpu, -1);
    Tensor<float> back = gpu.To(Device::CPU);
    for (size_t i = 0; i < cpu.Size(); ++i) EXPECT_EQ(back.Data()[i], cpu.Data()[i]);
}

TEST(SoftmaxCUDA, AxisOutOfRangeThrows) {
    Tensor<float> gpu({2, 3}, Device::CUDA);
    EXPECT_ANY_THROW(tensor::Softmax(gpu, 2));
}

// --- softmax launcher ----------------------------------------------------------

TEST(SoftmaxLauncher, ZeroRowsIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchSoftMaxDevice<float>(nullptr, nullptr, 0, 4, 1));
}

TEST(SoftmaxLauncher, ZeroReducedIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchSoftMaxDevice<float>(nullptr, nullptr, 2, 0, 1));
}

TEST(SoftmaxLauncher, WorksInPlace) {
    auto cpu = Ramp<float>({3, 8});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    cuda::LaunchSoftMaxDevice(gpu.Data(), gpu.Data(), 3, 8, 1);
    Tensor<float> back = gpu.To(Device::CPU);
    Tensor<float> expected = tensor::Softmax(cpu, -1);
    for (size_t i = 0; i < expected.Size(); ++i) EXPECT_NEAR(back.Data()[i], expected.Data()[i], 1e-6f);
}
