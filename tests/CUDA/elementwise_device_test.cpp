#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>
#include <Tensor/Tensor.h>

#include <cmath>
#include <limits>
#include <stdexcept>
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

}

// --- in-place scalar operators ------------------------------------------------

TEST(ScalarInPlaceCUDA, AddMatchesCPU) {
    auto cpu = Ramp<float>({3, 5});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu += 1.25f;
    cpu += 1.25f;
    EXPECT_TRUE(gpu.IsCUDA());
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(ScalarInPlaceCUDA, SubMatchesCPU) {
    auto cpu = Ramp<float>({3, 5});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu -= 0.75f;
    cpu -= 0.75f;
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(ScalarInPlaceCUDA, MulMatchesCPU) {
    auto cpu = Ramp<float>({3, 5});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu *= -2.5f;
    cpu *= -2.5f;
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(ScalarInPlaceCUDA, DivMatchesCPU) {
    auto cpu = Ramp<float>({3, 5});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu /= 4.0f;
    cpu /= 4.0f;
    ExpectNear(gpu, cpu, 1e-6f);
}

TEST(ScalarInPlaceCUDA, DoubleMatchesCPU) {
    auto cpu = Ramp<double>({2, 7});
    Tensor<double> gpu = cpu.To(Device::CUDA);
    gpu *= 3.0;
    gpu += 0.5;
    cpu *= 3.0;
    cpu += 0.5;
    ExpectNear(gpu, cpu, 1e-12);
}

TEST(ScalarInPlaceCUDA, ReturnsSameTensorAndStorage) {
    Tensor<float> gpu = Ramp<float>({4}).To(Device::CUDA);
    Tensor<float>& ref = (gpu += 1.0f);
    EXPECT_EQ(&ref, &gpu);
}

TEST(ScalarInPlaceCUDA, SpansManyBlocks) {
    auto cpu = Ramp<float>({10007});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu *= 2.0f;
    cpu *= 2.0f;
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(ScalarInPlaceCUDA, WritesThroughSharedStorage) {
    Tensor<float> gpu = Ramp<float>({2, 3}).To(Device::CUDA);
    Tensor<float> alias = gpu;
    alias += 10.0f;
    EXPECT_EQ(Values(gpu), Values(alias));
}

TEST(ScalarInPlaceCUDA, DivisionByZeroThrows) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    EXPECT_THROW(gpu /= 0.0f, std::invalid_argument);
}

TEST(ScalarInPlaceCUDA, ColumnSliceEveryOpMatchesCPU) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> gslice = gpu.Slice(1, 1, 2);
    Tensor<float> cslice = cpu.Slice(1, 1, 2);
    ASSERT_FALSE(gslice.IsContiguous());
    gslice += 1.0f; cslice += 1.0f;
    gslice -= 0.25f; cslice -= 0.25f;
    gslice *= 3.0f; cslice *= 3.0f;
    gslice /= 2.0f; cslice /= 2.0f;
    ExpectNear(gpu, cpu, 1e-6f);
}

TEST(ScalarInPlaceCUDA, ColumnSliceLeavesOtherColumnsUntouched) {
    Tensor<float> cpu({3, 4}, 1.0f);
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu.Slice(1, 1, 2) *= 10.0f;
    const auto v = Values(gpu);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c)
            EXPECT_EQ(v[r * 4 + c], (c == 1 || c == 2) ? 10.0f : 1.0f) << "at (" << r << "," << c << ")";
}

TEST(ScalarInPlaceCUDA, TransposedViewWritesThroughToParent) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> t = gpu.Transpose();
    ASSERT_FALSE(t.IsContiguous());
    t += 2.0f;
    cpu += 2.0f;
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(ScalarInPlaceCUDA, SliceOf3DTransposeMatchesCPU) {
    auto cpu = Ramp<float>({2, 3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu.Transpose(0, 2).Slice(0, 1, 2) *= -1.5f;
    cpu.Transpose(0, 2).Slice(0, 1, 2) *= -1.5f;
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(ScalarInPlaceCUDA, StridedIntView) {
    Tensor<int> cpu({3, 4});
    for (size_t i = 0; i < cpu.Size(); ++i) cpu.Data()[i] = static_cast<int>(i);
    Tensor<int> gpu = cpu.To(Device::CUDA);
    gpu.Slice(1, 0, 2) *= 3;
    cpu.Slice(1, 0, 2) *= 3;
    EXPECT_EQ(Values(gpu), Values(cpu));
}

TEST(ScalarInPlaceCUDA, StridedDivisionByZeroThrows) {
    Tensor<float> gpu = Ramp<float>({3, 4}).To(Device::CUDA);
    Tensor<float> slice = gpu.Slice(1, 1, 2);
    EXPECT_THROW(slice /= 0.0f, std::invalid_argument);
}

TEST(ScalarInPlaceCUDA, StridedReturnsSelf) {
    Tensor<float> gpu = Ramp<float>({3, 4}).To(Device::CUDA);
    Tensor<float> slice = gpu.Slice(1, 1, 2);
    EXPECT_EQ(&(slice += 1.0f), &slice);
}

TEST(StridedScalarLauncher, EveryOpWritesOnlyMappedElements) {
    const cuda::BinaryOp ops[] = {cuda::BinaryOp::Add, cuda::BinaryOp::Sub, cuda::BinaryOp::Mul, cuda::BinaryOp::Div};
    const float expected[] = {6.0f, -2.0f, 8.0f, 0.5f};
    for (int k = 0; k < 4; ++k) {
        Tensor<float> gpu = Tensor<float>({3, 4}, 2.0f).To(Device::CUDA);
        cuda::LaunchStridedScalarDevice(gpu.Data() + 1, 4.0f, {3, 2}, {4, 1}, ops[k]);
        const auto v = Values(gpu);
        for (size_t i = 0; i < v.size(); ++i) {
            const size_t c = i % 4;
            EXPECT_FLOAT_EQ(v[i], (c == 1 || c == 2) ? expected[k] : 2.0f) << "op " << k << " at index " << i;
        }
    }
}

TEST(StridedScalarLauncher, ZeroSizeIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchStridedScalarDevice<float>(nullptr, 1.0f, {0, 3}, {3, 1}, cuda::BinaryOp::Add));
}

TEST(StridedScalarLauncher, RankTooLargeThrows) {
    Tensor<float> gpu({1}, Device::CUDA);
    std::vector<size_t> shape(9, 1), stride(9, 1);
    EXPECT_THROW(cuda::LaunchStridedScalarDevice(gpu.Data(), 1.0f, shape, stride, cuda::BinaryOp::Add), std::invalid_argument);
}

TEST(StridedScalarLauncher, RankMismatchThrows) {
    Tensor<float> gpu({4}, Device::CUDA);
    EXPECT_THROW(cuda::LaunchStridedScalarDevice(gpu.Data(), 1.0f, {2, 2}, {1}, cuda::BinaryOp::Add), std::invalid_argument);
}

TEST(ScalarInPlaceCUDA, IntMatchesCPU) {
    Tensor<int> cpu({2, 5});
    for (size_t i = 0; i < cpu.Size(); ++i) cpu.Data()[i] = static_cast<int>(i) * 3 - 7;
    Tensor<int> gpu = cpu.To(Device::CUDA);
    gpu *= 4;
    gpu += 9;
    gpu -= 2;
    gpu /= 3;
    cpu *= 4;
    cpu += 9;
    cpu -= 2;
    cpu /= 3;
    EXPECT_TRUE(gpu.IsCUDA());
    EXPECT_EQ(Values(gpu), Values(cpu));
}

// --- out-of-place scalar operators -------------------------------------------

TEST(ScalarOperatorsCUDA, BinaryScalarOperatorsMatchCPU) {
    auto cpu = Ramp<float>({4, 6});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    ExpectNear(gpu + 2.0f, cpu + 2.0f, 0.0f);
    ExpectNear(2.0f + gpu, 2.0f + cpu, 0.0f);
    ExpectNear(gpu - 1.5f, cpu - 1.5f, 0.0f);
    ExpectNear(1.5f - gpu, 1.5f - cpu, 0.0f);
    ExpectNear(gpu * 0.5f, cpu * 0.5f, 0.0f);
    ExpectNear(0.5f * gpu, 0.5f * cpu, 0.0f);
    ExpectNear(gpu / 3.0f, cpu / 3.0f, 1e-6f);
}

TEST(ScalarOperatorsCUDA, ResultStaysOnGPUAndLeavesInputUnchanged) {
    auto cpu = Ramp<float>({2, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> scaled = gpu * 4.0f;
    EXPECT_TRUE(scaled.IsCUDA());
    EXPECT_FALSE(scaled.SharesStorage(gpu));
    EXPECT_EQ(Values(gpu), Values(cpu));
}

TEST(ScalarOperatorsCUDA, WorksOnNonContiguousInput) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    ExpectNear(gpu.Transpose() * 2.0f, cpu.Transpose() * 2.0f, 0.0f);
    ExpectNear(gpu.Slice(1, 1, 2) + 1.0f, cpu.Slice(1, 1, 2) + 1.0f, 0.0f);
}

TEST(ScalarOperatorsCUDA, DivisionByZeroThrows) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    EXPECT_THROW(gpu / 0.0f, std::invalid_argument);
}

// --- launcher ------------------------------------------------------------------

TEST(ScalarLauncher, ZeroSizeIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchScalarDevice<float>(nullptr, nullptr, 1.0f, 0, cuda::BinaryOp::Add));
}

TEST(ScalarLauncher, OutOfPlaceLeavesInputUnchanged) {
    auto cpu = Ramp<float>({8});
    Tensor<float> in = cpu.To(Device::CUDA);
    Tensor<float> out({8}, Device::CUDA);
    cuda::LaunchScalarDevice(in.Data(), out.Data(), 3.0f, in.Size(), cuda::BinaryOp::Mul);
    EXPECT_EQ(Values(in), Values(cpu));
    ExpectNear(out, cpu * 3.0f, 0.0f);
}

// --- tensor + tensor ---------------------------------------------------------------

TEST(BinaryAddCUDA, MatchesCPU) {
    auto a = Ramp<float>({4, 6});
    auto b = Ramp<float>({4, 6}) * 0.25f + 1.0f;
    Tensor<float> sum = a.To(Device::CUDA) + b.To(Device::CUDA);
    EXPECT_TRUE(sum.IsCUDA());
    EXPECT_TRUE(sum.IsContiguous());
    ExpectNear(sum, a + b, 0.0f);
}

TEST(BinaryAddCUDA, DoubleMatchesCPU) {
    auto a = Ramp<double>({3, 7});
    auto b = Ramp<double>({3, 7}) * 2.0;
    ExpectNear(a.To(Device::CUDA) + b.To(Device::CUDA), a + b, 1e-12);
}

TEST(BinaryAddCUDA, IntMatchesCPU) {
    Tensor<int> a({2, 3});
    Tensor<int> b({2, 3});
    for (size_t i = 0; i < a.Size(); ++i) {
        a.Data()[i] = static_cast<int>(i) - 3;
        b.Data()[i] = static_cast<int>(i) * 10;
    }
    EXPECT_EQ(Values(a.To(Device::CUDA) + b.To(Device::CUDA)), Values(a + b));
}

TEST(BinaryAddCUDA, SpansManyBlocks) {
    auto a = Ramp<float>({10007});
    auto b = Ramp<float>({10007}) * 3.0f;
    ExpectNear(a.To(Device::CUDA) + b.To(Device::CUDA), a + b, 0.0f);
}

TEST(BinaryAddCUDA, LeavesInputsUnchanged) {
    auto a = Ramp<float>({2, 3});
    auto b = Ramp<float>({2, 3}) * 2.0f;
    Tensor<float> ga = a.To(Device::CUDA);
    Tensor<float> gb = b.To(Device::CUDA);
    Tensor<float> sum = ga + gb;
    EXPECT_FALSE(sum.SharesStorage(ga));
    EXPECT_FALSE(sum.SharesStorage(gb));
    EXPECT_EQ(Values(ga), Values(a));
    EXPECT_EQ(Values(gb), Values(b));
}

TEST(BinaryAddCUDA, AddingTensorToItself) {
    auto a = Ramp<float>({3, 3});
    Tensor<float> ga = a.To(Device::CUDA);
    ExpectNear(ga + ga, a + a, 0.0f);
}

TEST(BinaryAddCUDA, MixedDevicesThrow) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    Tensor<float> cpu({2, 2}, 1.0f);
    EXPECT_THROW(gpu + cpu, std::invalid_argument);
    EXPECT_THROW(cpu + gpu, std::invalid_argument);
}

TEST(BinaryAddCUDA, IncompatibleShapesThrow) {
    Tensor<float> a({2, 3}, Device::CUDA);
    Tensor<float> b({4}, Device::CUDA);
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(BinaryOperatorsCUDA, SubMulDivMatchCPU) {
    auto a = Ramp<float>({4, 6});
    auto b = Ramp<float>({4, 6}) * 0.5f + 5.0f;
    Tensor<float> ga = a.To(Device::CUDA);
    Tensor<float> gb = b.To(Device::CUDA);

    Tensor<float> diff = ga - gb;
    Tensor<float> prod = ga * gb;
    Tensor<float> quot = ga / gb;

    EXPECT_TRUE(diff.IsCUDA());
    EXPECT_TRUE(prod.IsCUDA());
    EXPECT_TRUE(quot.IsCUDA());
    ExpectNear(diff, a - b, 0.0f);
    ExpectNear(prod, a * b, 0.0f);
    ExpectNear(quot, a / b, 1e-6f);
}

TEST(BinaryOperatorsCUDA, DoubleMatchesCPU) {
    auto a = Ramp<double>({3, 5});
    auto b = Ramp<double>({3, 5}) * 0.25 + 4.0;
    Tensor<double> ga = a.To(Device::CUDA);
    Tensor<double> gb = b.To(Device::CUDA);
    ExpectNear(ga - gb, a - b, 1e-12);
    ExpectNear(ga * gb, a * b, 1e-12);
    ExpectNear(ga / gb, a / b, 1e-12);
}

TEST(BinaryOperatorsCUDA, IntMatchesCPU) {
    Tensor<int> a({2, 4});
    Tensor<int> b({2, 4});
    for (size_t i = 0; i < a.Size(); ++i) {
        a.Data()[i] = static_cast<int>(i) * 7 - 20;
        b.Data()[i] = static_cast<int>(i % 3) + 1;
    }
    Tensor<int> ga = a.To(Device::CUDA);
    Tensor<int> gb = b.To(Device::CUDA);
    EXPECT_EQ(Values(ga - gb), Values(a - b));
    EXPECT_EQ(Values(ga * gb), Values(a * b));
    EXPECT_EQ(Values(ga / gb), Values(a / b));
}

TEST(BinaryOperatorsCUDA, ChainedExpressionStaysOnGPU) {
    auto a = Ramp<float>({3, 4});
    auto b = Ramp<float>({3, 4}) * 2.0f + 9.0f;
    Tensor<float> ga = a.To(Device::CUDA);
    Tensor<float> gb = b.To(Device::CUDA);
    Tensor<float> result = (ga + gb) * (ga - gb) / gb;
    EXPECT_TRUE(result.IsCUDA());
    ExpectNear(result, (a + b) * (a - b) / b, 1e-5f);
}

TEST(BinaryOperatorsCUDA, FloatDivisionByZeroGivesInfinityOnGPU) {
    Tensor<float> a({2}, 1.0f, Device::CUDA);
    Tensor<float> b({2}, Device::CUDA);
    const auto q = Values(a / b);
    EXPECT_TRUE(std::isinf(q[0]));
    EXPECT_TRUE(std::isinf(q[1]));
}

TEST(BinaryOperatorsCUDA, MixedDevicesThrowForEveryOperator) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    Tensor<float> cpu({2, 2}, 1.0f);
    EXPECT_THROW(gpu - cpu, std::invalid_argument);
    EXPECT_THROW(cpu - gpu, std::invalid_argument);
    EXPECT_THROW(gpu * cpu, std::invalid_argument);
    EXPECT_THROW(cpu * gpu, std::invalid_argument);
    EXPECT_THROW(gpu / cpu, std::invalid_argument);
    EXPECT_THROW(cpu / gpu, std::invalid_argument);
}

// --- broadcasting --------------------------------------------------------------

namespace {

void ExpectAllOpsMatchCPU(const Tensor<float>& a, const Tensor<float>& b) {
    Tensor<float> ga = a.To(Device::CUDA);
    Tensor<float> gb = b.To(Device::CUDA);
    ExpectNear(ga + gb, a + b, 0.0f);
    ExpectNear(ga - gb, a - b, 0.0f);
    ExpectNear(ga * gb, a * b, 0.0f);
    ExpectNear(ga / gb, a / b, 1e-6f);
    ExpectNear(gb + ga, b + a, 0.0f);
    ExpectNear(gb - ga, b - a, 0.0f);
}

Tensor<float> NonZero(const std::vector<size_t>& shape) {
    return Ramp<float>(shape) * 0.5f + 7.0f;
}

}

TEST(BroadcastCUDA, RowVectorAcrossMatrix) {
    ExpectAllOpsMatchCPU(Ramp<float>({2, 3}), NonZero({3}));
}

TEST(BroadcastCUDA, ColumnVectorAcrossMatrix) {
    ExpectAllOpsMatchCPU(Ramp<float>({2, 3}), NonZero({2, 1}));
}

TEST(BroadcastCUDA, BothSidesGrow) {
    Tensor<float> a = Ramp<float>({2, 1, 4});
    Tensor<float> b = NonZero({3, 1});
    Tensor<float> sum = a.To(Device::CUDA) + b.To(Device::CUDA);
    EXPECT_EQ(sum.Shape(), (std::vector<size_t>{2, 3, 4}));
    ExpectAllOpsMatchCPU(a, b);
}

TEST(BroadcastCUDA, SingleElementOperand) {
    ExpectAllOpsMatchCPU(Ramp<float>({4}), NonZero({1}));
    ExpectAllOpsMatchCPU(Ramp<float>({3, 4}), NonZero({1, 1}));
}

TEST(BroadcastCUDA, BiasAcrossSequence) {
    ExpectAllOpsMatchCPU(Ramp<float>({2, 5, 8}), NonZero({8}));
}

TEST(BroadcastCUDA, DoubleAndInt) {
    Tensor<double> a = Ramp<double>({3, 4});
    Tensor<double> b = Ramp<double>({4}) + 9.0;
    ExpectNear(a.To(Device::CUDA) * b.To(Device::CUDA), a * b, 1e-12);

    Tensor<int> ia({2, 3});
    Tensor<int> ib({3});
    for (size_t i = 0; i < ia.Size(); ++i) ia.Data()[i] = static_cast<int>(i) * 5 - 4;
    for (size_t i = 0; i < ib.Size(); ++i) ib.Data()[i] = static_cast<int>(i) + 1;
    EXPECT_EQ(Values(ia.To(Device::CUDA) / ib.To(Device::CUDA)), Values(ia / ib));
}

TEST(BroadcastCUDA, TransposedOperand) {
    Tensor<float> a = Ramp<float>({3, 2});
    Tensor<float> b = NonZero({2, 3});
    ExpectAllOpsMatchCPU(a, b.Transpose());
    ExpectNear(a.To(Device::CUDA) + b.To(Device::CUDA).Transpose(), a + b.Transpose(), 0.0f);
}

TEST(BroadcastCUDA, SlicedOperands) {
    Tensor<float> big = Ramp<float>({4, 6});
    Tensor<float> gbig = big.To(Device::CUDA);
    ExpectNear(gbig.Slice(1, 1, 3) * gbig.Slice(1, 3, 3), big.Slice(1, 1, 3) * big.Slice(1, 3, 3), 0.0f);
}

TEST(BroadcastCUDA, BothOperandsNonContiguousWithBroadcast) {
    Tensor<float> a = Ramp<float>({4, 3});
    Tensor<float> b = NonZero({1, 4});
    Tensor<float> ga = a.To(Device::CUDA);
    Tensor<float> gb = b.To(Device::CUDA);
    ExpectNear(ga.Transpose() + gb, a.Transpose() + b, 0.0f);
}

// --- in-place tensor operators ---------------------------------------------------

TEST(InPlaceTensorCUDA, BiasAddMatchesCPU) {
    Tensor<float> x = Ramp<float>({2, 5, 8});
    Tensor<float> bias = NonZero({8});
    Tensor<float> gx = x.To(Device::CUDA);
    gx += bias.To(Device::CUDA);
    x += bias;
    EXPECT_TRUE(gx.IsCUDA());
    ExpectNear(gx, x, 0.0f);
}

TEST(InPlaceTensorCUDA, EveryOperatorMatchesCPU) {
    Tensor<float> b = NonZero({3});
    Tensor<float> gb = b.To(Device::CUDA);

    Tensor<float> x = Ramp<float>({2, 3});
    Tensor<float> gx = x.To(Device::CUDA);
    gx += gb; x += b;
    gx -= gb * 0.5f; x -= b * 0.5f;
    gx *= gb; x *= b;
    gx /= gb; x /= b;
    ExpectNear(gx, x, 1e-5f);
}

TEST(InPlaceTensorCUDA, GammaScaleMatchesCPU) {
    Tensor<float> x = Ramp<float>({4, 6});
    Tensor<float> gamma = NonZero({6});
    Tensor<float> gx = x.To(Device::CUDA);
    gx *= gamma.To(Device::CUDA);
    x *= gamma;
    ExpectNear(gx, x, 0.0f);
}

TEST(InPlaceTensorCUDA, WritesIntoSliceAndLeavesOtherRowsUntouched) {
    Tensor<float> cache = Ramp<float>({2, 4, 3});
    Tensor<float> update = NonZero({2, 2, 3});
    Tensor<float> gcache = cache.To(Device::CUDA);

    Tensor<float> gslice = gcache.Slice(1, 1, 2);
    gslice += update.To(Device::CUDA);
    Tensor<float> slice = cache.Slice(1, 1, 2);
    slice += update;

    ExpectNear(gcache, cache, 0.0f);
}

TEST(InPlaceTensorCUDA, ReturnsSelf) {
    Tensor<float> gx({2, 2}, Device::CUDA);
    Tensor<float> gb({2}, Device::CUDA);
    EXPECT_EQ(&(gx += gb), &gx);
}

TEST(InPlaceTensorCUDA, ShapeGrowingThrows) {
    Tensor<float> bias({3}, Device::CUDA);
    Tensor<float> x({2, 3}, Device::CUDA);
    EXPECT_THROW(bias += x, std::invalid_argument);
}

TEST(InPlaceTensorCUDA, MixedDevicesThrow) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    Tensor<float> cpu({2, 2}, 1.0f);
    EXPECT_THROW(gpu += cpu, std::invalid_argument);
    EXPECT_THROW(cpu += gpu, std::invalid_argument);
    EXPECT_THROW(gpu *= cpu, std::invalid_argument);
    EXPECT_THROW(cpu /= gpu, std::invalid_argument);
}

TEST(InPlaceTensorCUDA, CPUInPlaceStillWorksInCUDABuild) {
    Tensor<float> x({2, 3}, 1.0f);
    Tensor<float> b({3}, 2.0f);
    x += b;
    x *= b;
    EXPECT_EQ(Values(x), std::vector<float>(6, 6.0f));
}

// --- overlapping memory ----------------------------------------------------------

TEST(OverlapCUDA, AddTransposeOfItself) {
    Tensor<float> a = Ramp<float>({5, 5});
    const Tensor<float> expected = a + a.Transpose();
    Tensor<float> ga = a.To(Device::CUDA);
    ga += ga.Transpose();
    ExpectNear(ga, expected, 0.0f);
}

TEST(OverlapCUDA, AddItselfDoubles) {
    Tensor<float> a = Ramp<float>({3, 4});
    Tensor<float> ga = a.To(Device::CUDA);
    ga += ga;
    ExpectNear(ga, a * 2.0f, 0.0f);
}

TEST(OverlapCUDA, BroadcastOwnFirstRow) {
    Tensor<float> a = Ramp<float>({4, 3});
    const Tensor<float> expected = a * a.Row(0);
    Tensor<float> ga = a.To(Device::CUDA);
    ga *= ga.Row(0);
    ExpectNear(ga, expected, 0.0f);
}

TEST(OverlapCUDA, ShiftedSliceOfSameBuffer) {
    Tensor<float> a = Ramp<float>({6, 2});
    const Tensor<float> expected = a.Slice(0, 0, 5) + a.Slice(0, 1, 5);
    Tensor<float> ga = a.To(Device::CUDA);
    Tensor<float> head = ga.Slice(0, 0, 5);
    head += ga.Slice(0, 1, 5);
    ExpectNear(head, expected, 0.0f);
}

TEST(OverlapCUDA, DisjointSlicesOfSameBuffer) {
    Tensor<float> a = Ramp<float>({4, 3});
    const Tensor<float> expected = a.Slice(0, 0, 2) - a.Slice(0, 2, 2);
    Tensor<float> ga = a.To(Device::CUDA);
    Tensor<float> top = ga.Slice(0, 0, 2);
    top -= ga.Slice(0, 2, 2);
    ExpectNear(top, expected, 0.0f);
    ExpectNear(ga.Slice(0, 2, 2), a.Slice(0, 2, 2), 0.0f);
}

// --- binary launcher -----------------------------------------------------------

TEST(BinaryLauncher, ZeroSizeIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchBinaryDevice<float>(nullptr, nullptr, nullptr, 0, cuda::BinaryOp::Add));
}

TEST(BinaryLauncher, EveryOpMatchesCPU) {
    auto a = Ramp<float>({16});
    auto b = Ramp<float>({16}) * 0.5f + 7.0f;
    Tensor<float> ga = a.To(Device::CUDA);
    Tensor<float> gb = b.To(Device::CUDA);
    Tensor<float> out({16}, Device::CUDA);

    cuda::LaunchBinaryDevice(ga.Data(), gb.Data(), out.Data(), out.Size(), cuda::BinaryOp::Add);
    ExpectNear(out, a + b, 0.0f);
    cuda::LaunchBinaryDevice(ga.Data(), gb.Data(), out.Data(), out.Size(), cuda::BinaryOp::Sub);
    ExpectNear(out, a - b, 0.0f);
    cuda::LaunchBinaryDevice(ga.Data(), gb.Data(), out.Data(), out.Size(), cuda::BinaryOp::Mul);
    ExpectNear(out, a * b, 0.0f);
    cuda::LaunchBinaryDevice(ga.Data(), gb.Data(), out.Data(), out.Size(), cuda::BinaryOp::Div);
    ExpectNear(out, a / b, 1e-6f);
}

TEST(BinaryLauncher, WorksInPlace) {
    auto a = Ramp<float>({8});
    auto b = Ramp<float>({8}) * 2.0f;
    Tensor<float> ga = a.To(Device::CUDA);
    Tensor<float> gb = b.To(Device::CUDA);
    cuda::LaunchBinaryDevice(ga.Data(), gb.Data(), ga.Data(), ga.Size(), cuda::BinaryOp::Add);
    ExpectNear(ga, a + b, 0.0f);
}

// --- sqrt ----------------------------------------------------------------------

namespace {

template <typename T>
Tensor<T> PositiveRamp(const std::vector<size_t>& shape) {
    Tensor<T> t(shape);
    T* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i) {
        data[i] = T(i % 17) * T(0.75) + T(0.25);
    }
    return t;
}

}

TEST(SqrtCUDA, MatchesCPU) {
    auto cpu = PositiveRamp<float>({3, 5});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> out = gpu.Sqrt();
    EXPECT_TRUE(out.IsCUDA());
    ExpectNear(out, cpu.Sqrt(), 1e-6f);
}

TEST(SqrtCUDA, MatchesCPUForDouble) {
    auto cpu = PositiveRamp<double>({4, 7});
    Tensor<double> gpu = cpu.To(Device::CUDA);
    ExpectNear(gpu.Sqrt(), cpu.Sqrt(), 1e-12);
}

TEST(SqrtCUDA, LargeTensorSpansManyBlocks) {
    auto cpu = PositiveRamp<float>({1000, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    ExpectNear(gpu.Sqrt(), cpu.Sqrt(), 1e-6f);
}

TEST(SqrtCUDA, DoesNotMutateOrAliasTheSource) {
    auto cpu = PositiveRamp<float>({2, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> out = gpu.Sqrt();
    EXPECT_FALSE(out.SharesStorage(gpu));
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(SqrtCUDA, HandlesANonContiguousSource) {
    auto cpu = PositiveRamp<float>({3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> t = gpu.Transpose();
    ASSERT_FALSE(t.IsContiguous());
    ExpectNear(t.Sqrt(), cpu.Transpose().Sqrt(), 1e-6f);
}

TEST(SqrtCUDA, HandlesAnOffsetSlice) {
    auto cpu = PositiveRamp<float>({4, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    ExpectNear(gpu.Slice(0, 1, 3).Sqrt(), cpu.Slice(0, 1, 3).Sqrt(), 1e-6f);
}

TEST(SqrtCUDA, NegativeInputGivesNaNLikeCPU) {
    Tensor<float> cpu({2}, -4.0f);
    const auto v = Values(cpu.To(Device::CUDA).Sqrt());
    EXPECT_TRUE(std::isnan(v[0]));
    EXPECT_TRUE(std::isnan(v[1]));
}

TEST(SqrtCUDA, FreeFormForwardsToTheMember) {
    auto cpu = PositiveRamp<float>({2, 2});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    ExpectNear(tensor::Sqrt(gpu), cpu.Sqrt(), 1e-6f);
}

// --- unary launcher ------------------------------------------------------------

TEST(UnaryLauncher, ZeroSizeIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchUnaryDevice<float>(nullptr, nullptr, 0, cuda::UnaryOp::Sqrt));
}

TEST(UnaryLauncher, SqrtWorksInPlace) {
    auto cpu = PositiveRamp<float>({8});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    cuda::LaunchUnaryDevice(gpu.Data(), gpu.Data(), gpu.Size(), cuda::UnaryOp::Sqrt);
    ExpectNear(gpu, cpu.Sqrt(), 1e-6f);
}

// --- fill ----------------------------------------------------------------------

TEST(FillCUDA, FloatMatchesCPU) {
    Tensor<float> cpu({3, 5});
    Tensor<float> gpu({3, 5}, Device::CUDA);
    cpu.Fill(2.5f);
    gpu.Fill(2.5f);
    EXPECT_TRUE(gpu.IsCUDA());
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(FillCUDA, NonZeroValueIsNotABytePattern) {
    Tensor<float> gpu({4}, Device::CUDA);
    gpu.Fill(1.0f);
    for (float v : Values(gpu)) EXPECT_EQ(v, 1.0f);
}

TEST(FillCUDA, OverwritesExistingValues) {
    Tensor<float> gpu = Ramp<float>({2, 3}).To(Device::CUDA);
    gpu.Fill(-7.0f);
    for (float v : Values(gpu)) EXPECT_EQ(v, -7.0f);
}

TEST(FillCUDA, LargeTensorSpansManyBlocks) {
    Tensor<float> gpu({1000, 3}, Device::CUDA);
    gpu.Fill(3.0f);
    const auto v = Values(gpu);
    ASSERT_EQ(v.size(), 3000u);
    for (float x : v) EXPECT_EQ(x, 3.0f);
}

TEST(FillCUDA, Double) {
    Tensor<double> gpu({2, 2}, Device::CUDA);
    gpu.Fill(0.125);
    for (double v : Values(gpu)) EXPECT_EQ(v, 0.125);
}

TEST(FillCUDA, Int) {
    Tensor<int> gpu({3, 3}, Device::CUDA);
    gpu.Fill(42);
    for (int v : Values(gpu)) EXPECT_EQ(v, 42);
}

TEST(FillCUDA, Bool) {
    Tensor<bool> gpu({2, 3}, Device::CUDA);
    gpu.Fill(true);
    Tensor<bool> cpu = gpu.To(Device::CPU);
    for (size_t i = 0; i < cpu.Size(); ++i) EXPECT_TRUE(cpu.Data()[i]);
}

TEST(FillCUDA, Int64AndSizeT) {
    Tensor<int64_t> a({3}, Device::CUDA);
    Tensor<size_t> b({3}, Device::CUDA);
    a.Fill(int64_t(-5000000000));
    b.Fill(size_t(123456789012));
    for (int64_t v : Values(a)) EXPECT_EQ(v, int64_t(-5000000000));
    for (size_t v : Values(b)) EXPECT_EQ(v, size_t(123456789012));
}

TEST(FillCUDA, ContiguousRowSliceOnlyWritesItsRows) {
    auto cpu = Ramp<float>({4, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu.Slice(0, 1, 2).Fill(9.0f);
    cpu.Slice(0, 1, 2).Fill(9.0f);
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(FillCUDA, ColumnSliceOnlyWritesItsElements) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> col = gpu.Slice(1, 1, 2);
    ASSERT_FALSE(col.IsContiguous());
    col.Fill(9.0f);
    cpu.Slice(1, 1, 2).Fill(9.0f);
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(FillCUDA, TransposedViewFillsEverything) {
    auto cpu = Ramp<float>({3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> t = gpu.Transpose();
    ASSERT_FALSE(t.IsContiguous());
    t.Fill(2.0f);
    for (float v : Values(gpu)) EXPECT_EQ(v, 2.0f);
}

TEST(FillCUDA, StridedSliceOf3DTensor) {
    auto cpu = Ramp<float>({2, 3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu.Slice(2, 1, 2).Fill(-1.0f);
    cpu.Slice(2, 1, 2).Fill(-1.0f);
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(FillCUDA, SliceOfTransposeOnlyWritesItsElements) {
    auto cpu = Ramp<float>({4, 5});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    gpu.Transpose().Slice(0, 1, 2).Fill(7.0f);
    cpu.Transpose().Slice(0, 1, 2).Fill(7.0f);
    ExpectNear(gpu, cpu, 0.0f);
}

TEST(FillCUDA, StridedIntView) {
    Tensor<int> cpu({3, 4});
    for (size_t i = 0; i < cpu.Size(); ++i) cpu.Data()[i] = static_cast<int>(i);
    Tensor<int> gpu = cpu.To(Device::CUDA);
    gpu.Slice(1, 0, 1).Fill(-5);
    cpu.Slice(1, 0, 1).Fill(-5);
    EXPECT_EQ(Values(gpu), Values(cpu));
}

TEST(FillCUDA, CPUFillStillWorksInCUDABuild) {
    Tensor<float> cpu({2, 2});
    cpu.Fill(4.0f);
    EXPECT_FALSE(cpu.IsCUDA());
    EXPECT_EQ(cpu(1, 1), 4.0f);
}

// --- fill launcher -------------------------------------------------------------

TEST(FillLauncher, ZeroSizeIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchFillDevice<float>(nullptr, 1.0f, 0));
}

TEST(FillLauncher, WritesExactlySizeElements) {
    Tensor<float> gpu({8}, Device::CUDA);
    gpu.Fill(0.0f);
    cuda::LaunchFillDevice(gpu.Data(), 5.0f, 5);
    const auto v = Values(gpu);
    for (size_t i = 0; i < 5; ++i) EXPECT_EQ(v[i], 5.0f) << "at index " << i;
    for (size_t i = 5; i < 8; ++i) EXPECT_EQ(v[i], 0.0f) << "at index " << i;
}

TEST(FillLauncher, StridedWritesOnlyMappedElements) {
    Tensor<float> gpu({3, 4}, Device::CUDA);
    cuda::LaunchStridedFillDevice(gpu.Data() + 1, 5.0f, {3, 2}, {4, 1});
    const auto v = Values(gpu);
    const std::vector<float> expected{0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0};
    EXPECT_EQ(v, expected);
}

TEST(FillLauncher, StridedZeroSizeIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchStridedFillDevice<float>(nullptr, 1.0f, {0, 3}, {3, 1}));
}

TEST(FillLauncher, StridedRankTooLargeThrows) {
    Tensor<float> gpu({1}, Device::CUDA);
    std::vector<size_t> shape(9, 1), stride(9, 1);
    EXPECT_THROW(cuda::LaunchStridedFillDevice(gpu.Data(), 1.0f, shape, stride), std::invalid_argument);
}

TEST(FillLauncher, StridedRankMismatchThrows) {
    Tensor<float> gpu({4}, Device::CUDA);
    EXPECT_THROW(cuda::LaunchStridedFillDevice(gpu.Data(), 1.0f, {2, 2}, {1}), std::invalid_argument);
}

// --- masked fill ---------------------------------------------------------------

namespace {

Tensor<bool> Checker(const std::vector<size_t>& shape, size_t period) {
    Tensor<bool> m(shape);
    for (size_t i = 0; i < m.Size(); ++i) m.Data()[i] = (i % period) == 0;
    return m;
}

Tensor<bool> Causal(size_t rows, size_t cols) {
    Tensor<bool> m({rows, cols});
    for (size_t q = 0; q < rows; ++q)
        for (size_t k = 0; k < cols; ++k) m(q, k) = k > q;
    return m;
}

void ExpectMaskedFillMatchesCPU(const Tensor<float>& cpu, const Tensor<bool>& mask, float value) {
    Tensor<float> gpu = cpu.To(Device::CUDA).MaskedFill(mask.To(Device::CUDA), value);
    Tensor<float> expected = cpu.MaskedFill(mask, value);
    ASSERT_TRUE(gpu.IsCUDA());
    ASSERT_EQ(gpu.Shape(), expected.Shape());
    const auto a = Values(gpu);
    const auto e = Values(expected);
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isinf(e[i])) EXPECT_EQ(a[i], e[i]) << "at index " << i;
        else EXPECT_FLOAT_EQ(a[i], e[i]) << "at index " << i;
    }
}

}

TEST(MaskedFillCUDA, SameShapeMatchesCPU) {
    ExpectMaskedFillMatchesCPU(Ramp<float>({3, 5}), Checker({3, 5}, 3), -1.0f);
}

TEST(MaskedFillCUDA, AllFalseChangesNothing) {
    auto cpu = Ramp<float>({2, 4});
    Tensor<bool> mask({2, 4}, false);
    EXPECT_EQ(Values(cpu.To(Device::CUDA).MaskedFill(mask.To(Device::CUDA), 9.0f)), Values(cpu));
}

TEST(MaskedFillCUDA, AllTrueReplacesEverything) {
    Tensor<bool> mask({2, 4}, true);
    for (float v : Values(Ramp<float>({2, 4}).To(Device::CUDA).MaskedFill(mask.To(Device::CUDA), 7.0f))) EXPECT_EQ(v, 7.0f);
}

TEST(MaskedFillCUDA, NegativeInfinityForAttention) {
    ExpectMaskedFillMatchesCPU(Ramp<float>({4, 4}), Causal(4, 4), -std::numeric_limits<float>::infinity());
}

TEST(MaskedFillCUDA, CausalMaskBroadcastsAcrossBatchAndHeads) {
    Tensor<bool> causal = Causal(5, 5).Unsqueeze(0).Unsqueeze(0);
    ExpectMaskedFillMatchesCPU(Ramp<float>({2, 3, 5, 5}), causal, -std::numeric_limits<float>::infinity());
}

TEST(MaskedFillCUDA, LowerRankMaskBroadcasts) {
    ExpectMaskedFillMatchesCPU(Ramp<float>({2, 3, 4}), Checker({3, 4}, 2), 0.0f);
}

TEST(MaskedFillCUDA, LengthOneAxisBroadcasts) {
    ExpectMaskedFillMatchesCPU(Ramp<float>({3, 4}), Checker({3, 1}, 2), 5.0f);
}

TEST(MaskedFillCUDA, ColumnMaskBroadcastsAlongRows) {
    ExpectMaskedFillMatchesCPU(Ramp<float>({3, 4}), Checker({1, 4}, 2), 5.0f);
}

TEST(MaskedFillCUDA, NonContiguousSourceMatchesCPU) {
    auto cpu = Ramp<float>({4, 3});
    Tensor<bool> mask = Checker({3, 4}, 3);
    Tensor<float> view = cpu.To(Device::CUDA).Transpose();
    ASSERT_FALSE(view.IsContiguous());
    Tensor<float> out = view.MaskedFill(mask.To(Device::CUDA), -1.0f);
    EXPECT_EQ(Values(out), Values(cpu.Transpose().MaskedFill(mask, -1.0f)));
}

TEST(MaskedFillCUDA, NonContiguousMaskMatchesCPU) {
    Tensor<bool> big = Checker({4, 6}, 4);
    Tensor<bool> cpu_mask = big.Slice(1, 1, 4);
    Tensor<bool> gpu_mask = big.To(Device::CUDA).Slice(1, 1, 4);
    ASSERT_FALSE(gpu_mask.IsContiguous());
    auto cpu = Ramp<float>({4, 4});
    Tensor<float> out = cpu.To(Device::CUDA).MaskedFill(gpu_mask, -2.0f);
    EXPECT_EQ(Values(out), Values(cpu.MaskedFill(cpu_mask, -2.0f)));
}

TEST(MaskedFillCUDA, SlicedCausalWindowForDecoding) {
    Tensor<bool> full = Causal(8, 8);
    Tensor<bool> cpu_window = full.Slice(0, 5, 1).Slice(1, 0, 6);
    Tensor<bool> gpu_window = full.To(Device::CUDA).Slice(0, 5, 1).Slice(1, 0, 6);
    auto scores = Ramp<float>({2, 3, 1, 6});
    const float ninf = -std::numeric_limits<float>::infinity();
    Tensor<float> out = scores.To(Device::CUDA).MaskedFill(gpu_window, ninf);
    EXPECT_EQ(Values(out), Values(scores.MaskedFill(cpu_window, ninf)));
}

TEST(MaskedFillCUDA, TransposedBroadcastMaskMatchesCPU) {
    Tensor<bool> base = Checker({4, 3}, 2);
    Tensor<bool> cpu_mask = base.Transpose();
    Tensor<bool> gpu_mask = base.To(Device::CUDA).Transpose();
    ASSERT_FALSE(gpu_mask.IsContiguous());
    auto cpu = Ramp<float>({2, 3, 4});
    Tensor<float> out = cpu.To(Device::CUDA).MaskedFill(gpu_mask, 1.5f);
    EXPECT_EQ(Values(out), Values(cpu.MaskedFill(cpu_mask, 1.5f)));
}

TEST(MaskedFillCUDA, ManyBlocks) {
    ExpectMaskedFillMatchesCPU(Ramp<float>({300, 64}), Checker({300, 64}, 7), -3.0f);
}

TEST(MaskedFillCUDA, Double) {
    Tensor<double> cpu = Ramp<double>({3, 4});
    Tensor<bool> mask = Checker({3, 4}, 2);
    EXPECT_EQ(Values(cpu.To(Device::CUDA).MaskedFill(mask.To(Device::CUDA), 0.25)), Values(cpu.MaskedFill(mask, 0.25)));
}

TEST(MaskedFillCUDA, Int) {
    Tensor<int> cpu({2, 3});
    for (size_t i = 0; i < cpu.Size(); ++i) cpu.Data()[i] = static_cast<int>(i);
    Tensor<bool> mask = Checker({2, 3}, 2);
    EXPECT_EQ(Values(cpu.To(Device::CUDA).MaskedFill(mask.To(Device::CUDA), -9)), Values(cpu.MaskedFill(mask, -9)));
}

TEST(MaskedFillCUDA, DoesNotMutateOrAliasTheSource) {
    auto cpu = Ramp<float>({2, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> out = gpu.MaskedFill(Tensor<bool>({2, 3}, true).To(Device::CUDA), 0.0f);
    EXPECT_FALSE(out.SharesStorage(gpu));
    EXPECT_EQ(Values(gpu), Values(cpu));
}

TEST(MaskedFillCUDA, MixedDevicesThrow) {
    Tensor<float> cpu({2, 2}, 1.0f);
    Tensor<bool> cpu_mask({2, 2}, true);
    Tensor<float> gpu({2, 2}, Device::CUDA);
    Tensor<bool> gpu_mask = cpu_mask.To(Device::CUDA);
    EXPECT_THROW(gpu.MaskedFill(cpu_mask, 0.0f), std::invalid_argument);
    EXPECT_THROW(cpu.MaskedFill(gpu_mask, 0.0f), std::invalid_argument);
}

TEST(MaskedFillCUDA, IncompatibleMaskThrows) {
    Tensor<float> gpu({2, 3}, Device::CUDA);
    Tensor<bool> mask = Tensor<bool>({2, 4}, true).To(Device::CUDA);
    EXPECT_THROW(gpu.MaskedFill(mask, 0.0f), std::invalid_argument);
}

// --- masked fill launchers -----------------------------------------------------

TEST(MaskedFillLauncher, ZeroSizeIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchMaskedFillDevice<float>(nullptr, nullptr, nullptr, 1.0f, 0));
    EXPECT_NO_THROW(cuda::LaunchStridedMaskedFillDevice<float>(nullptr, nullptr, nullptr, 1.0f, {0, 3}, {3, 1}));
}

TEST(MaskedFillLauncher, StridedRankMismatchThrows) {
    Tensor<float> gpu({4}, Device::CUDA);
    Tensor<bool> mask({4}, Device::CUDA);
    EXPECT_THROW(cuda::LaunchStridedMaskedFillDevice(gpu.Data(), gpu.Data(), mask.Data(), 1.0f, {2, 2}, {1}), std::invalid_argument);
}
