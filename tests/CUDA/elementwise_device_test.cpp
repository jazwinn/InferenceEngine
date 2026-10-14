#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>
#include <Tensor/Tensor.h>

#include <cmath>
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

TEST(ScalarInPlaceCUDA, NonContiguousViewThrows) {
    Tensor<float> gpu = Ramp<float>({3, 4}).To(Device::CUDA);
    Tensor<float> slice = gpu.Slice(1, 1, 2);
    EXPECT_THROW(slice += 1.0f, std::logic_error);
    EXPECT_THROW(slice -= 1.0f, std::logic_error);
    EXPECT_THROW(slice *= 2.0f, std::logic_error);
    EXPECT_THROW(slice /= 2.0f, std::logic_error);
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
