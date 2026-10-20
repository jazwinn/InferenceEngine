#include <gtest/gtest.h>

#include <Tensor/Tensor.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using tensor::Device;
using tensor::Tensor;

namespace {

Tensor<float> Wave(const std::vector<size_t>& shape, double phase = 0.0) {
    Tensor<float> t(shape);
    for (size_t i = 0; i < t.Size(); ++i) {
        t.Data()[i] = static_cast<float>(std::sin(phase + 0.37 * static_cast<double>(i)));
    }
    return t;
}

std::vector<float> Values(const Tensor<float>& t) {
    Tensor<float> cpu = t.To(Device::CPU).Contiguous();
    return std::vector<float>(cpu.Data(), cpu.Data() + cpu.Size());
}

void ExpectNear(const Tensor<float>& actual, const Tensor<float>& expected, float tol = 1e-4f) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    const auto a = Values(actual);
    const auto e = Values(expected);
    for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], e[i], tol) << "at index " << i;
}

void ExpectMatchesCPU(const Tensor<float>& a, const Tensor<float>& b) {
    const Tensor<float> gpu = MatMul(a.To(Device::CUDA), b.To(Device::CUDA));
    EXPECT_TRUE(gpu.IsCUDA());
    ExpectNear(gpu, MatMul(a, b));
}

}

// --- batched shapes ---------------------------------------------------------

TEST(BatchedMatMulCUDA, TwoDimensionalStillMatchesCPU) {
    ExpectMatchesCPU(Wave({5, 7}), Wave({7, 3}, 1.0));
}

TEST(BatchedMatMulCUDA, ThreeDimensionalMatchesCPU) {
    ExpectMatchesCPU(Wave({4, 5, 7}), Wave({4, 7, 3}, 1.0));
}

TEST(BatchedMatMulCUDA, FourDimensionalMatchesCPU) {
    ExpectMatchesCPU(Wave({2, 3, 6, 8}), Wave({2, 3, 8, 5}, 1.0));
}

TEST(BatchedMatMulCUDA, EdgesNotMultipleOfBlockMatchCPU) {
    ExpectMatchesCPU(Wave({3, 17, 33}), Wave({3, 33, 19}, 1.0));
}

TEST(BatchedMatMulCUDA, ResultShapeIsBatchThenRowsCols) {
    const Tensor<float> out = MatMul(Wave({2, 3, 4, 5}).To(Device::CUDA), Wave({2, 3, 5, 6}).To(Device::CUDA));
    EXPECT_EQ(out.Shape(), (std::vector<size_t>{2, 3, 4, 6}));
}

// --- broadcasting -----------------------------------------------------------

TEST(BatchedMatMulCUDA, BatchedLeftWith2DRightMatchesCPU) {
    ExpectMatchesCPU(Wave({2, 2, 3}), Wave({3, 2}, 1.0));
}

TEST(BatchedMatMulCUDA, 2DLeftWithBatchedRightMatchesCPU) {
    ExpectMatchesCPU(Wave({3, 5}), Wave({7, 5, 6}, 1.0));
}

TEST(BatchedMatMulCUDA, SizeOneBatchAxisBroadcastsMatchesCPU) {
    ExpectMatchesCPU(Wave({2, 4, 3, 5, 8}), Wave({2, 4, 1, 8, 6}, 1.0));
}

TEST(BatchedMatMulCUDA, MissingLeadingBatchAxesBroadcastMatchesCPU) {
    ExpectMatchesCPU(Wave({2, 3, 4, 5}), Wave({3, 5, 6}, 1.0));
}

TEST(BatchedMatMulCUDA, BothSidesBroadcastMatchesCPU) {
    ExpectMatchesCPU(Wave({3, 1, 4, 5}), Wave({1, 2, 5, 6}, 1.0));
}

TEST(BatchedMatMulCUDA, IncompatibleBatchThrows) {
    EXPECT_THROW(MatMul(Wave({2, 3, 4}).To(Device::CUDA), Wave({3, 4, 5}).To(Device::CUDA)),
                 std::invalid_argument);
}

TEST(BatchedMatMulCUDA, InnerMismatchThrows) {
    EXPECT_THROW(MatMul(Wave({2, 3, 4}).To(Device::CUDA), Wave({2, 5, 6}).To(Device::CUDA)),
                 std::invalid_argument);
}

// --- strided operands -------------------------------------------------------

TEST(BatchedMatMulCUDA, TransposedLastTwoAxesMatchesCPU) {
    const Tensor<float> q = Wave({2, 4, 6, 8});
    const Tensor<float> k = Wave({2, 4, 9, 8}, 1.0);
    const Tensor<float> gk = k.To(Device::CUDA).Transpose(-2, -1);
    ASSERT_FALSE(gk.IsContiguous());
    ExpectNear(MatMul(q.To(Device::CUDA), gk), MatMul(q, k.Transpose(-2, -1)));
}

TEST(BatchedMatMulCUDA, TransposedLeftOperandMatchesCPU) {
    const Tensor<float> a = Wave({3, 7, 5});
    const Tensor<float> b = Wave({3, 7, 4}, 1.0);
    ExpectNear(MatMul(a.To(Device::CUDA).Transpose(-2, -1), b.To(Device::CUDA)),
               MatMul(a.Transpose(-2, -1), b));
}

TEST(BatchedMatMulCUDA, PermutedBatchAxesMatchCPU) {
    const Tensor<float> a = Wave({3, 2, 4, 5});
    const Tensor<float> b = Wave({2, 3, 5, 6}, 1.0);
    const Tensor<float> ga = a.To(Device::CUDA).Transpose(0, 1);
    ASSERT_FALSE(ga.IsContiguous());
    ExpectNear(MatMul(ga, b.To(Device::CUDA)), MatMul(a.Transpose(0, 1), b));
}

TEST(BatchedMatMulCUDA, KVCacheStyleSliceMatchesCPU) {
    const Tensor<float> cache = Wave({1, 2, 16, 8}, 1.0);
    const Tensor<float> q = Wave({1, 2, 3, 8});
    const Tensor<float> gk = cache.To(Device::CUDA).Slice(2, 0, 5);
    ASSERT_FALSE(gk.IsContiguous());
    ExpectNear(MatMul(q.To(Device::CUDA), gk.Transpose(-2, -1)),
               MatMul(q, cache.Slice(2, 0, 5).Transpose(-2, -1)));
}

TEST(BatchedMatMulCUDA, GQAShapeWithSlicedCacheMatchesCPU) {
    const Tensor<float> cache = Wave({1, 2, 16, 8}, 1.0);
    const Tensor<float> q = Wave({1, 2, 3, 4, 8});
    const Tensor<float> gk = cache.To(Device::CUDA).Slice(2, 0, 7).Unsqueeze(2);
    const Tensor<float> ck = cache.Slice(2, 0, 7).Unsqueeze(2);
    const Tensor<float> scores = MatMul(q.To(Device::CUDA), gk.Transpose(-2, -1));
    EXPECT_EQ(scores.Shape(), (std::vector<size_t>{1, 2, 3, 4, 7}));
    ExpectNear(scores, MatMul(q, ck.Transpose(-2, -1)));

    const Tensor<float> weights = Wave({1, 2, 3, 4, 7}, 2.0);
    ExpectNear(MatMul(weights.To(Device::CUDA), gk), MatMul(weights, ck));
}

TEST(BatchedMatMulCUDA, StridedOperandsAreNotModified) {
    const Tensor<float> a = Wave({2, 3, 4});
    const Tensor<float> b = Wave({2, 5, 4}, 1.0);
    const Tensor<float> ga = a.To(Device::CUDA);
    const Tensor<float> gb = b.To(Device::CUDA);
    MatMul(ga, gb.Transpose(-2, -1));
    ExpectNear(ga, a, 0.0f);
    ExpectNear(gb, b, 0.0f);
}

// --- decode-shaped ----------------------------------------------------------

TEST(BatchedMatMulCUDA, SingleQueryRowMatchesCPU) {
    const Tensor<float> q = Wave({1, 8, 1, 64});
    const Tensor<float> k = Wave({1, 8, 33, 64}, 1.0);
    ExpectNear(MatMul(q.To(Device::CUDA), k.To(Device::CUDA).Transpose(-2, -1)),
               MatMul(q, k.Transpose(-2, -1)));
}
