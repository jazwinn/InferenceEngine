#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>

#include <cmath>
#include <limits>
#include <vector>

using cuda::DeviceBuffer;
using cuda::LaunchLinearDevice;

namespace {

template <typename T>
std::vector<T> Pattern(size_t count, int seed) {
    std::vector<T> v(count);
    for (size_t i = 0; i < count; ++i)
        v[i] = T(static_cast<int>((i * 7 + seed * 13) % 11) - 5) * T(0.25);
    return v;
}

struct View {
    size_t rs, cs;
};

struct Problem {
    std::vector<size_t> batch;
    std::vector<size_t> x_batch_stride, w_batch_stride;
    size_t rows, inner, cols;
    View x, w;
};

size_t BatchCount(const std::vector<size_t>& batch) {
    size_t n = 1;
    for (size_t d : batch) n *= d;
    return n;
}

size_t BatchOffset(size_t z, const std::vector<size_t>& batch, const std::vector<size_t>& stride) {
    size_t off = 0;
    for (size_t ax = batch.size(); ax-- > 0;) {
        off += (z % batch[ax]) * stride[ax];
        z /= batch[ax];
    }
    return off;
}

// Smallest buffer that holds every element a strided view can touch.
size_t Extent(const Problem& p, const std::vector<size_t>& batch_stride, View v, size_t r, size_t c) {
    if (r == 0 || c == 0) return 1;
    size_t last = (r - 1) * v.rs + (c - 1) * v.cs;
    for (size_t ax = 0; ax < p.batch.size(); ++ax) last += (p.batch[ax] - 1) * batch_stride[ax];
    return last + 1;
}

template <typename T>
std::vector<T> Reference(const Problem& p, const std::vector<T>& X, const std::vector<T>& W, const T* B) {
    const size_t batches = BatchCount(p.batch);
    std::vector<T> C(batches * p.rows * p.cols);
    for (size_t z = 0; z < batches; ++z) {
        const size_t xo = BatchOffset(z, p.batch, p.x_batch_stride);
        const size_t wo = BatchOffset(z, p.batch, p.w_batch_stride);
        for (size_t r = 0; r < p.rows; ++r)
            for (size_t c = 0; c < p.cols; ++c) {
                T sum = B ? B[c] : T(0);
                for (size_t n = 0; n < p.inner; ++n)
                    sum += X[xo + r * p.x.rs + n * p.x.cs] * W[wo + n * p.w.rs + c * p.w.cs];
                C[(z * p.rows + r) * p.cols + c] = sum;
            }
    }
    return C;
}

// Runs the strided launcher on a C buffer pre-filled with NaN, so any element
// the launcher fails to overwrite (or reads before writing) shows up as NaN.
template <typename T>
std::vector<T> RunDevice(const Problem& p, const std::vector<T>& X, const std::vector<T>& W, const std::vector<T>* B) {
    const size_t out = BatchCount(p.batch) * p.rows * p.cols;
    std::vector<T> C(out, std::numeric_limits<T>::quiet_NaN());

    DeviceBuffer<T> dX(X.data(), X.size()), dW(W.data(), W.size()), dC(C.data(), C.size());
    DeviceBuffer<T> dB = B ? DeviceBuffer<T>(B->data(), B->size()) : DeviceBuffer<T>();

    LaunchLinearDevice(dX.Data(), dW.Data(), B ? dB.Data() : static_cast<const T*>(nullptr), dC.Data(),
                       p.batch, p.x_batch_stride, p.w_batch_stride,
                       p.rows, p.inner, p.cols, p.x.rs, p.x.cs, p.w.rs, p.w.cs);

    dC.CopyToHost(C.data(), C.size());
    return C;
}

template <typename T>
void ExpectMatches(const Problem& p, bool with_bias, T tol, int seed = 1) {
    const auto X = Pattern<T>(Extent(p, p.x_batch_stride, p.x, p.rows, p.inner), seed);
    const auto W = Pattern<T>(Extent(p, p.w_batch_stride, p.w, p.inner, p.cols), seed + 1);
    const auto B = Pattern<T>(p.cols, seed + 2);

    const std::vector<T> actual = RunDevice(p, X, W, with_bias ? &B : nullptr);
    const std::vector<T> expected = Reference(p, X, W, with_bias ? B.data() : static_cast<const T*>(nullptr));

    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], tol) << "at index " << i;
}

// X: rows x inner row-major. W in the nn.Linear layout: out x in row-major,
// read as an inner x cols view (row stride 1, column stride inner).
Problem Dense(size_t rows, size_t inner, size_t cols) {
    return {{}, {}, {}, rows, inner, cols, {inner, 1}, {1, inner}};
}

}

// --- cuBLAS path: the bias is copied into C, then GEMM runs with beta = 1 ---

TEST(LinearStrided, DenseWithBias) {
    ExpectMatches<float>(Dense(9, 12, 20), true, 1e-4f);
}

TEST(LinearStrided, DenseWithoutBiasIgnoresStaleOutput) {
    ExpectMatches<float>(Dense(9, 12, 20), false, 1e-4f);
}

TEST(LinearStrided, SingleTokenDecodeShape) {
    ExpectMatches<float>(Dense(1, 256, 1000), true, 1e-3f);
}

TEST(LinearStrided, PrefillShape) {
    ExpectMatches<float>(Dense(70, 128, 96), true, 1e-3f);
}

TEST(LinearStrided, RowMajorWeightView) {
    Problem p = Dense(7, 11, 5);
    p.w = {5, 1};
    ExpectMatches<float>(p, true, 1e-4f);
}

TEST(LinearStrided, TransposedInputView) {
    Problem p = Dense(6, 10, 8);
    p.x = {1, 6};
    ExpectMatches<float>(p, true, 1e-4f);
}

TEST(LinearStrided, PaddedRowsUseLeadingDimension) {
    Problem p = Dense(5, 7, 4);
    p.x = {16, 1};
    p.w = {1, 9};
    ExpectMatches<float>(p, true, 1e-4f);
}

TEST(LinearStrided, WorksForDouble) {
    ExpectMatches<double>(Dense(6, 9, 7), true, 1e-12);
}

// --- fallback kernel: layouts cuBLAS cannot express, and batches ---

TEST(LinearStrided, NonUnitStridesFallBackToKernel) {
    Problem p = Dense(5, 6, 7);
    p.x = {13, 2};
    ExpectMatches<float>(p, true, 1e-4f);
}

TEST(LinearStrided, NonUnitStridesWithoutBias) {
    Problem p = Dense(5, 6, 7);
    p.w = {3, 20};
    ExpectMatches<float>(p, false, 1e-4f);
}

TEST(LinearStrided, BatchedWithSharedWeight) {
    Problem p = Dense(4, 6, 5);
    p.batch = {2, 3};
    p.x_batch_stride = {3 * 4 * 6, 4 * 6};
    p.w_batch_stride = {0, 0};
    ExpectMatches<float>(p, true, 1e-4f);
}

TEST(LinearStrided, BatchedWithPerBatchWeight) {
    Problem p = Dense(3, 5, 4);
    p.batch = {3};
    p.x_batch_stride = {3 * 5};
    p.w_batch_stride = {4 * 5};
    ExpectMatches<float>(p, false, 1e-4f);
}

// --- edge cases ---

TEST(LinearStrided, ZeroInnerGivesBias) {
    ExpectMatches<float>(Dense(3, 0, 4), true, 0.0f);
}

TEST(LinearStrided, ZeroInnerWithoutBiasGivesZeros) {
    ExpectMatches<float>(Dense(3, 0, 4), false, 0.0f);
}

TEST(LinearStrided, ZeroRowsLeavesOutputUntouched) {
    const Problem p = Dense(0, 4, 3);
    std::vector<float> sentinel(3, 7.0f);
    DeviceBuffer<float> dX(4), dW(12), dB(3), dC(sentinel.data(), sentinel.size());

    LaunchLinearDevice(dX.Data(), dW.Data(), dB.Data(), dC.Data(),
                       p.batch, p.x_batch_stride, p.w_batch_stride,
                       p.rows, p.inner, p.cols, p.x.rs, p.x.cs, p.w.rs, p.w.cs);

    std::vector<float> C(3);
    dC.CopyToHost(C.data(), C.size());
    EXPECT_EQ(C, sentinel);
}

TEST(LinearStrided, MatchesSevenArgumentOverload) {
    const size_t M = 9, in = 12, out = 20;
    const auto X = Pattern<float>(M * in, 7);
    const auto W = Pattern<float>(out * in, 8);
    const auto B = Pattern<float>(out, 9);

    DeviceBuffer<float> dX(X.data(), X.size()), dW(W.data(), W.size()), dB(B.data(), B.size());
    DeviceBuffer<float> dY1(M * out), dY2(M * out);

    LaunchLinearDevice(dX.Data(), dW.Data(), dB.Data(), dY1.Data(), M, in, out);
    LaunchLinearDevice(dX.Data(), dW.Data(), dB.Data(), dY2.Data(), {}, {}, {}, M, in, out, in, 1, 1, in);

    std::vector<float> Y1(M * out), Y2(M * out);
    dY1.CopyToHost(Y1.data(), Y1.size());
    dY2.CopyToHost(Y2.data(), Y2.size());
    EXPECT_EQ(Y1, Y2);
}
