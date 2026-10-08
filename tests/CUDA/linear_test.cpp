#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>

#include <stdexcept>
#include <tuple>
#include <vector>

using cuda::DeviceBuffer;
using cuda::LaunchLinear;
using cuda::LaunchLinearDevice;
using cuda::LaunchLinearHost;

namespace {

template <typename T>
std::vector<T> ReferenceLinear(const std::vector<T>& X, const std::vector<T>& W, const T* B,
                               size_t M, size_t in, size_t out) {
    std::vector<T> Y(M * out);
    for (size_t r = 0; r < M; ++r)
        for (size_t c = 0; c < out; ++c) {
            T sum = B ? B[c] : T(0);
            for (size_t k = 0; k < in; ++k) sum += X[r * in + k] * W[c * in + k];
            Y[r * out + c] = sum;
        }
    return Y;
}

template <typename T>
std::vector<T> Pattern(size_t count, int seed) {
    std::vector<T> v(count);
    for (size_t i = 0; i < count; ++i)
        v[i] = T(static_cast<int>((i * 7 + seed * 13) % 11) - 5) * T(0.25);
    return v;
}

template <typename T>
void ExpectNear(const std::vector<T>& actual, const std::vector<T>& expected, T tol) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], tol) << "at index " << i;
}

template <typename T>
std::vector<T> HostLinear(const std::vector<T>& X, const std::vector<T>& W, const T* B,
                          size_t M, size_t in, size_t out) {
    std::vector<T> Y(M * out, T(-999));
    LaunchLinearHost(X.data(), W.data(), B, Y.data(), M, in, out);
    return Y;
}

}

// --- host launcher ----------------------------------------------------------

TEST(LinearHost, KnownSmallResult) {
    std::vector<float> X = {1, 2, 3, 4, 5, 6};
    std::vector<float> W = {1, 0, 0, 0, 1, 1};
    std::vector<float> B = {10, 20};
    std::vector<float> expected = {11, 25, 14, 31};
    EXPECT_EQ(HostLinear(X, W, B.data(), 2, 3, 2), expected);
}

TEST(LinearHost, BiasIsSharedAcrossRows) {
    std::vector<float> X(4 * 2, 0.0f);
    std::vector<float> W(3 * 2, 0.0f);
    std::vector<float> B = {1, 2, 3};
    std::vector<float> expected = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    EXPECT_EQ(HostLinear(X, W, B.data(), 4, 2, 3), expected);
}

TEST(LinearHost, NullBiasMeansNoBias) {
    std::vector<float> X = {1, 2, 3, 4};
    std::vector<float> W = {1, 1, 2, -1};
    std::vector<float> expected = {3, 0, 7, 2};
    EXPECT_EQ(HostLinear(X, W, static_cast<const float*>(nullptr), 2, 2, 2), expected);
}

TEST(LinearHost, SingleRow) {
    auto X = Pattern<float>(16, 1);
    auto W = Pattern<float>(5 * 16, 2);
    auto B = Pattern<float>(5, 3);
    ExpectNear(HostLinear(X, W, B.data(), 1, 16, 5), ReferenceLinear(X, W, B.data(), 1, 16, 5), 1e-5f);
}

TEST(LinearHost, WorksForDouble) {
    auto X = Pattern<double>(6 * 9, 4);
    auto W = Pattern<double>(7 * 9, 5);
    auto B = Pattern<double>(7, 6);
    ExpectNear(HostLinear(X, W, B.data(), 6, 9, 7), ReferenceLinear(X, W, B.data(), 6, 9, 7), 1e-12);
}

class LinearShapes : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

TEST_P(LinearShapes, MatchesReferenceWithBias) {
    auto [M, in, out] = GetParam();
    auto X = Pattern<float>(M * in, 1);
    auto W = Pattern<float>(out * in, 2);
    auto B = Pattern<float>(out, 3);
    ExpectNear(HostLinear(X, W, B.data(), M, in, out), ReferenceLinear(X, W, B.data(), M, in, out), 1e-3f);
}

TEST_P(LinearShapes, MatchesReferenceWithoutBias) {
    auto [M, in, out] = GetParam();
    auto X = Pattern<float>(M * in, 4);
    auto W = Pattern<float>(out * in, 5);
    const float* none = nullptr;
    ExpectNear(HostLinear(X, W, none, M, in, out), ReferenceLinear(X, W, none, M, in, out), 1e-3f);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, LinearShapes,
    ::testing::Values(
        std::make_tuple(1, 1, 1),
        std::make_tuple(16, 16, 16),
        std::make_tuple(17, 5, 17),
        std::make_tuple(3, 64, 1),
        std::make_tuple(37, 19, 23),
        std::make_tuple(64, 128, 96),
        std::make_tuple(1, 256, 300)));

// --- device launchers -------------------------------------------------------

TEST(LinearDevice, RawPointerLauncherMatchesHost) {
    const size_t M = 9, in = 12, out = 20;
    auto X = Pattern<float>(M * in, 7);
    auto W = Pattern<float>(out * in, 8);
    auto B = Pattern<float>(out, 9);

    DeviceBuffer<float> dX(X.data(), X.size()), dW(W.data(), W.size()), dB(B.data(), B.size()), dY(M * out);
    LaunchLinearDevice(dX.Data(), dW.Data(), dB.Data(), dY.Data(), M, in, out);

    std::vector<float> Y(M * out);
    dY.CopyToHost(Y.data(), Y.size());
    EXPECT_EQ(Y, HostLinear(X, W, B.data(), M, in, out));
}

TEST(LinearDevice, BufferOverloadMatchesHost) {
    const size_t M = 5, in = 8, out = 6;
    auto X = Pattern<float>(M * in, 1);
    auto W = Pattern<float>(out * in, 2);
    auto B = Pattern<float>(out, 3);

    DeviceBuffer<float> dX(X.data(), X.size()), dW(W.data(), W.size()), dB(B.data(), B.size()), dY(M * out);
    LaunchLinear(dX, dW, dB, dY, M, in, out);

    std::vector<float> Y(M * out);
    dY.CopyToHost(Y.data(), Y.size());
    EXPECT_EQ(Y, HostLinear(X, W, B.data(), M, in, out));
}

TEST(LinearDevice, EmptyBiasBufferMeansNoBias) {
    const size_t M = 3, in = 4, out = 2;
    auto X = Pattern<float>(M * in, 1);
    auto W = Pattern<float>(out * in, 2);

    DeviceBuffer<float> dX(X.data(), X.size()), dW(W.data(), W.size()), dB, dY(M * out);
    LaunchLinear(dX, dW, dB, dY, M, in, out);

    std::vector<float> Y(M * out);
    dY.CopyToHost(Y.data(), Y.size());
    const float* none = nullptr;
    EXPECT_EQ(Y, HostLinear(X, W, none, M, in, out));
}

TEST(LinearDevice, ThrowsWhenInputBufferTooSmall) {
    DeviceBuffer<float> dX(5), dW(6), dB(2), dY(4);
    EXPECT_THROW(LaunchLinear(dX, dW, dB, dY, 2, 3, 2), std::invalid_argument);
}

TEST(LinearDevice, ThrowsWhenWeightBufferTooSmall) {
    DeviceBuffer<float> dX(6), dW(5), dB(2), dY(4);
    EXPECT_THROW(LaunchLinear(dX, dW, dB, dY, 2, 3, 2), std::invalid_argument);
}

TEST(LinearDevice, ThrowsWhenBiasBufferTooSmall) {
    DeviceBuffer<float> dX(6), dW(6), dB(1), dY(4);
    EXPECT_THROW(LaunchLinear(dX, dW, dB, dY, 2, 3, 2), std::invalid_argument);
}

TEST(LinearDevice, ThrowsWhenOutputBufferTooSmall) {
    DeviceBuffer<float> dX(6), dW(6), dB(2), dY(3);
    EXPECT_THROW(LaunchLinear(dX, dW, dB, dY, 2, 3, 2), std::invalid_argument);
}
