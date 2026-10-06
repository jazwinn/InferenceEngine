#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>

#include <stdexcept>
#include <tuple>
#include <vector>

using cuda::DeviceBuffer;
using cuda::LaunchMatMul;

namespace {

template <typename T>
std::vector<T> ReferenceMatMul(const std::vector<T>& A, const std::vector<T>& B, size_t M, size_t N, size_t K) {
    std::vector<T> C(M * K, T(0));
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < K; ++j) {
            T sum = T(0);
            for (size_t n = 0; n < N; ++n) sum += A[i * N + n] * B[n * K + j];
            C[i * K + j] = sum;
        }
    return C;
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
std::vector<T> HostMatMul(const std::vector<T>& A, const std::vector<T>& B, size_t M, size_t N, size_t K) {
    std::vector<T> C(M * K, T(-999));
    LaunchMatMul(A.data(), B.data(), C.data(), M, N, K);
    return C;
}

}

TEST(MatMulHost, KnownSmallResult) {
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {7, 8, 9, 10, 11, 12};
    std::vector<float> expected = {58, 64, 139, 154};
    EXPECT_EQ(HostMatMul(A, B, 2, 3, 2), expected);
}

TEST(MatMulHost, WorksForDouble) {
    std::vector<double> A = {1, 2, 3, 4, 5, 6};
    std::vector<double> B = {7, 8, 9, 10, 11, 12};
    std::vector<double> expected = {58, 64, 139, 154};
    EXPECT_EQ(HostMatMul(A, B, 2, 3, 2), expected);
}

TEST(MatMulHost, IdentityLeavesMatrixUnchanged) {
    const size_t n = 5;
    std::vector<float> I(n * n, 0.0f);
    for (size_t i = 0; i < n; ++i) I[i * n + i] = 1.0f;
    auto A = Pattern<float>(n * n, 1);

    EXPECT_EQ(HostMatMul(A, I, n, n, n), A);
    EXPECT_EQ(HostMatMul(I, A, n, n, n), A);
}

TEST(MatMulHost, DotProduct) {
    std::vector<float> row = {1, 2, 3, 4};
    std::vector<float> col = {5, 6, 7, 8};
    EXPECT_EQ(HostMatMul(row, col, 1, 4, 1), std::vector<float>{70});
}

TEST(MatMulHost, OuterProduct) {
    std::vector<float> col = {1, 2, 3};
    std::vector<float> row = {4, 5};
    std::vector<float> expected = {4, 5, 8, 10, 12, 15};
    EXPECT_EQ(HostMatMul(col, row, 3, 1, 2), expected);
}

TEST(MatMulHost, ZeroInnerDimensionGivesZeros) {
    std::vector<float> A, B;
    EXPECT_EQ(HostMatMul(A, B, 2, 0, 3), std::vector<float>(6, 0.0f));
}

class MatMulShapes : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

TEST_P(MatMulShapes, MatchesReferenceFloat) {
    auto [M, N, K] = GetParam();
    auto A = Pattern<float>(M * N, 1);
    auto B = Pattern<float>(N * K, 2);
    ExpectNear(HostMatMul(A, B, M, N, K), ReferenceMatMul(A, B, M, N, K), 1e-3f);
}

TEST_P(MatMulShapes, MatchesReferenceDouble) {
    auto [M, N, K] = GetParam();
    auto A = Pattern<double>(M * N, 3);
    auto B = Pattern<double>(N * K, 4);
    ExpectNear(HostMatMul(A, B, M, N, K), ReferenceMatMul(A, B, M, N, K), 1e-9);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, MatMulShapes,
    ::testing::Values(
        std::make_tuple(1, 1, 1),
        std::make_tuple(16, 16, 16),
        std::make_tuple(17, 5, 17),
        std::make_tuple(15, 33, 1),
        std::make_tuple(1, 64, 100),
        std::make_tuple(37, 19, 23),
        std::make_tuple(64, 128, 96),
        std::make_tuple(200, 3, 150)));

TEST(MatMulDevice, MatchesHostOverload) {
    const size_t M = 33, N = 20, K = 18;
    auto A = Pattern<float>(M * N, 5);
    auto B = Pattern<float>(N * K, 6);

    DeviceBuffer<float> dA(A.data(), A.size());
    DeviceBuffer<float> dB(B.data(), B.size());
    DeviceBuffer<float> dC(M * K);
    LaunchMatMul(dA, dB, dC, M, N, K);

    std::vector<float> C(M * K);
    dC.CopyToHost(C.data(), C.size());
    EXPECT_EQ(C, HostMatMul(A, B, M, N, K));
}

TEST(MatMulDevice, ChainedMultipliesStayOnDevice) {
    const size_t M = 9, N = 7, K = 11, P = 4;
    auto A = Pattern<float>(M * N, 1);
    auto B = Pattern<float>(N * K, 2);
    auto C = Pattern<float>(K * P, 3);

    DeviceBuffer<float> dA(A.data(), A.size());
    DeviceBuffer<float> dB(B.data(), B.size());
    DeviceBuffer<float> dC(C.data(), C.size());
    DeviceBuffer<float> dAB(M * K);
    DeviceBuffer<float> dABC(M * P);

    LaunchMatMul(dA, dB, dAB, M, N, K);
    LaunchMatMul(dAB, dC, dABC, M, K, P);

    std::vector<float> out(M * P);
    dABC.CopyToHost(out.data(), out.size());

    auto expected = ReferenceMatMul(ReferenceMatMul(A, B, M, N, K), C, M, K, P);
    ExpectNear(out, expected, 1e-3f);
}

TEST(MatMulDevice, OversizedBuffersAreAccepted) {
    std::vector<float> A = {1, 2, 3, 4, 5, 6, 0, 0};
    std::vector<float> B = {7, 8, 9, 10, 11, 12, 0, 0};
    DeviceBuffer<float> dA(A.data(), A.size());
    DeviceBuffer<float> dB(B.data(), B.size());
    DeviceBuffer<float> dC(10);

    LaunchMatMul(dA, dB, dC, 2, 3, 2);

    std::vector<float> C(4);
    dC.CopyToHost(C.data(), C.size());
    EXPECT_EQ(C, (std::vector<float>{58, 64, 139, 154}));
}

TEST(MatMulDevice, ThrowsWhenABufferTooSmall) {
    DeviceBuffer<float> dA(5), dB(6), dC(4);
    EXPECT_THROW(LaunchMatMul(dA, dB, dC, 2, 3, 2), std::invalid_argument);
}

TEST(MatMulDevice, ThrowsWhenBBufferTooSmall) {
    DeviceBuffer<float> dA(6), dB(5), dC(4);
    EXPECT_THROW(LaunchMatMul(dA, dB, dC, 2, 3, 2), std::invalid_argument);
}

TEST(MatMulDevice, ThrowsWhenCBufferTooSmall) {
    DeviceBuffer<float> dA(6), dB(6), dC(3);
    EXPECT_THROW(LaunchMatMul(dA, dB, dC, 2, 3, 2), std::invalid_argument);
}

TEST(MatMulDevice, ThrowDoesNotTouchOutput) {
    std::vector<float> sentinel(4, 42.0f);
    DeviceBuffer<float> dA(6), dB(5), dC(sentinel.data(), sentinel.size());

    EXPECT_THROW(LaunchMatMul(dA, dB, dC, 2, 3, 2), std::invalid_argument);

    std::vector<float> out(4);
    dC.CopyToHost(out.data(), out.size());
    EXPECT_EQ(out, sentinel);
}
