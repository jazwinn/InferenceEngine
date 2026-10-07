#include <gtest/gtest.h>

#include <NeuralNet/Transformer/AttentionOps.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using tensor::Tensor;
using nn::transformer::MergeHeads;
using nn::transformer::NormalizeMask;
using nn::transformer::SDPA;
using nn::transformer::SplitHeads;
using nn::transformer::ValidatedDivide;

namespace {

constexpr float kTol = 1e-5f;

void IotaFill(Tensor<float> &t) {
    float next = 1.0f;
    for (size_t b = 0; b < t.Shape()[0]; ++b)
        for (size_t s = 0; s < t.Shape()[1]; ++s)
            for (size_t d = 0; d < t.Shape()[2]; ++d) t(b, s, d) = next++;
}

void WaveFill(Tensor<float> &t, double phase) {
    float *data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i)
        data[i] = static_cast<float>(0.5 * std::sin(phase + 0.37 * static_cast<double>(i)));
}

Tensor<float> HeadSlice(const Tensor<float> &x, size_t head) {
    const std::vector<size_t> &shape = x.Shape();
    Tensor<float> out({shape[0], 1, shape[2], shape[3]});
    for (size_t b = 0; b < shape[0]; ++b)
        for (size_t s = 0; s < shape[2]; ++s)
            for (size_t d = 0; d < shape[3]; ++d) out(b, 0, s, d) = x(b, head, s, d);
    return out;
}

}  // namespace

// --- ValidatedDivide --------------------------------------------------------

TEST(ValidatedDivide, DividesModelWidthByHeads) {
    EXPECT_EQ(ValidatedDivide(512, 8), 64u);
    EXPECT_EQ(ValidatedDivide(8, 1), 8u);
}

TEST(ValidatedDivide, RejectsZeroHeads) {
    EXPECT_THROW(ValidatedDivide(8, 0), std::invalid_argument);
}

TEST(ValidatedDivide, RejectsNonDivisibleModelWidth) {
    EXPECT_THROW(ValidatedDivide(100, 8), std::invalid_argument);
    EXPECT_THROW(ValidatedDivide(8, 16), std::invalid_argument);
}

// --- SplitHeads / MergeHeads ------------------------------------------------

TEST(SplitHeads, ProducesBatchHeadSequenceDim) {
    Tensor<float> x({3, 5, 8}, 1.0f);

    const Tensor<float> split = SplitHeads(x, 2, 4);

    EXPECT_EQ(split.Shape(), (std::vector<size_t>{3, 2, 5, 4}));
}

TEST(SplitHeads, EachHeadTakesAContiguousChannelRange) {
    Tensor<float> x({1, 2, 8});
    IotaFill(x);

    const Tensor<float> split = SplitHeads(x, 2, 4);

    for (size_t d = 0; d < 4; ++d) {
        EXPECT_FLOAT_EQ(split(0, 0, 0, d), static_cast<float>(1 + d)) << "head 0";
        EXPECT_FLOAT_EQ(split(0, 1, 0, d), static_cast<float>(5 + d)) << "head 1";
    }
}

TEST(SplitHeads, ResultIsContiguous) {
    Tensor<float> x({1, 4, 8}, 1.0f);

    EXPECT_TRUE(SplitHeads(x, 2, 4).IsContiguous());
}

TEST(SplitHeads, SingleHeadIsAReshape) {
    Tensor<float> x({1, 3, 8});
    IotaFill(x);

    const Tensor<float> split = SplitHeads(x, 1, 8);

    ASSERT_EQ(split.Shape(), (std::vector<size_t>{1, 1, 3, 8}));
    EXPECT_FLOAT_EQ(split(0, 0, 2, 7), x(0, 2, 7));
}

TEST(SplitHeads, HeadCountIsIndependentOfModelWidth) {
    Tensor<float> kv({1, 3, 4});
    IotaFill(kv);

    const Tensor<float> split = SplitHeads(kv, 1, 4);

    ASSERT_EQ(split.Shape(), (std::vector<size_t>{1, 1, 3, 4}));
    EXPECT_FLOAT_EQ(split(0, 0, 1, 0), 5.0f);
}

TEST(MergeHeads, InvertsSplit) {
    Tensor<float> x({2, 3, 8});
    IotaFill(x);

    const Tensor<float> back = MergeHeads(SplitHeads(x, 2, 4), 8);

    ASSERT_EQ(back.Shape(), x.Shape());
    for (size_t b = 0; b < 2; ++b)
        for (size_t s = 0; s < 3; ++s)
            for (size_t d = 0; d < 8; ++d)
                EXPECT_FLOAT_EQ(back(b, s, d), x(b, s, d))
                    << "at (" << b << ", " << s << ", " << d << ")";
}

// --- SDPA -------------------------------------------------------------------

TEST(SDPA, ReturnsPerHeadContextWithoutMerging) {
    Tensor<float> Q({2, 3, 5, 4}, 0.1f);
    Tensor<float> K({2, 3, 7, 4}, 0.2f);
    Tensor<float> V({2, 3, 7, 6}, 0.3f);

    const Tensor<float> out = SDPA(Q, K, V);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{2, 3, 5, 6}));
}

TEST(SDPA, ZeroQueriesAverageTheValues) {
    Tensor<float> Q({1, 1, 2, 4}, 0.0f);
    Tensor<float> K({1, 1, 3, 4});
    WaveFill(K, 0.0);
    Tensor<float> V({1, 1, 3, 2});
    V(0, 0, 0, 0) = 1.0f; V(0, 0, 0, 1) = -3.0f;
    V(0, 0, 1, 0) = 2.0f; V(0, 0, 1, 1) = 0.0f;
    V(0, 0, 2, 0) = 6.0f; V(0, 0, 2, 1) = 6.0f;

    const Tensor<float> out = SDPA(Q, K, V);

    for (size_t q = 0; q < 2; ++q) {
        EXPECT_NEAR(out(0, 0, q, 0), 3.0f, kTol);
        EXPECT_NEAR(out(0, 0, q, 1), 1.0f, kTol);
    }
}

TEST(SDPA, DefaultScaleIsInverseSqrtOfHeadDim) {
    Tensor<float> Q({1, 1, 1, 4}, 1.0f);
    Tensor<float> K({1, 1, 2, 4}, 0.0f);
    for (size_t d = 0; d < 4; ++d) K(0, 0, 1, d) = 1.0f;
    Tensor<float> V({1, 1, 2, 1});
    V(0, 0, 0, 0) = 0.0f;
    V(0, 0, 1, 0) = 1.0f;

    const Tensor<float> out = SDPA(Q, K, V);

    EXPECT_NEAR(out(0, 0, 0, 0), 0.8807971f, kTol);
}

TEST(SDPA, ExplicitScaleOverridesTheDefault) {
    Tensor<float> Q({1, 1, 1, 4}, 1.0f);
    Tensor<float> K({1, 1, 2, 4}, 0.0f);
    for (size_t d = 0; d < 4; ++d) K(0, 0, 1, d) = 1.0f;
    Tensor<float> V({1, 1, 2, 1});
    V(0, 0, 0, 0) = 0.0f;
    V(0, 0, 1, 0) = 1.0f;

    const Tensor<float> out = SDPA(Q, K, V, Tensor<bool>(), false, 1.0f);

    EXPECT_NEAR(out(0, 0, 0, 0), 0.9820138f, kTol);
}

TEST(SDPA, CausalFirstQuerySeesOnlyTheFirstKey) {
    Tensor<float> Q({1, 1, 3, 4});
    Tensor<float> K({1, 1, 3, 4});
    Tensor<float> V({1, 1, 3, 2});
    WaveFill(Q, 0.3);
    WaveFill(K, 1.1);
    WaveFill(V, 2.0);

    const Tensor<float> out = SDPA(Q, K, V, Tensor<bool>(), true);

    EXPECT_NEAR(out(0, 0, 0, 0), V(0, 0, 0, 0), kTol);
    EXPECT_NEAR(out(0, 0, 0, 1), V(0, 0, 0, 1), kTol);
}

TEST(SDPA, CausalIgnoresFutureKeys) {
    Tensor<float> Q({1, 1, 3, 4});
    Tensor<float> K({1, 1, 3, 4});
    Tensor<float> V({1, 1, 3, 2});
    WaveFill(Q, 0.3);
    WaveFill(K, 1.1);
    WaveFill(V, 2.0);

    const Tensor<float> before = SDPA(Q, K, V, Tensor<bool>(), true);
    for (size_t d = 0; d < 2; ++d) V(0, 0, 2, d) = 100.0f;
    const Tensor<float> after = SDPA(Q, K, V, Tensor<bool>(), true);

    for (size_t q = 0; q < 2; ++q)
        for (size_t d = 0; d < 2; ++d)
            EXPECT_FLOAT_EQ(before(0, 0, q, d), after(0, 0, q, d)) << "query " << q;
}

TEST(SDPA, SingleCausalQueryAttendsToEveryCachedKey) {
    Tensor<float> Q({1, 1, 1, 4});
    Tensor<float> K({1, 1, 4, 4});
    Tensor<float> V({1, 1, 4, 2});
    WaveFill(Q, 0.3);
    WaveFill(K, 1.1);
    WaveFill(V, 2.0);

    const Tensor<float> causal = SDPA(Q, K, V, Tensor<bool>(), true);
    const Tensor<float> open = SDPA(Q, K, V, Tensor<bool>(), false);

    for (size_t d = 0; d < 2; ++d) EXPECT_FLOAT_EQ(causal(0, 0, 0, d), open(0, 0, 0, d));
}

TEST(SDPA, MaskedKeyContributesNothing) {
    Tensor<float> Q({1, 1, 2, 4});
    Tensor<float> K({1, 1, 3, 4});
    Tensor<float> V({1, 1, 3, 2});
    WaveFill(Q, 0.3);
    WaveFill(K, 1.1);
    WaveFill(V, 2.0);
    Tensor<bool> mask({2, 3}, false);
    mask(0, 1) = true;
    mask(1, 1) = true;

    const Tensor<float> before = SDPA(Q, K, V, mask);
    for (size_t d = 0; d < 2; ++d) V(0, 0, 1, d) = 100.0f;
    const Tensor<float> after = SDPA(Q, K, V, mask);

    for (size_t q = 0; q < 2; ++q)
        for (size_t d = 0; d < 2; ++d)
            EXPECT_FLOAT_EQ(before(0, 0, q, d), after(0, 0, q, d)) << "query " << q;
}

TEST(SDPA, SharedKeyValueHeadBroadcastsAcrossQueryHeads) {
    Tensor<float> Q({1, 2, 3, 4});
    Tensor<float> K({1, 1, 3, 4});
    Tensor<float> V({1, 1, 3, 2});
    WaveFill(Q, 0.3);
    WaveFill(K, 1.1);
    WaveFill(V, 2.0);

    const Tensor<float> out = SDPA(Q, K, V, Tensor<bool>(), true);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{1, 2, 3, 2}));
    for (size_t h = 0; h < 2; ++h) {
        const Tensor<float> alone = SDPA(HeadSlice(Q, h), K, V, Tensor<bool>(), true);
        for (size_t s = 0; s < 3; ++s)
            for (size_t d = 0; d < 2; ++d)
                EXPECT_NEAR(out(0, h, s, d), alone(0, 0, s, d), kTol)
                    << "head " << h << " at (" << s << ", " << d << ")";
    }
}

TEST(SDPA, LeavesInputsUntouched) {
    Tensor<float> Q({1, 1, 2, 4});
    Tensor<float> K({1, 1, 2, 4});
    Tensor<float> V({1, 1, 2, 2});
    WaveFill(Q, 0.3);
    WaveFill(K, 1.1);
    WaveFill(V, 2.0);
    const float q0 = Q(0, 0, 0, 0);
    const float k0 = K(0, 0, 0, 0);
    const float v0 = V(0, 0, 0, 0);

    SDPA(Q, K, V, Tensor<bool>(), true);

    EXPECT_FLOAT_EQ(Q(0, 0, 0, 0), q0);
    EXPECT_FLOAT_EQ(K(0, 0, 0, 0), k0);
    EXPECT_FLOAT_EQ(V(0, 0, 0, 0), v0);
}

TEST(SDPA, FullyMaskedRowStaysFinite) {
    Tensor<float> Q({1, 1, 2, 4});
    Tensor<float> K({1, 1, 3, 4});
    Tensor<float> V({1, 1, 3, 2});
    WaveFill(Q, 0.3);
    WaveFill(K, 1.1);
    WaveFill(V, 2.0);
    Tensor<bool> mask({2, 3}, false);
    for (size_t k = 0; k < 3; ++k) mask(0, k) = true;

    const Tensor<float> out = SDPA(Q, K, V, mask);

    for (size_t d = 0; d < 2; ++d) {
        EXPECT_TRUE(std::isfinite(out(0, 0, 0, d)));
        EXPECT_TRUE(std::isfinite(out(0, 0, 1, d)));
    }
}

TEST(NormalizeMask, EmptyMaskPassesThrough) {
    EXPECT_EQ(NormalizeMask(Tensor<bool>(), 2, 4, 3, 5).Size(), 0u);
}

TEST(NormalizeMask, PromotesLowerRanksToFourDimensions) {
    EXPECT_EQ(NormalizeMask(Tensor<bool>({3, 5}, false), 2, 4, 3, 5).Shape(), (std::vector<size_t>{1, 1, 3, 5}));
    EXPECT_EQ(NormalizeMask(Tensor<bool>({2, 3, 5}, false), 2, 4, 3, 5).Shape(), (std::vector<size_t>{2, 1, 3, 5}));
    EXPECT_EQ(NormalizeMask(Tensor<bool>({2, 4, 3, 5}, false), 2, 4, 3, 5).Shape(), (std::vector<size_t>{2, 4, 3, 5}));
    EXPECT_EQ(NormalizeMask(Tensor<bool>({1, 1, 1, 5}, false), 2, 4, 3, 5).Shape(), (std::vector<size_t>{1, 1, 1, 5}));
}

TEST(NormalizeMask, KeepsMaskValuesInPlace) {
    Tensor<bool> mask({2, 3, 5}, false);
    mask(1, 2, 4) = true;

    const Tensor<bool> normalized = NormalizeMask(mask, 2, 4, 3, 5);

    EXPECT_TRUE(normalized(1, 0, 2, 4));
    EXPECT_FALSE(normalized(0, 0, 2, 4));
}

TEST(NormalizeMask, RejectsShapesThatDoNotBroadcast) {
    EXPECT_THROW(NormalizeMask(Tensor<bool>({5}, false), 2, 4, 3, 5), std::invalid_argument);
    EXPECT_THROW(NormalizeMask(Tensor<bool>({1, 1, 1, 3, 5}, false), 2, 4, 3, 5), std::invalid_argument);
    EXPECT_THROW(NormalizeMask(Tensor<bool>({3, 3, 5}, false), 2, 4, 3, 5), std::invalid_argument);
    EXPECT_THROW(NormalizeMask(Tensor<bool>({2, 2, 3, 5}, false), 2, 4, 3, 5), std::invalid_argument);
    EXPECT_THROW(NormalizeMask(Tensor<bool>({3, 4}, false), 2, 4, 3, 5), std::invalid_argument);
}
