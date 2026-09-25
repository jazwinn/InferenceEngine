#include <gtest/gtest.h>

#include <NeuralNet/Module.h>
#include <NeuralNet/Transformer/MultiHeadAttention.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using math::Tensor;
using nn::Module;
using transformer::MultiHeadAttention;
using transformer::SplitHeads;

namespace {

constexpr float kTol = 1e-4f;

// Fills a rank-3 {B, S, D} tensor with 1, 2, 3, ... in row-major order.
template <typename T>
void IotaFill(Tensor<T>& t) {
    T next = T(1);
    for (size_t b = 0; b < t.Shape()[0]; ++b)
        for (size_t s = 0; s < t.Shape()[1]; ++s)
            for (size_t d = 0; d < t.Shape()[2]; ++d) t(b, s, d) = next++;
}

// Linear zero-initialises, so an untouched attention outputs all zeros and
// most behavioural assertions would hold vacuously. Seed the projections with
// a deterministic, varied pattern so the tests stay repeatable.
template <typename T>
void SeedProjections(MultiHeadAttention<T>& attn) {
    int k = 0;
    for (const auto& linear : {attn.query, attn.key, attn.value, attn.output_proj}) {
        T* w = linear->weight->Data();
        for (size_t i = 0; i < linear->weight->Size(); ++i) {
            w[i] = static_cast<T>(std::sin(0.7 * static_cast<double>(++k)));
        }
    }
}

// Small, varied values. Large inputs drive the scores far apart, and softmax
// then underflows every weight but the largest to exactly zero -- correct, but
// useless for testing that a position is attended at all.
template <typename T>
void GentleFill(Tensor<T>& t) {
    int k = 0;
    for (size_t b = 0; b < t.Shape()[0]; ++b)
        for (size_t s = 0; s < t.Shape()[1]; ++s)
            for (size_t d = 0; d < t.Shape()[2]; ++d)
                t(b, s, d) = static_cast<T>(0.1 * std::sin(0.9 * static_cast<double>(++k)));
}

// Recomputes the attention weights the layer produces, so tests can inspect
// them without the layer exposing them.
template <typename T>
Tensor<T> WeightsOf(MultiHeadAttention<T>& attn, const Tensor<T>& x, bool causal) {
    Tensor<T> q = SplitHeads(attn.query->Forward(x), attn.num_heads, attn.head_dim);
    Tensor<T> k = SplitHeads(attn.key->Forward(x), attn.num_heads, attn.head_dim);
    Tensor<T> scores = math::MatMul(q, k.Transpose(-2, -1)) * attn.scale;

    if (causal) {
        const size_t seq = scores.Shape()[scores.Rank() - 1];
        Tensor<bool> mask({seq, seq}, false);
        for (size_t i = 0; i < seq; ++i)
            for (size_t j = i + 1; j < seq; ++j) mask(i, j) = true;
        scores = scores.MaskedFill(mask, -std::numeric_limits<T>::infinity());
    }
    return activation::Softmax<T>(-1).Forward(scores);
}


// Like GentleFill, but the phase lets separate query, key and value tensors
// hold different values.
template <typename T>
void PhaseFill(Tensor<T>& t, double phase) {
    int k = 0;
    for (size_t b = 0; b < t.Shape()[0]; ++b)
        for (size_t s = 0; s < t.Shape()[1]; ++s)
            for (size_t d = 0; d < t.Shape()[2]; ++d)
                t(b, s, d) = static_cast<T>(0.1 * std::sin(0.9 * static_cast<double>(++k) + phase));
}

// Copies sequence positions [begin, end) of a {B, S, D} tensor.
template <typename T>
Tensor<T> Positions(const Tensor<T>& t, size_t begin, size_t end) {
    Tensor<T> out({t.Shape()[0], end - begin, t.Shape()[2]});
    for (size_t b = 0; b < t.Shape()[0]; ++b)
        for (size_t s = begin; s < end; ++s)
            for (size_t d = 0; d < t.Shape()[2]; ++d) out(b, s - begin, d) = t(b, s, d);
    return out;
}

// Largest element-wise difference at sequence position s of two {B, S, D} tensors.
template <typename T>
double PositionDiff(const Tensor<T>& a, const Tensor<T>& b, size_t s) {
    double worst = 0.0;
    for (size_t n = 0; n < a.Shape()[0]; ++n)
        for (size_t d = 0; d < a.Shape()[2]; ++d)
            worst = std::max(worst, static_cast<double>(std::fabs(a(n, s, d) - b(n, s, d))));
    return worst;
}

// Largest element-wise difference anywhere in two {B, S, D} tensors.
template <typename T>
double MaxAbsDiff(const Tensor<T>& a, const Tensor<T>& b) {
    double worst = 0.0;
    for (size_t s = 0; s < a.Shape()[1]; ++s) worst = std::max(worst, PositionDiff(a, b, s));
    return worst;
}

}  // namespace

// --- construction -----------------------------------------------------------

TEST(MultiHeadAttentionConstruction, DerivesHeadDimension) {
    MultiHeadAttention<float> attn(512, 8);

    EXPECT_EQ(attn.d_model, 512u);
    EXPECT_EQ(attn.num_heads, 8u);
    EXPECT_EQ(attn.head_dim, 64u);
}

TEST(MultiHeadAttentionConstruction, ScaleIsInverseSqrtOfHeadDimension) {
    MultiHeadAttention<float> attn(512, 8);
    EXPECT_NEAR(attn.scale, 1.0f / std::sqrt(64.0f), kTol);
}

TEST(MultiHeadAttentionConstruction, RejectsNonDivisibleModelSize) {
    EXPECT_THROW(MultiHeadAttention<float>(100, 8), std::invalid_argument);
    EXPECT_THROW(MultiHeadAttention<float>(8, 16), std::invalid_argument);
}

TEST(MultiHeadAttentionConstruction, RejectsZeroHeads) {
    EXPECT_THROW(MultiHeadAttention<float>(8, 0), std::invalid_argument);
}

TEST(MultiHeadAttentionConstruction, SingleHeadIsAllowed) {
    MultiHeadAttention<float> attn(8, 1);
    EXPECT_EQ(attn.head_dim, 8u);
}

TEST(MultiHeadAttentionConstruction, AllFourProjectionsAreSquare) {
    MultiHeadAttention<float> attn(8, 2);

    for (const auto& proj : {attn.query, attn.key, attn.value, attn.output_proj}) {
        ASSERT_TRUE(proj);
        EXPECT_EQ(proj->weight->Shape(), (std::vector<size_t>{8, 8}));
    }
}

TEST(MultiHeadAttentionConstruction, RegistersEightParameters) {
    MultiHeadAttention<float> attn(8, 2);
    // four Linear layers, weight + bias each
    EXPECT_EQ(attn.Parameters().size(), 8u);
}

TEST(MultiHeadAttentionConstruction, WithoutBiasRegistersFour) {
    MultiHeadAttention<float> attn(8, 2, false);
    EXPECT_EQ(attn.Parameters().size(), 4u);
}

// --- forward ----------------------------------------------------------------

TEST(MultiHeadAttentionForward, PreservesInputShape) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> x({3, 5, 8}, 1.0f);

    EXPECT_EQ(attn.Forward(x).Shape(), (std::vector<size_t>{3, 5, 8}));
}

TEST(MultiHeadAttentionForward, CallOperatorMatchesForward) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 4, 8});
    IotaFill(x);

    const Tensor<float> viaCall = attn(x);
    const Tensor<float> viaForward = attn.Forward(x);

    for (size_t s = 0; s < 4; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_FLOAT_EQ(viaCall(0, s, d), viaForward(0, s, d));
}

TEST(MultiHeadAttentionForward, DoesNotMutateOrAliasTheInput) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> x({1, 4, 8});
    IotaFill(x);
    const Tensor<float> before = x.Clone();

    const Tensor<float> out = attn.Forward(x);

    EXPECT_FALSE(out.SharesStorage(x));
    for (size_t s = 0; s < 4; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_FLOAT_EQ(x(0, s, d), before(0, s, d));
}

TEST(MultiHeadAttentionForward, IsRepeatable) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 4, 8});
    IotaFill(x);

    const Tensor<float> first = attn.Forward(x);
    const Tensor<float> second = attn.Forward(x);

    for (size_t s = 0; s < 4; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_FLOAT_EQ(first(0, s, d), second(0, s, d));
}

TEST(MultiHeadAttentionForward, BatchesAreIndependent) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);

    Tensor<float> one({1, 3, 8});
    IotaFill(one);

    Tensor<float> two({2, 3, 8});
    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 8; ++d) {
            two(0, s, d) = one(0, s, d);
            two(1, s, d) = 100.0f;     // a wildly different second batch
        }

    const Tensor<float> single = attn.Forward(one);
    const Tensor<float> batched = attn.Forward(two);

    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_NEAR(batched(0, s, d), single(0, s, d), 1e-3f)
                << "batch 1 must not affect batch 0 at (" << s << ", " << d << ")";
}

TEST(MultiHeadAttentionForward, RejectsAMismatchedModelDimension) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> wrong({1, 4, 6}, 1.0f);

    EXPECT_THROW(attn.Forward(wrong), std::invalid_argument);
}

TEST(MultiHeadAttentionForward, RunsWithoutBias) {
    MultiHeadAttention<float> attn(8, 2, false);
    SeedProjections(attn);

    ASSERT_FALSE(attn.query->bias);
    ASSERT_FALSE(attn.output_proj->bias);

    Tensor<float> x({2, 4, 8});
    GentleFill(x);

    const Tensor<float> out = attn.Forward(x);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{2, 4, 8}));
    for (size_t b = 0; b < 2; ++b)
        for (size_t s = 0; s < 4; ++s)
            for (size_t d = 0; d < 8; ++d)
                EXPECT_FALSE(std::isnan(out(b, s, d)))
                    << "at (" << b << ", " << s << ", " << d << ")";
}

TEST(MultiHeadAttentionForward, WithoutBiasIsCausalToo) {
    MultiHeadAttention<float> attn(8, 2, false);
    SeedProjections(attn);

    Tensor<float> a({1, 4, 8});
    GentleFill(a);
    Tensor<float> b = a.Clone();
    for (size_t d = 0; d < 8; ++d) b(0, 3, d) = -5.0f;

    const Tensor<float> out_a = attn.Forward(a, nullptr, 0, Tensor<bool>(), true);
    const Tensor<float> out_b = attn.Forward(b, nullptr, 0, Tensor<bool>(), true);

    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_NEAR(out_a(0, s, d), out_b(0, s, d), 1e-4f)
                << "position " << s << " leaked information from position 3";
}

TEST(MultiHeadAttentionForward, BiasChangesTheResult) {
    // With identical weights, the bias-free layer must differ from the biased
    // one -- otherwise the bias is being silently dropped or silently added.
    MultiHeadAttention<float> with_bias(8, 2, true);
    MultiHeadAttention<float> without_bias(8, 2, false);
    SeedProjections(with_bias);
    SeedProjections(without_bias);
    with_bias.query->bias->Fill(0.5f);

    Tensor<float> x({1, 3, 8});
    GentleFill(x);

    const Tensor<float> biased = with_bias.Forward(x);
    const Tensor<float> plain = without_bias.Forward(x);

    bool differs = false;
    for (size_t s = 0; s < 3 && !differs; ++s)
        for (size_t d = 0; d < 8; ++d)
            if (std::fabs(biased(0, s, d) - plain(0, s, d)) > 1e-5f) { differs = true; break; }

    EXPECT_TRUE(differs) << "a non-zero bias must affect the output";
}

// --- attention weights ------------------------------------------------------

TEST(MultiHeadAttentionWeights, EachQueryFormsADistribution) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 4, 8});
    IotaFill(x);

    const Tensor<float> w = WeightsOf(attn, x, false);

    ASSERT_EQ(w.Shape(), (std::vector<size_t>{1, 2, 4, 4}));
    for (size_t h = 0; h < 2; ++h) {
        for (size_t i = 0; i < 4; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < 4; ++j) {
                EXPECT_GE(w(0, h, i, j), 0.0f);
                sum += w(0, h, i, j);
            }
            EXPECT_NEAR(sum, 1.0f, kTol) << "head " << h << " query " << i;
        }
    }
}

TEST(MultiHeadAttentionWeights, WithoutAMaskEveryPositionIsVisible) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 4, 8});
    GentleFill(x);

    const Tensor<float> w = WeightsOf(attn, x, false);

    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_GT(w(0, 0, i, j), 0.0f) << "(" << i << ", " << j << ")";
}

// --- causal masking ---------------------------------------------------------

TEST(MultiHeadAttentionCausal, FuturePositionsGetExactlyZeroWeight) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 5, 8});
    IotaFill(x);

    const Tensor<float> w = WeightsOf(attn, x, true);

    for (size_t h = 0; h < 2; ++h)
        for (size_t i = 0; i < 5; ++i)
            for (size_t j = i + 1; j < 5; ++j)
                EXPECT_FLOAT_EQ(w(0, h, i, j), 0.0f)
                    << "head " << h << ": query " << i << " saw future key " << j;
}

TEST(MultiHeadAttentionCausal, PastAndPresentKeepPositiveWeight) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 5, 8});
    GentleFill(x);

    const Tensor<float> w = WeightsOf(attn, x, true);

    for (size_t h = 0; h < 2; ++h)
        for (size_t i = 0; i < 5; ++i)
            for (size_t j = 0; j <= i; ++j)
                EXPECT_GT(w(0, h, i, j), 0.0f)
                    << "head " << h << ": query " << i << " lost key " << j;
}

TEST(MultiHeadAttentionCausal, RowsStillSumToOne) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 5, 8});
    IotaFill(x);

    const Tensor<float> w = WeightsOf(attn, x, true);

    for (size_t h = 0; h < 2; ++h) {
        for (size_t i = 0; i < 5; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < 5; ++j) sum += w(0, h, i, j);
            EXPECT_NEAR(sum, 1.0f, kTol) << "head " << h << " query " << i;
        }
    }
}

TEST(MultiHeadAttentionCausal, TheFirstQueryAttendsOnlyToItself) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> x({1, 4, 8});
    IotaFill(x);

    const Tensor<float> w = WeightsOf(attn, x, true);

    EXPECT_NEAR(w(0, 0, 0, 0), 1.0f, kTol);
}

TEST(MultiHeadAttentionCausal, NoNaNOnAnyRow) {
    // The inverted mask produced an all -inf row, which softmaxes to NaN.
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 6, 8});
    IotaFill(x);

    const Tensor<float> w = WeightsOf(attn, x, true);
    const Tensor<float> out = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);

    for (size_t h = 0; h < 2; ++h)
        for (size_t i = 0; i < 6; ++i)
            for (size_t j = 0; j < 6; ++j)
                ASSERT_FALSE(std::isnan(w(0, h, i, j)))
                    << "NaN weight at head " << h << " (" << i << ", " << j << ")";

    for (size_t s = 0; s < 6; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_FALSE(std::isnan(out(0, s, d)));
}

TEST(MultiHeadAttentionCausal, EarlierOutputsIgnoreLaterInput) {
    // The defining property: changing token 3 must not move token 0's output.
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);

    Tensor<float> a({1, 4, 8});
    IotaFill(a);
    Tensor<float> b = a.Clone();
    for (size_t d = 0; d < 8; ++d) b(0, 3, d) = -50.0f;

    const Tensor<float> out_a = attn.Forward(a, nullptr, 0, Tensor<bool>(), true);
    const Tensor<float> out_b = attn.Forward(b, nullptr, 0, Tensor<bool>(), true);

    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_NEAR(out_a(0, s, d), out_b(0, s, d), 1e-3f)
                << "position " << s << " leaked information from position 3";
}

TEST(MultiHeadAttentionCausal, ChangesTheOutputRelativeToUnmasked) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 4, 8});
    IotaFill(x);

    const Tensor<float> plain = attn.Forward(x);
    const Tensor<float> causal = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);

    bool differs = false;
    for (size_t s = 0; s < 4 && !differs; ++s)
        for (size_t d = 0; d < 8; ++d)
            if (std::fabs(plain(0, s, d) - causal(0, s, d)) > 1e-4f) { differs = true; break; }

    EXPECT_TRUE(differs) << "masking must change the result";
}

// --- explicit masks ---------------------------------------------------------

TEST(MultiHeadAttentionMask, AMaskBroadcastsOverBatchAndHeads) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> x({2, 4, 8});
    IotaFill(x);

    Tensor<bool> mask({4, 4}, false);
    for (size_t i = 0; i < 4; ++i) mask(i, 3) = true;   // block the last key

    EXPECT_NO_THROW(attn.Forward(x, nullptr, 0, mask));
}

TEST(MultiHeadAttentionMask, MaskedKeysGetZeroWeight) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 4, 8});
    IotaFill(x);

    Tensor<float> q = SplitHeads(attn.query->Forward(x), attn.num_heads, attn.head_dim);
    Tensor<float> k = SplitHeads(attn.key->Forward(x), attn.num_heads, attn.head_dim);
    Tensor<float> scores = math::MatMul(q, k.Transpose(-2, -1)) * attn.scale;

    Tensor<bool> mask({4, 4}, false);
    for (size_t i = 0; i < 4; ++i) mask(i, 1) = true;

    const Tensor<float> w = activation::Softmax<float>(-1).Forward(
        scores.MaskedFill(mask, -std::numeric_limits<float>::infinity()));

    for (size_t h = 0; h < 2; ++h)
        for (size_t i = 0; i < 4; ++i)
            EXPECT_FLOAT_EQ(w(0, h, i, 1), 0.0f) << "head " << h << " query " << i;
}

TEST(MultiHeadAttentionMask, AnEmptyMaskIsIgnored) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 4, 8});
    IotaFill(x);

    const Tensor<float> withEmpty = attn.Forward(x, nullptr, 0, Tensor<bool>());
    const Tensor<float> without = attn.Forward(x);

    for (size_t s = 0; s < 4; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_FLOAT_EQ(withEmpty(0, s, d), without(0, s, d));
}

TEST(MultiHeadAttentionMask, RankThreeMaskAppliesPerBatchNotPerHead) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({2, 4, 8});
    GentleFill(x);

    Tensor<bool> rank3({2, 4, 4}, false);
    Tensor<bool> rank4({2, 1, 4, 4}, false);
    for (size_t i = 0; i < 4; ++i) {
        rank3(1, i, 3) = true;
        rank4(1, 0, i, 3) = true;
    }

    const Tensor<float> masked = attn.Forward(x, nullptr, 0, rank3);
    const Tensor<float> reference = attn.Forward(x, nullptr, 0, rank4);
    const Tensor<float> unmasked = attn.Forward(x);

    bool batch1_changed = false;
    for (size_t s = 0; s < 4; ++s)
        for (size_t d = 0; d < 8; ++d) {
            EXPECT_NEAR(masked(0, s, d), reference(0, s, d), kTol);
            EXPECT_NEAR(masked(1, s, d), reference(1, s, d), kTol);
            EXPECT_NEAR(masked(0, s, d), unmasked(0, s, d), kTol);
            batch1_changed |= std::fabs(masked(1, s, d) - unmasked(1, s, d)) > kTol;
        }
    EXPECT_TRUE(batch1_changed);
}

TEST(MultiHeadAttentionMask, RankThreeMaskWorksWithSeparateQKV) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({2, 3, 8});
    Tensor<float> kv({2, 5, 8});
    GentleFill(q);
    GentleFill(kv);

    Tensor<bool> rank3({2, 3, 5}, false);
    Tensor<bool> rank4({2, 1, 3, 5}, false);
    for (size_t i = 0; i < 3; ++i) {
        rank3(0, i, 4) = true;
        rank4(0, 0, i, 4) = true;
    }

    const Tensor<float> masked = attn.Forward(q, kv, kv, rank3);
    const Tensor<float> reference = attn.Forward(q, kv, kv, rank4);

    for (size_t b = 0; b < 2; ++b)
        for (size_t s = 0; s < 3; ++s)
            for (size_t d = 0; d < 8; ++d) EXPECT_NEAR(masked(b, s, d), reference(b, s, d), kTol);
}

TEST(MultiHeadAttentionMask, RejectsMaskThatDoesNotBroadcast) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> x({2, 4, 8});
    GentleFill(x);

    EXPECT_THROW(attn.Forward(x, nullptr, 0, Tensor<bool>({3, 4, 4}, false)), std::invalid_argument);
    EXPECT_THROW(attn.Forward(x, nullptr, 0, Tensor<bool>({2, 3, 4, 4}, false)), std::invalid_argument);
    EXPECT_THROW(attn.Forward(x, nullptr, 0, Tensor<bool>({4}, false)), std::invalid_argument);
}

// --- separate query, key and value inputs -----------------------------------

TEST(MultiHeadAttentionQKV, SelfAttentionIsTheSpecialCaseOfQKV) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({2, 4, 8});
    PhaseFill(x, 0.0);

    EXPECT_EQ(MaxAbsDiff(attn.Forward(x), attn.Forward(x, x, x)), 0.0);
}

TEST(MultiHeadAttentionQKV, SelfAttentionPassesMaskAndCausalThrough) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 4, 8});
    PhaseFill(x, 0.0);
    Tensor<bool> mask({4, 4}, false);
    for (size_t i = 1; i < 4; ++i) mask(i, 0) = true;   // query 0 keeps key 0, its only causal key

    EXPECT_EQ(MaxAbsDiff(attn.Forward(x, nullptr, 0, mask, true), attn.Forward(x, x, x, mask, true)), 0.0);
}

TEST(MultiHeadAttentionQKV, OutputTakesTheQueryLength) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> q({2, 3, 8}, 1.0f);
    Tensor<float> kv({2, 5, 8}, 1.0f);

    EXPECT_EQ(attn.Forward(q, kv, kv).Shape(), (std::vector<size_t>{2, 3, 8}));
}

TEST(MultiHeadAttentionQKV, MoreQueriesThanKeysIsFine) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 6, 8});
    Tensor<float> kv({1, 2, 8});
    PhaseFill(q, 0.0);
    PhaseFill(kv, 1.0);

    const Tensor<float> out = attn.Forward(q, kv, kv);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{1, 6, 8}));
    for (size_t s = 0; s < 6; ++s)
        for (size_t d = 0; d < 8; ++d) EXPECT_FALSE(std::isnan(out(0, s, d)));
}

TEST(MultiHeadAttentionQKV, CallOperatorForwardsAllThreeInputs) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 2, 8}), k({1, 4, 8}), v({1, 4, 8});
    PhaseFill(q, 0.0);
    PhaseFill(k, 1.0);
    PhaseFill(v, 2.0);
    Tensor<bool> mask({2, 4}, false);
    mask(1, 0) = true;

    EXPECT_EQ(MaxAbsDiff(attn(q, k, v), attn.Forward(q, k, v)), 0.0);
    EXPECT_EQ(MaxAbsDiff(attn(q, k, v, mask, true), attn.Forward(q, k, v, mask, true)), 0.0);
}

TEST(MultiHeadAttentionQKV, DoesNotMutateTheInputs) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 2, 8}), k({1, 3, 8}), v({1, 3, 8});
    PhaseFill(q, 0.0);
    PhaseFill(k, 1.0);
    PhaseFill(v, 2.0);
    const Tensor<float> q0 = q.Clone(), k0 = k.Clone(), v0 = v.Clone();

    const Tensor<float> out = attn.Forward(q, k, v);

    EXPECT_FALSE(out.SharesStorage(q));
    EXPECT_EQ(MaxAbsDiff(q, q0), 0.0);
    EXPECT_EQ(MaxAbsDiff(k, k0), 0.0);
    EXPECT_EQ(MaxAbsDiff(v, v0), 0.0);
}

TEST(MultiHeadAttentionQKV, ChangingTheValuesChangesTheOutput) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 2, 8}), k({1, 3, 8}), v({1, 3, 8});
    PhaseFill(q, 0.0);
    PhaseFill(k, 1.0);
    PhaseFill(v, 2.0);
    Tensor<float> v2 = v.Clone();
    for (size_t d = 0; d < 8; ++d) v2(0, 1, d) = 5.0f;

    EXPECT_GT(MaxAbsDiff(attn.Forward(q, k, v), attn.Forward(q, k, v2)), 1e-3);
}

TEST(MultiHeadAttentionQKV, ChangingTheKeysChangesTheOutput) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 2, 8}), k({1, 3, 8}), v({1, 3, 8});
    PhaseFill(q, 0.0);
    PhaseFill(k, 1.0);
    PhaseFill(v, 2.0);
    Tensor<float> k2 = k.Clone();
    for (size_t d = 0; d < 8; ++d) k2(0, 1, d) = 5.0f;

    EXPECT_GT(MaxAbsDiff(attn.Forward(q, k, v), attn.Forward(q, k2, v)), 1e-3);
}

TEST(MultiHeadAttentionQKV, ZeroValuesGiveAZeroOutput) {
    // Biases start at zero, so zero values project to zero whatever the weights.
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 3, 8}), k({1, 4, 8});
    PhaseFill(q, 0.0);
    PhaseFill(k, 1.0);
    Tensor<float> v({1, 4, 8}, 0.0f);

    const Tensor<float> out = attn.Forward(q, k, v);

    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 8; ++d) EXPECT_FLOAT_EQ(out(0, s, d), 0.0f);
}

TEST(MultiHeadAttentionQKV, ASingleKeyGetsAllTheWeight) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 3, 8}), k({1, 1, 8}), v({1, 1, 8});
    PhaseFill(q, 0.0);
    PhaseFill(k, 1.0);
    PhaseFill(v, 2.0);

    const Tensor<float> out = attn.Forward(q, k, v);
    const Tensor<float> expected = attn.output_proj->Forward(attn.value->Forward(v));

    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_NEAR(out(0, s, d), expected(0, 0, d), kTol) << "query " << s;
}

TEST(MultiHeadAttentionQKV, IdenticalKeysAverageTheValues) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 2, 8}), k({1, 4, 8}), v({1, 4, 8});
    PhaseFill(q, 0.0);
    PhaseFill(v, 2.0);
    for (size_t s = 0; s < 4; ++s)
        for (size_t d = 0; d < 8; ++d) k(0, s, d) = 0.05f * static_cast<float>(d);

    Tensor<float> mean({1, 1, 8}, 0.0f);
    for (size_t s = 0; s < 4; ++s)
        for (size_t d = 0; d < 8; ++d) mean(0, 0, d) += v(0, s, d) / 4.0f;

    const Tensor<float> out = attn.Forward(q, k, v);
    const Tensor<float> expected = attn.output_proj->Forward(attn.value->Forward(mean));

    for (size_t s = 0; s < 2; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_NEAR(out(0, s, d), expected(0, 0, d), kTol) << "query " << s;
}

TEST(MultiHeadAttentionQKV, EachQueryIsAnsweredIndependently) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 4, 8}), kv({1, 5, 8});
    PhaseFill(q, 0.0);
    PhaseFill(kv, 1.0);
    Tensor<float> q2 = q.Clone();
    for (size_t d = 0; d < 8; ++d) q2(0, 2, d) = 5.0f;

    const Tensor<float> a = attn.Forward(q, kv, kv);
    const Tensor<float> b = attn.Forward(q2, kv, kv);

    for (size_t s : {size_t{0}, size_t{1}, size_t{3}})
        EXPECT_LT(PositionDiff(a, b, s), 1e-6) << "query " << s;
    EXPECT_GT(PositionDiff(a, b, 2), 1e-3);
}

TEST(MultiHeadAttentionQKV, PermutingQueriesPermutesTheOutput) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 3, 8}), kv({1, 4, 8});
    PhaseFill(q, 0.0);
    PhaseFill(kv, 1.0);
    const size_t order[3] = {2, 0, 1};
    Tensor<float> shuffled({1, 3, 8});
    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 8; ++d) shuffled(0, s, d) = q(0, order[s], d);

    const Tensor<float> out = attn.Forward(q, kv, kv);
    const Tensor<float> out_shuffled = attn.Forward(shuffled, kv, kv);

    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_NEAR(out_shuffled(0, s, d), out(0, order[s], d), 1e-6f);
}

TEST(MultiHeadAttentionQKV, ReorderingKeyValuePairsLeavesTheOutputUnchanged) {
    // Without a mask, attention is a weighted sum over key/value pairs, so
    // their order must not matter.
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 3, 8}), k({1, 4, 8}), v({1, 4, 8});
    PhaseFill(q, 0.0);
    PhaseFill(k, 1.0);
    PhaseFill(v, 2.0);
    const size_t order[4] = {3, 1, 0, 2};
    Tensor<float> k2({1, 4, 8}), v2({1, 4, 8});
    for (size_t s = 0; s < 4; ++s)
        for (size_t d = 0; d < 8; ++d) {
            k2(0, s, d) = k(0, order[s], d);
            v2(0, s, d) = v(0, order[s], d);
        }

    EXPECT_LT(MaxAbsDiff(attn.Forward(q, k, v), attn.Forward(q, k2, v2)), kTol);
}

TEST(MultiHeadAttentionQKV, RejectsKeysAndValuesOfDifferentLengths) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> q({1, 2, 8}, 1.0f), k({1, 5, 8}, 1.0f), v({1, 4, 8}, 1.0f);

    EXPECT_THROW(attn.Forward(q, k, v), std::invalid_argument);
}

TEST(MultiHeadAttentionQKV, RejectsAMismatchedModelDimension) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> good({1, 3, 8}, 1.0f), wrong({1, 3, 6}, 1.0f);

    EXPECT_THROW(attn.Forward(wrong, good, good), std::invalid_argument);
    EXPECT_THROW(attn.Forward(good, wrong, good), std::invalid_argument);
    EXPECT_THROW(attn.Forward(good, good, wrong), std::invalid_argument);
}

// --- masks over queries x keys ----------------------------------------------

TEST(MultiHeadAttentionQKVMask, AMaskedKeyHasNoInfluence) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 3, 8}), kv({1, 5, 8});
    PhaseFill(q, 0.0);
    PhaseFill(kv, 1.0);
    Tensor<float> kv2 = kv.Clone();
    for (size_t d = 0; d < 8; ++d) kv2(0, 4, d) = 5.0f;

    Tensor<bool> mask({3, 5}, false);
    for (size_t i = 0; i < 3; ++i) mask(i, 4) = true;

    EXPECT_LT(MaxAbsDiff(attn.Forward(q, kv, kv, mask), attn.Forward(q, kv2, kv2, mask)), 1e-6);
    EXPECT_GT(MaxAbsDiff(attn.Forward(q, kv, kv), attn.Forward(q, kv2, kv2)), 1e-3)
        << "without the mask the change must be visible";
}

TEST(MultiHeadAttentionQKVMask, EachMaskRowAppliesToItsOwnQuery) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 2, 8}), kv({1, 4, 8});
    PhaseFill(q, 0.0);
    PhaseFill(kv, 1.0);
    Tensor<float> kv2 = kv.Clone();
    for (size_t d = 0; d < 8; ++d) kv2(0, 3, d) = 5.0f;

    Tensor<bool> mask({2, 4}, false);
    mask(0, 3) = true;   // only query 0 is blocked from key 3

    const Tensor<float> a = attn.Forward(q, kv, kv, mask);
    const Tensor<float> b = attn.Forward(q, kv2, kv2, mask);

    EXPECT_LT(PositionDiff(a, b, 0), 1e-6);
    EXPECT_GT(PositionDiff(a, b, 1), 1e-3);
}

// --- causal masking with fewer queries than keys ----------------------------
// Queries line up with the end of the keys, as when decoding against a cache
// of earlier tokens: with 2 queries over 5 keys they sit at positions 3 and 4.

TEST(MultiHeadAttentionQKVCausal, MatchesTheTailOfFullSelfAttention) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 5, 8});
    PhaseFill(x, 0.0);

    const Tensor<float> full = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);
    const Tensor<float> tail = attn.Forward(Positions(x, 3, 5), x, x, Tensor<bool>(), true);

    ASSERT_EQ(tail.Shape(), (std::vector<size_t>{1, 2, 8}));
    for (size_t s = 0; s < 2; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_NEAR(tail(0, s, d), full(0, s + 3, d), kTol) << "query " << s;
}

TEST(MultiHeadAttentionQKVCausal, OneQueryAtATimeRebuildsTheSequence) {
    // Decoding token by token must reproduce the full causal pass.
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 5, 8});
    PhaseFill(x, 0.0);

    const Tensor<float> full = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);

    for (size_t t = 0; t < 5; ++t) {
        const Tensor<float> prefix = Positions(x, 0, t + 1);
        const Tensor<float> step = attn.Forward(Positions(x, t, t + 1), prefix, prefix, Tensor<bool>(), true);
        for (size_t d = 0; d < 8; ++d)
            EXPECT_NEAR(step(0, 0, d), full(0, t, d), kTol) << "step " << t;
    }
}

TEST(MultiHeadAttentionQKVCausal, ASingleFinalQuerySeesEveryKey) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({1, 1, 8}), kv({1, 4, 8});
    PhaseFill(q, 0.0);
    PhaseFill(kv, 1.0);

    EXPECT_EQ(MaxAbsDiff(attn.Forward(q, kv, kv, Tensor<bool>(), true), attn.Forward(q, kv, kv)), 0.0);
}

TEST(MultiHeadAttentionQKVCausal, QueriesIgnoreKeysAfterTheirPosition) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> kv({1, 5, 8});
    PhaseFill(kv, 1.0);
    const Tensor<float> q = Positions(kv, 3, 5);   // positions 3 and 4
    Tensor<float> kv2 = kv.Clone();
    for (size_t d = 0; d < 8; ++d) kv2(0, 4, d) = 5.0f;

    const Tensor<float> a = attn.Forward(q, kv, kv, Tensor<bool>(), true);
    const Tensor<float> b = attn.Forward(q, kv2, kv2, Tensor<bool>(), true);

    EXPECT_LT(PositionDiff(a, b, 0), 1e-6) << "position 3 must not see key 4";
    EXPECT_GT(PositionDiff(a, b, 1), 1e-3) << "position 4 must see key 4";
}

TEST(MultiHeadAttentionQKVCausal, CombinesWithAnExplicitMask) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> kv({1, 5, 8});
    PhaseFill(kv, 1.0);
    const Tensor<float> q = Positions(kv, 3, 5);
    Tensor<float> kv2 = kv.Clone();
    for (size_t d = 0; d < 8; ++d) kv2(0, 0, d) = 5.0f;

    Tensor<bool> mask({2, 5}, false);
    mask(0, 0) = true;
    mask(1, 0) = true;   // hide key 0 from both queries

    const Tensor<float> a = attn.Forward(q, kv, kv, mask, true);
    const Tensor<float> b = attn.Forward(q, kv2, kv2, mask, true);

    EXPECT_LT(MaxAbsDiff(a, b), 1e-6);
}

TEST(MultiHeadAttentionQKVCausal, NoNaNWithFewerQueries) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> q({2, 3, 8}), kv({2, 7, 8});
    PhaseFill(q, 0.0);
    PhaseFill(kv, 1.0);

    const Tensor<float> out = attn.Forward(q, kv, kv, Tensor<bool>(), true);

    for (size_t b = 0; b < 2; ++b)
        for (size_t s = 0; s < 3; ++s)
            for (size_t d = 0; d < 8; ++d) EXPECT_FALSE(std::isnan(out(b, s, d)));
}

// --- Module interface -------------------------------------------------------

TEST(MultiHeadAttentionModule, ReportsParametersThroughABasePointer) {
    std::shared_ptr<Module<float>> m = std::make_shared<MultiHeadAttention<float>>(8, 2);

    EXPECT_EQ(m->Parameters().size(), 8u);
}

TEST(MultiHeadAttentionModule, ParametersReachTheProjections) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);

    Tensor<float> probe({1, 3, 8}, 1.0f);
    const Tensor<float> before = attn.Forward(probe);
    bool nonzero = false;
    for (size_t s = 0; s < 3 && !nonzero; ++s)
        for (size_t d = 0; d < 8; ++d)
            if (before(0, s, d) != 0.0f) { nonzero = true; break; }
    ASSERT_TRUE(nonzero) << "seeded weights must give a non-zero output";

    for (auto& p : attn.Parameters()) {
        p->Fill(0.0f);
    }

    Tensor<float> x({1, 3, 8}, 1.0f);
    const Tensor<float> out = attn.Forward(x);

    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_FLOAT_EQ(out(0, s, d), 0.0f) << "zeroed weights must zero the output";
}

// --- element type -----------------------------------------------------------

TEST(MultiHeadAttentionElementType, WorksWithDouble) {
    MultiHeadAttention<double> attn(8, 2);
    Tensor<double> x({1, 3, 8}, 1.0);

    const Tensor<double> out = attn.Forward(x);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{1, 3, 8}));
    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_FALSE(std::isnan(out(0, s, d)));
}

TEST(MultiHeadAttentionElementType, QKVWorksWithDouble) {
    MultiHeadAttention<double> attn(8, 2);
    SeedProjections(attn);
    Tensor<double> q({1, 2, 8}), kv({1, 4, 8});
    PhaseFill(q, 0.0);
    PhaseFill(kv, 1.0);

    const Tensor<double> out = attn.Forward(q, kv, kv, Tensor<bool>(), true);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{1, 2, 8}));
    for (size_t s = 0; s < 2; ++s)
        for (size_t d = 0; d < 8; ++d) EXPECT_FALSE(std::isnan(out(0, s, d)));
}
