#include <gtest/gtest.h>

#include <NeuralNet/Module.h>
#include <NeuralNet/MultiHeadAttention.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using math::Tensor;
using nn::Module;
using nn::MultiHeadAttention;

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
    Tensor<T> q = attn.SplitHeads((*attn.query)(x));
    Tensor<T> k = attn.SplitHeads((*attn.key)(x));
    Tensor<T> scores = math::MatMul(q, k.Transpose(-2, -1)) / attn.scale;

    if (causal) {
        const size_t seq = scores.Shape()[scores.Rank() - 1];
        Tensor<bool> mask({seq, seq}, false);
        for (size_t i = 0; i < seq; ++i)
            for (size_t j = i + 1; j < seq; ++j) mask(i, j) = true;
        scores = scores.MaskedFill(mask, -std::numeric_limits<T>::infinity());
    }
    return activation::Softmax<T>(-1).Forward(scores);
}

}  // namespace

// --- construction -----------------------------------------------------------

TEST(MultiHeadAttentionConstruction, DerivesHeadDimension) {
    MultiHeadAttention<float> attn(512, 8);

    EXPECT_EQ(attn.d_model, 512u);
    EXPECT_EQ(attn.num_heads, 8u);
    EXPECT_EQ(attn.head_dim, 64u);
}

TEST(MultiHeadAttentionConstruction, ScaleIsSqrtOfHeadDimension) {
    MultiHeadAttention<float> attn(512, 8);
    EXPECT_NEAR(attn.scale, std::sqrt(64.0f), kTol);
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

// --- head splitting and merging ---------------------------------------------

TEST(MultiHeadAttentionHeads, SplitProducesBatchHeadSequenceDim) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> x({3, 5, 8}, 1.0f);

    const Tensor<float> split = attn.SplitHeads(x);

    EXPECT_EQ(split.Shape(), (std::vector<size_t>{3, 2, 5, 4}));
}

TEST(MultiHeadAttentionHeads, EachHeadTakesAContiguousChannelRange) {
    MultiHeadAttention<float> attn(8, 2);   // head_dim 4
    Tensor<float> x({1, 2, 8});
    IotaFill(x);                            // position 0 holds 1..8

    const Tensor<float> split = attn.SplitHeads(x);

    for (size_t d = 0; d < 4; ++d) {
        EXPECT_FLOAT_EQ(split(0, 0, 0, d), static_cast<float>(1 + d)) << "head 0";
        EXPECT_FLOAT_EQ(split(0, 1, 0, d), static_cast<float>(5 + d)) << "head 1";
    }
}

TEST(MultiHeadAttentionHeads, MergeInvertsSplit) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> x({2, 3, 8});
    IotaFill(x);

    const Tensor<float> back = attn.MergeHeads(attn.SplitHeads(x));

    ASSERT_EQ(back.Shape(), x.Shape());
    for (size_t b = 0; b < 2; ++b)
        for (size_t s = 0; s < 3; ++s)
            for (size_t d = 0; d < 8; ++d)
                EXPECT_FLOAT_EQ(back(b, s, d), x(b, s, d))
                    << "at (" << b << ", " << s << ", " << d << ")";
}

TEST(MultiHeadAttentionHeads, SplitResultIsContiguous) {
    MultiHeadAttention<float> attn(8, 2);
    Tensor<float> x({1, 4, 8}, 1.0f);

    EXPECT_TRUE(attn.SplitHeads(x).IsContiguous());
}

TEST(MultiHeadAttentionHeads, SingleHeadSplitIsAReshape) {
    MultiHeadAttention<float> attn(8, 1);
    Tensor<float> x({1, 3, 8});
    IotaFill(x);

    const Tensor<float> split = attn.SplitHeads(x);

    ASSERT_EQ(split.Shape(), (std::vector<size_t>{1, 1, 3, 8}));
    EXPECT_FLOAT_EQ(split(0, 0, 2, 7), x(0, 2, 7));
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

    const Tensor<float> out_a = attn.Forward(a, Tensor<bool>(), true);
    const Tensor<float> out_b = attn.Forward(b, Tensor<bool>(), true);

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
    const Tensor<float> out = attn.Forward(x, Tensor<bool>(), true);

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

    const Tensor<float> out_a = attn.Forward(a, Tensor<bool>(), true);
    const Tensor<float> out_b = attn.Forward(b, Tensor<bool>(), true);

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
    const Tensor<float> causal = attn.Forward(x, Tensor<bool>(), true);

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

    EXPECT_NO_THROW(attn.Forward(x, mask));
}

TEST(MultiHeadAttentionMask, MaskedKeysGetZeroWeight) {
    MultiHeadAttention<float> attn(8, 2);
    SeedProjections(attn);
    Tensor<float> x({1, 4, 8});
    IotaFill(x);

    Tensor<float> q = attn.SplitHeads((*attn.query)(x));
    Tensor<float> k = attn.SplitHeads((*attn.key)(x));
    Tensor<float> scores = math::MatMul(q, k.Transpose(-2, -1)) / attn.scale;

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

    const Tensor<float> withEmpty = attn.Forward(x, Tensor<bool>());
    const Tensor<float> without = attn.Forward(x);

    for (size_t s = 0; s < 4; ++s)
        for (size_t d = 0; d < 8; ++d)
            EXPECT_FLOAT_EQ(withEmpty(0, s, d), without(0, s, d));
}

// --- Module interface -------------------------------------------------------

TEST(MultiHeadAttentionModule, DispatchesThroughABasePointer) {
    std::shared_ptr<Module<float>> m = std::make_shared<MultiHeadAttention<float>>(8, 2);

    Tensor<float> x({1, 4, 8}, 1.0f);
    const Tensor<float> out = m->Forward(x);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{1, 4, 8}));
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
