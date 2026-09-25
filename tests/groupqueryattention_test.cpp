#include <gtest/gtest.h>

#include <NeuralNet/Transformer/GroupQueryAttention.h>
#include <NeuralNet/Transformer/KVCache.h>
#include <NeuralNet/Transformer/MultiHeadAttention.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using math::Tensor;
using transformer::GroupQueryAttention;
using transformer::KVCache;
using transformer::MultiHeadAttention;
using transformer::TransformerConfig;

namespace {

constexpr float kTol = 1e-5f;

void WaveFill(Tensor<float>& t, double phase) {
    float* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i)
        data[i] = static_cast<float>(0.5 * std::sin(phase + 0.37 * static_cast<double>(i)));
}

TransformerConfig Config(size_t d_model, size_t heads, size_t kv_heads, size_t head_dim = 0,
                         bool attention_bias = true, bool output_bias = true) {
    TransformerConfig cfg;
    cfg.d_model = d_model;
    cfg.num_heads = heads;
    cfg.num_kv_heads = kv_heads;
    cfg.head_dim = head_dim;
    cfg.attention_bias = attention_bias;
    cfg.output_bias = output_bias;
    return cfg;
}

void SeedProjections(GroupQueryAttention<float>& attn) {
    double phase = 0.1;
    for (const auto& linear : {attn.query, attn.key, attn.value, attn.output_proj}) {
        WaveFill(*linear->weight, phase);
        if (linear->bias) WaveFill(*linear->bias, phase + 0.05);
        phase += 0.2;
    }
}

void ExpandHeads(const Tensor<float>& src, Tensor<float>& dst, size_t head_dim, size_t groups) {
    const size_t cols = src.Rank() == 2 ? src.Shape()[1] : 1;
    for (size_t row = 0; row < dst.Shape()[0]; ++row) {
        const size_t src_row = (row / head_dim / groups) * head_dim + row % head_dim;
        for (size_t c = 0; c < cols; ++c) dst.Data()[row * cols + c] = src.Data()[src_row * cols + c];
    }
}

MultiHeadAttention<float> ExpandedMHA(const GroupQueryAttention<float>& gqa) {
    MultiHeadAttention<float> mha(gqa.dModel, gqa.numHeadsQ);
    *mha.query->weight = gqa.query->weight->Clone();
    *mha.output_proj->weight = gqa.output_proj->weight->Clone();
    ExpandHeads(*gqa.key->weight, *mha.key->weight, gqa.headDim, gqa.numGroups);
    ExpandHeads(*gqa.value->weight, *mha.value->weight, gqa.headDim, gqa.numGroups);
    if (gqa.query->bias) {
        *mha.query->bias = gqa.query->bias->Clone();
        ExpandHeads(*gqa.key->bias, *mha.key->bias, gqa.headDim, gqa.numGroups);
        ExpandHeads(*gqa.value->bias, *mha.value->bias, gqa.headDim, gqa.numGroups);
    }
    if (gqa.output_proj->bias) *mha.output_proj->bias = gqa.output_proj->bias->Clone();
    return mha;
}

void ExpectNear(const Tensor<float>& actual, const Tensor<float>& expected) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    Tensor<float> a = actual.Contiguous();
    Tensor<float> e = expected.Contiguous();
    for (size_t i = 0; i < a.Size(); ++i) EXPECT_NEAR(a.Data()[i], e.Data()[i], kTol) << "at flat index " << i;
}

float MaxAbsDiff(const Tensor<float>& lhs, const Tensor<float>& rhs) {
    Tensor<float> a = lhs.Contiguous();
    Tensor<float> b = rhs.Contiguous();
    float diff = 0.0f;
    for (size_t i = 0; i < a.Size(); ++i) diff = std::fmax(diff, std::fabs(a.Data()[i] - b.Data()[i]));
    return diff;
}

constexpr size_t kBatch = 2;
constexpr size_t kSeq = 5;
constexpr size_t kModel = 16;
constexpr size_t kHeadsQ = 4;
constexpr size_t kHeadDim = 8;

}  // namespace

TEST(GroupQueryAttention, ProjectionShapesFollowHeadCounts) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, 2));

    EXPECT_EQ(attn.headDim, 4u);
    EXPECT_EQ(attn.numGroups, 2u);
    EXPECT_EQ(attn.query->weight->Shape(), (std::vector<size_t>{16, 16}));
    EXPECT_EQ(attn.key->weight->Shape(), (std::vector<size_t>{8, 16}));
    EXPECT_EQ(attn.value->weight->Shape(), (std::vector<size_t>{8, 16}));
    EXPECT_EQ(attn.output_proj->weight->Shape(), (std::vector<size_t>{16, 16}));
}

TEST(GroupQueryAttention, ZeroKVHeadsFallsBackToMultiHead) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, 0));

    EXPECT_EQ(attn.numHeadsKV, kHeadsQ);
    EXPECT_EQ(attn.numGroups, 1u);
    EXPECT_EQ(attn.key->weight->Shape(), (std::vector<size_t>{16, 16}));
}

TEST(GroupQueryAttention, RejectsZeroModelWidth) {
    EXPECT_THROW(GroupQueryAttention<float>(Config(0, kHeadsQ, 2)), std::invalid_argument);
    EXPECT_THROW(GroupQueryAttention<float>(Config(0, kHeadsQ, 2, kHeadDim)), std::invalid_argument);
}

TEST(GroupQueryAttention, RejectsZeroQueryHeads) {
    EXPECT_THROW(GroupQueryAttention<float>(Config(kModel, 0, 0)), std::invalid_argument);
    EXPECT_THROW(GroupQueryAttention<float>(Config(kModel, 0, 2)), std::invalid_argument);
}

TEST(GroupQueryAttention, RejectsQueryHeadsNotDivisibleByKVHeads) {
    EXPECT_THROW(GroupQueryAttention<float>(Config(kModel, kHeadsQ, 3)), std::invalid_argument);
}

TEST(GroupQueryAttention, RejectsModelWidthNotDivisibleByQueryHeads) {
    EXPECT_THROW(GroupQueryAttention<float>(Config(18, kHeadsQ, 2)), std::invalid_argument);
}

TEST(GroupQueryAttention, ExplicitHeadDimDecouplesProjectionWidth) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, 2, kHeadDim));

    EXPECT_EQ(attn.headDim, 8u);
    EXPECT_EQ(attn.numGroups, 2u);
    EXPECT_FLOAT_EQ(attn.scale, 1.0f / std::sqrt(8.0f));
    EXPECT_EQ(attn.query->weight->Shape(), (std::vector<size_t>{32, 16}));
    EXPECT_EQ(attn.key->weight->Shape(), (std::vector<size_t>{16, 16}));
    EXPECT_EQ(attn.value->weight->Shape(), (std::vector<size_t>{16, 16}));
    EXPECT_EQ(attn.output_proj->weight->Shape(), (std::vector<size_t>{16, 32}));
}

TEST(GroupQueryAttention, ZeroHeadDimDerivesFromModelWidth) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, 2, 0));

    EXPECT_EQ(attn.headDim, kModel / kHeadsQ);
}

TEST(GroupQueryAttention, ExplicitHeadDimSkipsModelWidthDivisibility) {
    GroupQueryAttention<float> attn(Config(18, kHeadsQ, 2, kHeadDim));

    EXPECT_EQ(attn.query->weight->Shape(), (std::vector<size_t>{32, 18}));
    EXPECT_EQ(attn.output_proj->weight->Shape(), (std::vector<size_t>{18, 32}));
}

TEST(GroupQueryAttention, ExplicitHeadDimStillValidatesKVHeads) {
    EXPECT_THROW(GroupQueryAttention<float>(Config(kModel, kHeadsQ, 3, kHeadDim)), std::invalid_argument);
}

TEST(GroupQueryAttention, BiasFlagsControlProjectionsIndependently) {
    GroupQueryAttention<float> qwen_style(Config(kModel, kHeadsQ, 2, 0, true, false));
    EXPECT_TRUE(qwen_style.query->bias);
    EXPECT_TRUE(qwen_style.key->bias);
    EXPECT_TRUE(qwen_style.value->bias);
    EXPECT_FALSE(qwen_style.output_proj->bias);

    GroupQueryAttention<float> llama_style(Config(kModel, kHeadsQ, 2, 0, false, false));
    EXPECT_FALSE(llama_style.query->bias);
    EXPECT_FALSE(llama_style.key->bias);
    EXPECT_FALSE(llama_style.value->bias);
    EXPECT_FALSE(llama_style.output_proj->bias);
}

TEST(GroupQueryAttention, BiasFreeLayerMatchesMHAWithZeroBias) {
    GroupQueryAttention<float> gqa(Config(kModel, kHeadsQ, 2, 0, false, false));
    SeedProjections(gqa);
    MultiHeadAttention<float> mha = ExpandedMHA(gqa);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    ExpectNear(gqa.Forward(x, nullptr, 0, Tensor<bool>(), true), mha.Forward(x, nullptr, 0, Tensor<bool>(), true));
}

TEST(GroupQueryAttention, ExplicitHeadDimMatchingDefaultGivesSameOutput) {
    GroupQueryAttention<float> implicit(Config(kModel, kHeadsQ, 2));
    GroupQueryAttention<float> explicit_dim(Config(kModel, kHeadsQ, 2, kModel / kHeadsQ));
    SeedProjections(implicit);
    SeedProjections(explicit_dim);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    ExpectNear(explicit_dim.Forward(x, nullptr, 0, Tensor<bool>(), true),
               implicit.Forward(x, nullptr, 0, Tensor<bool>(), true));
}

TEST(GroupQueryAttention, ExplicitHeadDimKeepsModelWidthOutput) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, 2, kHeadDim));
    SeedProjections(attn);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    EXPECT_EQ(attn.Forward(x).Shape(), x.Shape());
}

TEST(GroupQueryAttention, ExplicitHeadDimCachedDecodeMatchesFullCausalPass) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, 2, kHeadDim));
    SeedProjections(attn);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    Tensor<float> full = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);
    ASSERT_GT(MaxAbsDiff(full, attn.Forward(x)), 1e-2f);

    KVCache<float> cache(1, kBatch, attn.numHeadsKV, kSeq, attn.headDim);
    for (size_t t = 0; t < kSeq; ++t) {
        SCOPED_TRACE(t);
        ExpectNear(attn.Forward(x.Slice(1, t, 1), &cache.Layer(0), cache.Room(1), Tensor<bool>(), true),
                   full.Slice(1, t, 1));
        cache.Advance(1);
    }
}

TEST(GroupQueryAttention, OutputKeepsInputShape) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, 2));
    SeedProjections(attn);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    EXPECT_EQ(attn.Forward(x).Shape(), x.Shape());
}

class GroupQueryAttentionHeads : public ::testing::TestWithParam<size_t> {};

TEST_P(GroupQueryAttentionHeads, MatchesMHAWithKVHeadsSharedAcrossGroup) {
    GroupQueryAttention<float> gqa(Config(kModel, kHeadsQ, GetParam()));
    SeedProjections(gqa);
    MultiHeadAttention<float> mha = ExpandedMHA(gqa);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    for (bool causal : {false, true}) {
        SCOPED_TRACE(causal ? "causal" : "bidirectional");
        ExpectNear(gqa.Forward(x, nullptr, 0, Tensor<bool>(), causal),
                   mha.Forward(x, nullptr, 0, Tensor<bool>(), causal));
    }
}

TEST_P(GroupQueryAttentionHeads, CachedPrefillThenDecodeMatchesFullCausalPass) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, GetParam()));
    SeedProjections(attn);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    Tensor<float> full = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);
    ASSERT_GT(MaxAbsDiff(full, attn.Forward(x)), 1e-2f);

    KVCache<float> cache(1, kBatch, attn.numHeadsKV, kSeq, attn.headDim);
    const size_t prefill = 2;
    ExpectNear(attn.Forward(x.Slice(1, 0, prefill), &cache.Layer(0), cache.Room(prefill), Tensor<bool>(), true),
               full.Slice(1, 0, prefill));
    cache.Advance(prefill);

    for (size_t t = prefill; t < kSeq; ++t) {
        SCOPED_TRACE(t);
        ExpectNear(attn.Forward(x.Slice(1, t, 1), &cache.Layer(0), cache.Room(1), Tensor<bool>(), true),
                   full.Slice(1, t, 1));
        cache.Advance(1);
    }
}

TEST_P(GroupQueryAttentionHeads, PaddingMaskAppliesPerBatchNotPerKVHead) {
    GroupQueryAttention<float> gqa(Config(kModel, kHeadsQ, GetParam()));
    SeedProjections(gqa);
    MultiHeadAttention<float> mha = ExpandedMHA(gqa);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    Tensor<bool> padding({kBatch, 1, kSeq, kSeq}, false);
    for (size_t q = 0; q < kSeq; ++q) padding(1, 0, q, kSeq - 1) = true;

    Tensor<float> masked = gqa.Forward(x, nullptr, 0, padding);
    Tensor<float> unmasked = gqa.Forward(x);

    ExpectNear(masked, mha.Forward(x, nullptr, 0, padding));
    ExpectNear(masked.Slice(0, 0, 1), unmasked.Slice(0, 0, 1));
    EXPECT_GT(MaxAbsDiff(masked.Slice(0, 1, 1), unmasked.Slice(0, 1, 1)), 1e-3f);
}

TEST_P(GroupQueryAttentionHeads, RankThreeMaskMatchesRankFourEquivalent) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, GetParam()));
    SeedProjections(attn);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    Tensor<bool> rank3({kBatch, kSeq, kSeq}, false);
    Tensor<bool> rank4({kBatch, 1, kSeq, kSeq}, false);
    for (size_t q = 0; q < kSeq; ++q) {
        rank3(1, q, 0) = true;
        rank4(1, 0, q, 0) = true;
    }

    ExpectNear(attn.Forward(x, nullptr, 0, rank3), attn.Forward(x, nullptr, 0, rank4));
}

TEST_P(GroupQueryAttentionHeads, PerHeadMaskMatchesMHA) {
    GroupQueryAttention<float> gqa(Config(kModel, kHeadsQ, GetParam()));
    SeedProjections(gqa);
    MultiHeadAttention<float> mha = ExpandedMHA(gqa);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    Tensor<bool> per_head({kBatch, kHeadsQ, kSeq, kSeq}, false);
    for (size_t b = 0; b < kBatch; ++b)
        for (size_t h = 0; h < kHeadsQ; ++h)
            for (size_t q = 0; q < kSeq; ++q) per_head(b, h, q, (h + b) % kSeq) = true;

    Tensor<float> masked = gqa.Forward(x, nullptr, 0, per_head);
    ExpectNear(masked, mha.Forward(x, nullptr, 0, per_head));
    EXPECT_GT(MaxAbsDiff(masked, gqa.Forward(x)), 1e-3f);
}

INSTANTIATE_TEST_SUITE_P(KVHeadCounts, GroupQueryAttentionHeads, ::testing::Values(4, 2, 1),
                         [](const ::testing::TestParamInfo<size_t>& info) {
                             return "Hkv" + std::to_string(info.param);
                         });

TEST(GroupQueryAttention, CacheStoresOnlyKVHeads) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, 2));
    SeedProjections(attn);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    KVCache<float> compact(1, kBatch, attn.numHeadsKV, kSeq, attn.headDim);
    EXPECT_NO_THROW(attn.Forward(x, &compact.Layer(0), 0, Tensor<bool>(), true));

    KVCache<float> full_heads(1, kBatch, attn.numHeadsQ, kSeq, attn.headDim);
    EXPECT_THROW(attn.Forward(x, &full_heads.Layer(0), 0, Tensor<bool>(), true), std::invalid_argument);
}

TEST(GroupQueryAttention, RejectsMaskNotBroadcastableToQueryHeads) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, 2));
    SeedProjections(attn);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    EXPECT_THROW(attn.Forward(x, nullptr, 0, Tensor<bool>({kBatch, 2, kSeq, kSeq}, false)), std::invalid_argument);
    EXPECT_THROW(attn.Forward(x, nullptr, 0, Tensor<bool>({3, kSeq, kSeq}, false)), std::invalid_argument);
    EXPECT_THROW(attn.Forward(x, nullptr, 0, Tensor<bool>({kBatch, 1, 1, kSeq, kSeq}, false)), std::invalid_argument);
}

TEST(GroupQueryAttention, FullyMaskedRowStaysFinite) {
    GroupQueryAttention<float> attn(Config(kModel, kHeadsQ, 2));
    SeedProjections(attn);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    Tensor<bool> mask({kBatch, 1, kSeq, kSeq}, false);
    for (size_t k = 0; k < kSeq; ++k) mask(0, 0, 0, k) = true;

    Tensor<float> out = attn.Forward(x, nullptr, 0, mask).Contiguous();
    for (size_t i = 0; i < out.Size(); ++i) ASSERT_TRUE(std::isfinite(out.Data()[i])) << "at flat index " << i;
}
