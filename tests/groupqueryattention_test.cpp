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

namespace {

constexpr float kTol = 1e-5f;

void WaveFill(Tensor<float>& t, double phase) {
    float* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i)
        data[i] = static_cast<float>(0.5 * std::sin(phase + 0.37 * static_cast<double>(i)));
}

void SeedProjections(GroupQueryAttention<float>& attn) {
    double phase = 0.1;
    for (const auto& linear : {attn.query, attn.key, attn.value, attn.output_proj}) {
        WaveFill(*linear->weight, phase);
        WaveFill(*linear->bias, phase + 0.05);
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
    *mha.query->bias = gqa.query->bias->Clone();
    *mha.output_proj->weight = gqa.output_proj->weight->Clone();
    *mha.output_proj->bias = gqa.output_proj->bias->Clone();
    ExpandHeads(*gqa.key->weight, *mha.key->weight, gqa.headDim, gqa.numGroups);
    ExpandHeads(*gqa.key->bias, *mha.key->bias, gqa.headDim, gqa.numGroups);
    ExpandHeads(*gqa.value->weight, *mha.value->weight, gqa.headDim, gqa.numGroups);
    ExpandHeads(*gqa.value->bias, *mha.value->bias, gqa.headDim, gqa.numGroups);
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

}  // namespace

TEST(GroupQueryAttention, ProjectionShapesFollowHeadCounts) {
    GroupQueryAttention<float> attn(kModel, kHeadsQ, 2);

    EXPECT_EQ(attn.headDim, 4u);
    EXPECT_EQ(attn.numGroups, 2u);
    EXPECT_EQ(attn.query->weight->Shape(), (std::vector<size_t>{16, 16}));
    EXPECT_EQ(attn.key->weight->Shape(), (std::vector<size_t>{8, 16}));
    EXPECT_EQ(attn.value->weight->Shape(), (std::vector<size_t>{8, 16}));
    EXPECT_EQ(attn.output_proj->weight->Shape(), (std::vector<size_t>{16, 16}));
}

TEST(GroupQueryAttention, RejectsZeroKVHeads) {
    EXPECT_THROW(GroupQueryAttention<float>(kModel, kHeadsQ, 0), std::invalid_argument);
}

TEST(GroupQueryAttention, RejectsQueryHeadsNotDivisibleByKVHeads) {
    EXPECT_THROW(GroupQueryAttention<float>(kModel, kHeadsQ, 3), std::invalid_argument);
}

TEST(GroupQueryAttention, RejectsModelWidthNotDivisibleByQueryHeads) {
    EXPECT_THROW(GroupQueryAttention<float>(18, kHeadsQ, 2), std::invalid_argument);
}

TEST(GroupQueryAttention, OutputKeepsInputShape) {
    GroupQueryAttention<float> attn(kModel, kHeadsQ, 2);
    SeedProjections(attn);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    EXPECT_EQ(attn.Forward(x).Shape(), x.Shape());
}

class GroupQueryAttentionHeads : public ::testing::TestWithParam<size_t> {};

TEST_P(GroupQueryAttentionHeads, MatchesMHAWithKVHeadsSharedAcrossGroup) {
    GroupQueryAttention<float> gqa(kModel, kHeadsQ, GetParam());
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
    GroupQueryAttention<float> attn(kModel, kHeadsQ, GetParam());
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

INSTANTIATE_TEST_SUITE_P(KVHeadCounts, GroupQueryAttentionHeads, ::testing::Values(4, 2, 1),
                         [](const ::testing::TestParamInfo<size_t>& info) {
                             return "Hkv" + std::to_string(info.param);
                         });

TEST(GroupQueryAttention, CacheStoresOnlyKVHeads) {
    GroupQueryAttention<float> attn(kModel, kHeadsQ, 2);
    SeedProjections(attn);
    Tensor<float> x({kBatch, kSeq, kModel});
    WaveFill(x, 1.3);

    KVCache<float> compact(1, kBatch, attn.numHeadsKV, kSeq, attn.headDim);
    EXPECT_NO_THROW(attn.Forward(x, &compact.Layer(0), 0, Tensor<bool>(), true));

    KVCache<float> full_heads(1, kBatch, attn.numHeadsQ, kSeq, attn.headDim);
    EXPECT_THROW(attn.Forward(x, &full_heads.Layer(0), 0, Tensor<bool>(), true), std::invalid_argument);
}
