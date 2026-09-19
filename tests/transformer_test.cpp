#include <gtest/gtest.h>

#include <Models/ModelConfigs.h>
#include <NeuralNet/Transformer/Transformer.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using math::Tensor;
using nn::Module;
using transformer::Transformer;
using transformer::TransformerBlock;
using transformer::TransformerConfig;

namespace {

constexpr float kTol = 1e-4f;

// Linear layers start at zero, which makes every block an exact identity
// (both residual branches add nothing). Seed every parameter with a varied,
// deterministic pattern so the tests exercise the real computation.
template <typename T>
void SeedParameters(Module<T>& module) {
    int k = 0;
    for (const auto& param : module.NamedParameters()) {
        T* data = param.tensor->Data();
        for (size_t i = 0; i < param.tensor->Size(); ++i)
            data[i] = static_cast<T>(0.3 * std::sin(0.37 * static_cast<double>(++k)));
    }
}

template <typename T>
Tensor<T> Wave(const std::vector<size_t>& shape, double phase = 0.0) {
    Tensor<T> t(shape);
    T* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i)
        data[i] = static_cast<T>(std::sin(0.9 * static_cast<double>(i + 1) + phase));
    return t;
}

template <typename T>
double MaxAbsDiff(const Tensor<T>& a, const Tensor<T>& b) {
    const Tensor<T> ac = a.Contiguous();
    const Tensor<T> bc = b.Contiguous();
    double worst = 0.0;
    for (size_t i = 0; i < ac.Size(); ++i)
        worst = std::max(worst, static_cast<double>(std::fabs(ac.Data()[i] - bc.Data()[i])));
    return worst;
}

// Largest difference at one sequence position of two {B, S, D} tensors.
template <typename T>
double PositionDiff(const Tensor<T>& a, const Tensor<T>& b, size_t s) {
    double worst = 0.0;
    for (size_t n = 0; n < a.Shape()[0]; ++n)
        for (size_t d = 0; d < a.Shape()[2]; ++d)
            worst = std::max(worst, static_cast<double>(std::fabs(a(n, s, d) - b(n, s, d))));
    return worst;
}

template <typename T>
std::vector<std::string> NamesOf(const Module<T>& m) {
    std::vector<std::string> names;
    for (const auto& param : m.NamedParameters()) names.push_back(param.name);
    return names;
}

template <typename T>
size_t ElementCount(const Module<T>& m) {
    size_t total = 0;
    for (const auto& param : m.NamedParameters()) total += param.tensor->Size();
    return total;
}

TransformerConfig SmallConfig(size_t layers = 2) {
    TransformerConfig c;
    c.d_model = 16;
    c.num_heads = 4;
    c.ffn_dim = 64;
    c.num_layers = layers;
    return c;
}

}  // namespace

// --- TransformerConfig ------------------------------------------------------

TEST(TransformerConfig, GPT2SmallPreset) {
    const TransformerConfig c = models::gpt2::Small();

    EXPECT_EQ(c.vocab_size, 50257u);
    EXPECT_EQ(c.max_seq_len, 1024u);
    EXPECT_EQ(c.d_model, 768u);
    EXPECT_EQ(c.num_layers, 12u);
    EXPECT_EQ(c.num_heads, 12u);
    EXPECT_EQ(c.ffn_dim, 4 * c.d_model);
    EXPECT_FLOAT_EQ(c.norm_eps, 1e-5f);
    EXPECT_EQ(c.eos_token_id, 50256);
    EXPECT_EQ(c.d_model % c.num_heads, 0u);
    EXPECT_NO_THROW(c.Validate());
}

TEST(TransformerConfig, DefaultsAreNeutralAndFailValidation) {
    const TransformerConfig c;

    EXPECT_EQ(c.vocab_size, 0u);
    EXPECT_EQ(c.d_model, 0u);
    EXPECT_THROW(c.Validate(), std::invalid_argument);
}

TEST(TransformerConfig, ValidateRejectsHeadsThatDoNotGroup) {
    TransformerConfig c = models::gpt2::Small();
    c.num_kv_heads = 5;   // 12 query heads cannot split over 5 KV heads
    EXPECT_THROW(c.Validate(), std::invalid_argument);
}

TEST(TransformerConfig, DerivedHeadCounts) {
    TransformerConfig c = models::gpt2::Small();
    EXPECT_EQ(c.KVHeads(), 12u);
    EXPECT_EQ(c.HeadDim(), 64u);

    c.num_kv_heads = 4;
    c.head_dim = 32;
    EXPECT_EQ(c.KVHeads(), 4u);
    EXPECT_EQ(c.HeadDim(), 32u);
}

// --- TransformerBlock: construction -----------------------------------------

TEST(TransformerBlockConstruction, BuildsAttentionWithTheGivenHeads) {
    TransformerBlock<float> block(16, 4, 64);

    EXPECT_EQ(block.attention->d_model, 16u);
    EXPECT_EQ(block.attention->num_heads, 4u);
    EXPECT_EQ(block.attention->head_dim, 4u);
}

TEST(TransformerBlockConstruction, FeedForwardExpandsThenProjectsBack) {
    TransformerBlock<float> block(16, 4, 64);

    EXPECT_EQ(block.mlp->Get<0>()->weight->Shape(), (std::vector<size_t>{64, 16}));
    EXPECT_EQ(block.mlp->Get<2>()->weight->Shape(), (std::vector<size_t>{16, 64}));
}

TEST(TransformerBlockConstruction, LayerNormsNormaliseTheModelWidth) {
    TransformerBlock<float> block(16, 4, 64, 1e-6f);

    EXPECT_EQ(block.ln_1->NormalizedShape(), (std::vector<size_t>{16}));
    EXPECT_EQ(block.ln_2->NormalizedShape(), (std::vector<size_t>{16}));
    EXPECT_FLOAT_EQ(block.ln_1->Epsilon(), 1e-6f);
    EXPECT_FLOAT_EQ(block.ln_2->Epsilon(), 1e-6f);
}

TEST(TransformerBlockConstruction, EpsilonDefaultsToGPT2s) {
    TransformerBlock<float> block(16, 4, 64);
    EXPECT_FLOAT_EQ(block.ln_1->Epsilon(), 1e-5f);
}

TEST(TransformerBlockConstruction, RejectsHeadsThatDoNotDivideTheModelWidth) {
    EXPECT_THROW(TransformerBlock<float>(16, 5, 64), std::invalid_argument);
}

// --- TransformerBlock: parameters -------------------------------------------

TEST(TransformerBlockParameters, AreNamedAfterGPT2sLayout) {
    TransformerBlock<float> block(16, 4, 64);

    EXPECT_EQ(NamesOf(block), (std::vector<std::string>{
        "ln_1.gamma", "ln_1.beta",
        "attn.query.weight", "attn.query.bias",
        "attn.key.weight", "attn.key.bias",
        "attn.value.weight", "attn.value.bias",
        "attn.output_proj.weight", "attn.output_proj.bias",
        "ln_2.gamma", "ln_2.beta",
        "mlp.0.weight", "mlp.0.bias",
        "mlp.2.weight", "mlp.2.bias"}));
}

TEST(TransformerBlockParameters, HaveTheExpectedElementCount) {
    const size_t d = 16, ff = 64;
    TransformerBlock<float> block(d, 4, ff);

    const size_t norms = 2 * (2 * d);
    const size_t attention = 4 * (d * d + d);
    const size_t mlp = (d * ff + ff) + (ff * d + d);
    EXPECT_EQ(ElementCount(block), norms + attention + mlp);
}

TEST(TransformerBlockParameters, ReachTheLayersTensors) {
    TransformerBlock<float> block(16, 4, 64);
    const auto params = block.Parameters();

    for (const auto& expected : {block.ln_1->gamma, block.attention->query->weight,
                                 block.ln_2->beta, block.mlp->Get<2>()->bias})
        EXPECT_NE(std::find(params.begin(), params.end(), expected), params.end());
}

// --- TransformerBlock: forward ----------------------------------------------

TEST(TransformerBlockForward, PreservesShape) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);

    EXPECT_EQ(block.Forward(Wave<float>({2, 5, 16})).Shape(), (std::vector<size_t>{2, 5, 16}));
}

TEST(TransformerBlockForward, AFreshBlockIsTheIdentity) {
    // Every Linear starts at zero, so both residual branches contribute zero.
    TransformerBlock<float> block(16, 4, 64);
    const Tensor<float> x = Wave<float>({1, 4, 16});

    EXPECT_EQ(MaxAbsDiff(block.Forward(x), x), 0.0);
}

TEST(TransformerBlockForward, IsPreNormWithTwoResiduals) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);
    const Tensor<float> x = Wave<float>({2, 4, 16});

    const Tensor<float> h = x + block.attention->Forward(block.ln_1->Forward(x), Tensor<bool>(), true);
    const Tensor<float> expected = h + block.mlp->Forward(block.ln_2->Forward(h));

    EXPECT_LT(MaxAbsDiff(block.Forward(x, Tensor<bool>(), true), expected), kTol);
}

TEST(TransformerBlockForward, IsNotPostNorm) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);
    const Tensor<float> x = Wave<float>({1, 4, 16});

    const Tensor<float> h = block.ln_1->Forward(x + block.attention->Forward(x));
    const Tensor<float> post_norm = block.ln_2->Forward(h + block.mlp->Forward(h));

    EXPECT_GT(MaxAbsDiff(block.Forward(x), post_norm), 1e-3);
}

TEST(TransformerBlockForward, OnlyTheAttentionBranchWhenTheMLPIsZero) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);
    block.mlp->Get<2>()->weight->Fill(0.0f);
    block.mlp->Get<2>()->bias->Fill(0.0f);
    const Tensor<float> x = Wave<float>({1, 4, 16});

    const Tensor<float> expected = x + block.attention->Forward(block.ln_1->Forward(x));

    EXPECT_LT(MaxAbsDiff(block.Forward(x), expected), kTol);
}

TEST(TransformerBlockForward, OnlyTheMLPBranchWhenAttentionIsZero) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);
    block.attention->output_proj->weight->Fill(0.0f);
    block.attention->output_proj->bias->Fill(0.0f);
    const Tensor<float> x = Wave<float>({1, 4, 16});

    const Tensor<float> expected = x + block.mlp->Forward(block.ln_2->Forward(x));

    EXPECT_LT(MaxAbsDiff(block.Forward(x), expected), kTol);
}

TEST(TransformerBlockForward, DefaultsToBidirectional) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);
    const Tensor<float> x = Wave<float>({1, 4, 16});

    EXPECT_EQ(MaxAbsDiff(block.Forward(x), block.Forward(x, Tensor<bool>(), false)), 0.0);
}

TEST(TransformerBlockForward, CausalOutputsIgnoreLaterTokens) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);
    Tensor<float> a = Wave<float>({1, 4, 16});
    Tensor<float> b = a.Clone();
    for (size_t d = 0; d < 16; ++d) b(0, 3, d) = 5.0f;

    const Tensor<float> out_a = block.Forward(a, Tensor<bool>(), true);
    const Tensor<float> out_b = block.Forward(b, Tensor<bool>(), true);

    for (size_t s = 0; s < 3; ++s) EXPECT_LT(PositionDiff(out_a, out_b, s), 1e-5) << "position " << s;
    EXPECT_GT(PositionDiff(out_a, out_b, 3), 1e-3);
}

TEST(TransformerBlockForward, BidirectionalOutputsSeeLaterTokens) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);
    Tensor<float> a = Wave<float>({1, 4, 16});
    Tensor<float> b = a.Clone();
    for (size_t d = 0; d < 16; ++d) b(0, 3, d) = 5.0f;

    EXPECT_GT(PositionDiff(block.Forward(a), block.Forward(b), 0), 1e-4);
}

TEST(TransformerBlockForward, AMaskedKeyHasNoInfluence) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);
    Tensor<float> a = Wave<float>({1, 4, 16});
    Tensor<float> b = a.Clone();
    for (size_t d = 0; d < 16; ++d) b(0, 3, d) = 5.0f;
    Tensor<bool> mask({4, 4}, false);
    for (size_t i = 0; i < 4; ++i) mask(i, 3) = true;

    const Tensor<float> out_a = block.Forward(a, mask);
    const Tensor<float> out_b = block.Forward(b, mask);

    for (size_t s = 0; s < 3; ++s) EXPECT_LT(PositionDiff(out_a, out_b, s), 1e-5) << "position " << s;
}

TEST(TransformerBlockForward, BatchesAreIndependent) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);
    const Tensor<float> one = Wave<float>({1, 3, 16});
    Tensor<float> two({2, 3, 16}, 3.0f);
    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 16; ++d) two(0, s, d) = one(0, s, d);

    const Tensor<float> single = block.Forward(one);
    const Tensor<float> batched = block.Forward(two);

    for (size_t s = 0; s < 3; ++s)
        for (size_t d = 0; d < 16; ++d) EXPECT_NEAR(batched(0, s, d), single(0, s, d), kTol);
}

TEST(TransformerBlockForward, DoesNotMutateTheInput) {
    TransformerBlock<float> block(16, 4, 64);
    SeedParameters(block);
    const Tensor<float> x = Wave<float>({1, 4, 16});
    const Tensor<float> before = x.Clone();

    const Tensor<float> out = block.Forward(x);

    EXPECT_FALSE(out.SharesStorage(x));
    EXPECT_EQ(MaxAbsDiff(x, before), 0.0);
}

TEST(TransformerBlockForward, RejectsTheWrongModelWidth) {
    TransformerBlock<float> block(16, 4, 64);
    EXPECT_THROW(block.Forward(Wave<float>({1, 4, 12})), std::invalid_argument);
}

TEST(TransformerBlockForward, WorksWithDouble) {
    TransformerBlock<double> block(16, 4, 64);
    SeedParameters(block);

    const Tensor<double> out = block.Forward(Wave<double>({1, 3, 16}), Tensor<bool>(), true);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{1, 3, 16}));
    for (size_t i = 0; i < out.Size(); ++i) EXPECT_FALSE(std::isnan(out.Contiguous().Data()[i]));
}

// --- Transformer: construction ----------------------------------------------

TEST(TransformerConstruction, BuildsOneBlockPerLayer) {
    Transformer<float> model(SmallConfig(3));
    EXPECT_EQ(model.blocks.size(), 3u);
}

TEST(TransformerConstruction, EveryBlockHasItsOwnWeights) {
    Transformer<float> model(SmallConfig(3));

    std::set<const void*> seen;
    for (const auto& block : model.blocks) {
        EXPECT_TRUE(seen.insert(block.get()).second);
        EXPECT_TRUE(seen.insert(block->attention->query->weight.get()).second);
        EXPECT_TRUE(seen.insert(block->ln_1->gamma.get()).second);
    }
}

TEST(TransformerConstruction, BlocksFollowTheConfig) {
    Transformer<float> model(SmallConfig());

    for (const auto& block : model.blocks) {
        EXPECT_EQ(block->attention->num_heads, 4u);
        EXPECT_EQ(block->attention->head_dim, 4u);
        EXPECT_EQ(block->mlp->Get<0>()->weight->Shape(), (std::vector<size_t>{64, 16}));
    }
}

TEST(TransformerConstruction, NormsUseTheConfiguredEpsilon) {
    TransformerConfig c = SmallConfig();
    c.norm_eps = 1e-6f;
    Transformer<float> model(c);

    EXPECT_FLOAT_EQ(model.ln_f->Epsilon(), 1e-6f);
    EXPECT_FLOAT_EQ(model.blocks[0]->ln_1->Epsilon(), 1e-6f);
    EXPECT_FLOAT_EQ(model.blocks[1]->ln_2->Epsilon(), 1e-6f);
}

TEST(TransformerConstruction, KeepsTheConfig) {
    Transformer<float> model(SmallConfig());
    EXPECT_EQ(model.config.d_model, 16u);
    EXPECT_EQ(model.config.num_layers, 2u);
}

TEST(TransformerConstruction, RejectsHeadsThatDoNotDivideTheModelWidth) {
    TransformerConfig c = SmallConfig();
    c.num_heads = 5;

    EXPECT_THROW(Transformer<float>{c}, std::invalid_argument);
}

// --- Transformer: parameters ------------------------------------------------

TEST(TransformerParameters, AreNamedByLayerIndex) {
    Transformer<float> model(SmallConfig(2));
    const auto names = NamesOf(model);

    ASSERT_EQ(names.size(), 2 * 16 + 2);
    EXPECT_EQ(names.front(), "h.0.ln_1.gamma");
    EXPECT_EQ(names[16], "h.1.ln_1.gamma");
    EXPECT_EQ(names[names.size() - 3], "h.1.mlp.2.bias");
    EXPECT_EQ(names[names.size() - 2], "ln_f.gamma");
    EXPECT_EQ(names.back(), "ln_f.beta");
}

TEST(TransformerParameters, NamesAreUnique) {
    Transformer<float> model(SmallConfig(3));
    const auto names = NamesOf(model);

    EXPECT_EQ(std::set<std::string>(names.begin(), names.end()).size(), names.size());
}

TEST(TransformerParameters, GPT2SmallHasTheExpectedSize) {
    // GPT-2 small is 124,439,808 parameters; without the token (50257 x 768)
    // and position (1024 x 768) embeddings that leaves 85,056,000.
    Transformer<float> model{models::gpt2::Small()};

    EXPECT_EQ(model.blocks.size(), 12u);
    EXPECT_EQ(model.NamedParameters().size(), 12u * 16u + 2u);
    EXPECT_EQ(ElementCount(model), 85'056'000u);
}

// --- Transformer: forward ---------------------------------------------------

TEST(TransformerForward, PreservesShape) {
    Transformer<float> model(SmallConfig());
    SeedParameters(model);

    EXPECT_EQ(model.Forward(Wave<float>({2, 5, 16})).Shape(), (std::vector<size_t>{2, 5, 16}));
}

TEST(TransformerForward, AFreshModelIsJustTheFinalNorm) {
    // Fresh blocks are identities, so only ln_f acts.
    Transformer<float> model(SmallConfig());
    const Tensor<float> x = Wave<float>({1, 4, 16});

    EXPECT_LT(MaxAbsDiff(model.Forward(x), model.ln_f->Forward(x)), kTol);
}

TEST(TransformerForward, ChainsEveryBlockThenTheFinalNorm) {
    Transformer<float> model(SmallConfig(3));
    SeedParameters(model);
    const Tensor<float> x = Wave<float>({2, 4, 16});

    Tensor<float> h = x;
    for (const auto& block : model.blocks) h = block->Forward(h, Tensor<bool>(), true);
    const Tensor<float> expected = model.ln_f->Forward(h);

    EXPECT_LT(MaxAbsDiff(model.Forward(x), expected), kTol);
}

TEST(TransformerForward, EveryLayerContributes) {
    Transformer<float> model(SmallConfig(3));
    SeedParameters(model);
    const Tensor<float> x = Wave<float>({1, 4, 16});
    const Tensor<float> before = model.Forward(x);

    model.blocks[2]->mlp->Get<2>()->bias->Fill(1.0f);

    EXPECT_GT(MaxAbsDiff(model.Forward(x), before), 1e-3);
}

TEST(TransformerForward, WithNoLayersIsTheFinalNorm) {
    Transformer<float> model(SmallConfig(0));
    SeedParameters(model);
    const Tensor<float> x = Wave<float>({1, 3, 16});

    EXPECT_TRUE(model.blocks.empty());
    EXPECT_LT(MaxAbsDiff(model.Forward(x), model.ln_f->Forward(x)), kTol);
}

TEST(TransformerForward, IsCausalByDefault) {
    Transformer<float> model(SmallConfig(3));
    SeedParameters(model);
    Tensor<float> a = Wave<float>({1, 5, 16});
    Tensor<float> b = a.Clone();
    for (size_t d = 0; d < 16; ++d) b(0, 4, d) = 5.0f;

    const Tensor<float> out_a = model.Forward(a);
    const Tensor<float> out_b = model.Forward(b);

    for (size_t s = 0; s < 4; ++s) EXPECT_LT(PositionDiff(out_a, out_b, s), 1e-5) << "position " << s;
    EXPECT_GT(PositionDiff(out_a, out_b, 4), 1e-3);
}

TEST(TransformerForward, CanRunBidirectionally) {
    Transformer<float> model(SmallConfig(2));
    SeedParameters(model);
    Tensor<float> a = Wave<float>({1, 4, 16});
    Tensor<float> b = a.Clone();
    for (size_t d = 0; d < 16; ++d) b(0, 3, d) = 5.0f;

    EXPECT_GT(PositionDiff(model.Forward(a, Tensor<bool>(), false), model.Forward(b, Tensor<bool>(), false), 0), 1e-4);
}

TEST(TransformerForward, PassesTheMaskToEveryBlock) {
    Transformer<float> model(SmallConfig(2));
    SeedParameters(model);
    Tensor<float> a = Wave<float>({1, 4, 16});
    Tensor<float> b = a.Clone();
    for (size_t d = 0; d < 16; ++d) b(0, 2, d) = 5.0f;
    Tensor<bool> mask({4, 4}, false);
    for (size_t i = 0; i < 4; ++i) mask(i, 2) = true;
    mask(2, 2) = false;   // position 2 still sees itself, so its row is not all masked

    const Tensor<float> out_a = model.Forward(a, mask, false);
    const Tensor<float> out_b = model.Forward(b, mask, false);

    for (size_t s : {size_t{0}, size_t{1}, size_t{3}})
        EXPECT_LT(PositionDiff(out_a, out_b, s), 1e-5) << "position " << s;
}

TEST(TransformerForward, NoNaNForGPT2ShapedInput) {
    TransformerConfig c = SmallConfig(4);
    Transformer<float> model(c);
    SeedParameters(model);

    const Tensor<float> out = model.Forward(Wave<float>({2, 8, 16}));
    const Tensor<float> flat = out.Contiguous();

    for (size_t i = 0; i < flat.Size(); ++i) ASSERT_FALSE(std::isnan(flat.Data()[i])) << "at " << i;
}

TEST(TransformerModule, ReportsParametersThroughABasePointer) {
    std::shared_ptr<Module<float>> m = std::make_shared<Transformer<float>>(SmallConfig(2));
    EXPECT_EQ(m->Parameters().size(), 2u * 16u + 2u);
}
