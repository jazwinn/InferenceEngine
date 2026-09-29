#include <gtest/gtest.h>

#include <NeuralNet/Transformer/Decoder.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using math::Tensor;
using nn::transformer::Decoder;
using nn::transformer::DecoderBlock;
using nn::transformer::RotaryEmbedding;
using nn::transformer::KVCache;
using nn::transformer::config::Activation;
using nn::transformer::config::MLPType;
using nn::transformer::config::NormPlacement;
using nn::transformer::config::NormType;
using nn::transformer::config::PositionEncoding;
using nn::transformer::config::TransformerConfig;

namespace {

constexpr float kTol = 1e-4f;
constexpr size_t kBatch = 2;
constexpr size_t kSeq = 5;
constexpr size_t kModel = 16;
constexpr size_t kLayers = 3;

TransformerConfig LlamaStyle() {
    TransformerConfig c;
    c.vocab_size = 32;
    c.d_model = kModel;
    c.num_layers = kLayers;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.ffn_dim = 24;
    c.max_seq_len = 8;
    c.norm_type = NormType::RMSNorm;
    c.mlp_type = MLPType::Gated;
    c.activation = Activation::SiLU;
    c.position = PositionEncoding::Rotary;
    c.attention_bias = false;
    c.output_bias = false;
    return c;
}

TransformerConfig GPT2Style() {
    TransformerConfig c = LlamaStyle();
    c.num_kv_heads = 0;
    c.norm_type = NormType::LayerNorm;
    c.mlp_type = MLPType::Standard;
    c.activation = Activation::GELUTanh;
    c.position = PositionEncoding::Learned;
    c.attention_bias = true;
    c.output_bias = true;
    return c;
}

void Seed(Decoder<float>& decoder) {
    double phase = 0.1;
    for (const auto& p : decoder.NamedParameters()) {
        float* data = p.tensor->Data();
        for (size_t i = 0; i < p.tensor->Size(); ++i)
            data[i] = static_cast<float>(0.4 * std::sin(phase + 0.37 * static_cast<double>(i)));
        phase += 0.7;
    }
}

Tensor<float> Input() {
    Tensor<float> x({kBatch, kSeq, kModel});
    float* data = x.Data();
    for (size_t i = 0; i < x.Size(); ++i) data[i] = static_cast<float>(0.5 * std::sin(1.3 + 0.21 * static_cast<double>(i)));
    return x;
}

void ExpectNear(const Tensor<float>& actual, const Tensor<float>& expected) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    Tensor<float> a = actual.Contiguous();
    Tensor<float> e = expected.Contiguous();
    for (size_t i = 0; i < a.Size(); ++i) EXPECT_NEAR(a.Data()[i], e.Data()[i], kTol) << "at flat index " << i;
}

void SeedModule(nn::Module<float>& module) {
    double phase = 0.3;
    for (const auto& p : module.NamedParameters()) {
        float* data = p.tensor->Data();
        for (size_t i = 0; i < p.tensor->Size(); ++i)
            data[i] = static_cast<float>(0.4 * std::sin(phase + 0.29 * static_cast<double>(i)));
        phase += 0.9;
    }
}

float MaxDiff(const Tensor<float>& lhs, const Tensor<float>& rhs) {
    Tensor<float> a = lhs.Contiguous();
    Tensor<float> b = rhs.Contiguous();
    float diff = 0.0f;
    for (size_t i = 0; i < a.Size(); ++i) diff = std::fmax(diff, std::fabs(a.Data()[i] - b.Data()[i]));
    return diff;
}

std::shared_ptr<RotaryEmbedding<float>> RopeFor(const TransformerConfig& cfg) {
    return std::make_shared<RotaryEmbedding<float>>(cfg.HeadDim(), cfg.max_seq_len, cfg.rope_theta);
}

}  // namespace

TEST(DecoderConstruction, BuildsOneBlockPerLayer) {
    Decoder<float> decoder(LlamaStyle());

    ASSERT_EQ(decoder.blocks.size(), kLayers);
    for (const auto& block : decoder.blocks) EXPECT_TRUE(block);
}

TEST(DecoderConstruction, RegistersBlocksUnderLayerIndex) {
    Decoder<float> decoder(LlamaStyle());

    bool saw_last_layer = false, saw_final_norm = false;
    for (const auto& p : decoder.NamedParameters()) {
        saw_last_layer |= p.name.rfind("h." + std::to_string(kLayers - 1) + ".", 0) == 0;
        saw_final_norm |= p.name.rfind("finalNorm.", 0) == 0;
    }
    EXPECT_TRUE(saw_last_layer);
    EXPECT_TRUE(saw_final_norm);
}

TEST(DecoderConstruction, RotaryConfigSharesOneTableAcrossBlocks) {
    Decoder<float> decoder(LlamaStyle());

    ASSERT_TRUE(decoder.rope);
    for (const auto& block : decoder.blocks) EXPECT_EQ(block->attention->RoPE, decoder.rope);
    EXPECT_EQ(decoder.rope.use_count(), static_cast<long>(kLayers + 1));
}

TEST(DecoderConstruction, LearnedPositionsHaveNoRotaryTable) {
    Decoder<float> decoder(GPT2Style());

    EXPECT_FALSE(decoder.rope);
    for (const auto& block : decoder.blocks) EXPECT_FALSE(block->attention->RoPE);
}

TEST(DecoderConstruction, FinalNormOnlyForPreNorm) {
    TransformerConfig post = LlamaStyle();
    post.norm_placement = NormPlacement::Post;

    EXPECT_TRUE(Decoder<float>(LlamaStyle()).finalNorm);
    EXPECT_FALSE(Decoder<float>(post).finalNorm);
}

TEST(DecoderConstruction, ParameterCountMatchesConfig) {
    Decoder<float> decoder(LlamaStyle());

    const size_t head_dim = kModel / 4;
    const size_t per_block = kModel * (4 * head_dim) + 2 * kModel * (2 * head_dim) + (4 * head_dim) * kModel
                             + 3 * kModel * 24 + 2 * kModel;
    size_t total = 0;
    for (const auto& p : decoder.NamedParameters()) total += p.tensor->Size();

    EXPECT_EQ(total, kLayers * per_block + kModel);
}

TEST(DecoderConstruction, RejectsInvalidConfigInsteadOfCrashing) {
    TransformerConfig bad = LlamaStyle();
    bad.num_heads = 0;
    EXPECT_THROW(Decoder<float>{bad}, std::invalid_argument);

    TransformerConfig bad_kv = LlamaStyle();
    bad_kv.num_kv_heads = 3;
    EXPECT_THROW(Decoder<float>{bad_kv}, std::invalid_argument);
}

TEST(DecoderForward, KeepsInputShapeForBothStyles) {
    for (const TransformerConfig& cfg : {LlamaStyle(), GPT2Style()}) {
        Decoder<float> decoder(cfg);
        Seed(decoder);
        EXPECT_EQ(decoder.Forward(Input()).Shape(), (std::vector<size_t>{kBatch, kSeq, kModel}));
    }
}

TEST(DecoderForward, ChainsBlocksThenFinalNorm) {
    Decoder<float> decoder(LlamaStyle());
    Seed(decoder);
    const Tensor<float> x = Input();

    Tensor<float> expected = x;
    for (const auto& block : decoder.blocks) expected = block->Forward(expected, nullptr, 0, Tensor<bool>(), true);
    expected = decoder.finalNorm->Forward(expected);

    ExpectNear(decoder.Forward(x), expected);
}

TEST(DecoderForward, CachedPrefillThenDecodeMatchesFullPass) {
    for (const TransformerConfig& cfg : {LlamaStyle(), GPT2Style()}) {
        Decoder<float> decoder(cfg);
        Seed(decoder);
        const Tensor<float> x = Input();
        const Tensor<float> full = decoder.Forward(x);

        KVCache<float> cache(kLayers, kBatch, cfg.KVHeads(), cfg.max_seq_len, cfg.HeadDim());
        ExpectNear(decoder.Forward(x.Slice(1, 0, 2), &cache), full.Slice(1, 0, 2));
        EXPECT_EQ(cache.Length(), 2u);

        for (size_t t = 2; t < kSeq; ++t) {
            SCOPED_TRACE(t);
            ExpectNear(decoder.Forward(x.Slice(1, t, 1), &cache), full.Slice(1, t, 1));
            EXPECT_EQ(cache.Length(), t + 1);
        }
    }
}

TEST(DecoderForward, CausalFlagIsPassedToEveryBlock) {
    Decoder<float> decoder(LlamaStyle());
    Seed(decoder);
    const Tensor<float> x = Input();

    const Tensor<float> causal = decoder.Forward(x, nullptr, Tensor<bool>(), true).Contiguous();
    const Tensor<float> open = decoder.Forward(x, nullptr, Tensor<bool>(), false).Contiguous();

    float diff = 0.0f;
    for (size_t i = 0; i < causal.Size(); ++i) diff = std::fmax(diff, std::fabs(causal.Data()[i] - open.Data()[i]));
    EXPECT_GT(diff, 1e-3f);
}

TEST(DecoderBlockConstruction, RegistersFourNamedParts) {
    DecoderBlock<float> block(GPT2Style());

    std::vector<std::string> names;
    for (const auto& p : block.NamedParameters()) names.push_back(p.name.substr(0, p.name.find('.')));
    for (const std::string part : {"attention", "attenNorm", "mlp", "mlpNorm"})
        EXPECT_NE(std::find(names.begin(), names.end(), part), names.end()) << part;
}

TEST(DecoderBlockConstruction, PassesRotaryTableToAttention) {
    const TransformerConfig cfg = LlamaStyle();
    auto rope = RopeFor(cfg);
    DecoderBlock<float> block(cfg, rope);

    EXPECT_EQ(block.attention->RoPE, rope);
}

TEST(DecoderBlockConstruction, BuildsPartsFromTheConfig) {
    DecoderBlock<float> llama(LlamaStyle(), RopeFor(LlamaStyle()));
    DecoderBlock<float> gpt2(GPT2Style());

    EXPECT_TRUE(dynamic_cast<nn::transformer::norm::RMSNorm<float>*>(llama.attenNorm.get()));
    EXPECT_TRUE(dynamic_cast<nn::transformer::norm::LayerNorm<float>*>(gpt2.attenNorm.get()));
    EXPECT_TRUE(llama.mlp->gate);
    EXPECT_FALSE(gpt2.mlp->gate);
    EXPECT_EQ(llama.attention->numHeadsKV, 2u);
    EXPECT_EQ(gpt2.attention->numHeadsKV, 4u);
}

TEST(DecoderBlockForward, FreshPreNormBlockIsTheIdentity) {
    DecoderBlock<float> block(GPT2Style());
    const Tensor<float> x = Input();

    ExpectNear(block.Forward(x), x);
}

TEST(DecoderBlockForward, PreNormAddsBothResiduals) {
    DecoderBlock<float> block(LlamaStyle(), RopeFor(LlamaStyle()));
    SeedModule(block);
    const Tensor<float> x = Input();

    const Tensor<float> h = x + block.attention->Forward(block.attenNorm->Forward(x), nullptr, 0, Tensor<bool>(), true);
    const Tensor<float> expected = h + block.mlp->Forward(block.mlpNorm->Forward(h));

    ExpectNear(block.Forward(x), expected);
}

TEST(DecoderBlockForward, PostNormNormalisesAfterEachResidual) {
    TransformerConfig cfg = GPT2Style();
    cfg.norm_placement = NormPlacement::Post;
    DecoderBlock<float> block(cfg);
    SeedModule(block);
    const Tensor<float> x = Input();

    const Tensor<float> h = block.attenNorm->Forward(x + block.attention->Forward(x, nullptr, 0, Tensor<bool>(), true));
    const Tensor<float> expected = block.mlpNorm->Forward(h + block.mlp->Forward(h));

    ExpectNear(block.Forward(x), expected);
}

TEST(DecoderBlockForward, PreAndPostPlacementDiffer) {
    TransformerConfig post = GPT2Style();
    post.norm_placement = NormPlacement::Post;
    DecoderBlock<float> pre_block(GPT2Style());
    DecoderBlock<float> post_block(post);
    SeedModule(pre_block);
    SeedModule(post_block);
    const Tensor<float> x = Input();

    EXPECT_GT(MaxDiff(pre_block.Forward(x), post_block.Forward(x)), 1e-3f);
}

TEST(DecoderBlockForward, IsCausalByDefault) {
    DecoderBlock<float> block(LlamaStyle(), RopeFor(LlamaStyle()));
    SeedModule(block);
    Tensor<float> x = Input();
    const Tensor<float> before = block.Forward(x);

    for (size_t b = 0; b < kBatch; ++b)
        for (size_t d = 0; d < kModel; ++d) x(b, kSeq - 1, d) += 0.3f * static_cast<float>(d) - 1.0f;
    const Tensor<float> after = block.Forward(x);

    ExpectNear(after.Slice(1, 0, kSeq - 1), before.Slice(1, 0, kSeq - 1));
    EXPECT_GT(MaxDiff(after.Slice(1, kSeq - 1, 1), before.Slice(1, kSeq - 1, 1)), 1e-3f);
}

TEST(DecoderBlockForward, BidirectionalSeesLaterTokens) {
    DecoderBlock<float> block(GPT2Style());
    SeedModule(block);
    Tensor<float> x = Input();
    const Tensor<float> before = block.Forward(x, nullptr, 0, Tensor<bool>(), false);

    for (size_t b = 0; b < kBatch; ++b)
        for (size_t d = 0; d < kModel; ++d) x(b, kSeq - 1, d) += 0.3f * static_cast<float>(d) - 1.0f;
    const Tensor<float> after = block.Forward(x, nullptr, 0, Tensor<bool>(), false);

    EXPECT_GT(MaxDiff(after.Slice(1, 0, 1), before.Slice(1, 0, 1)), 1e-4f);
}

TEST(DecoderBlockForward, MaskedKeyHasNoInfluence) {
    DecoderBlock<float> block(GPT2Style());
    SeedModule(block);
    Tensor<float> x = Input();
    Tensor<bool> mask({kSeq, kSeq}, false);
    for (size_t q = 0; q < kSeq; ++q) mask(q, 2) = true;

    const Tensor<float> before = block.Forward(x, nullptr, 0, mask, false);
    for (size_t b = 0; b < kBatch; ++b)
        for (size_t d = 0; d < kModel; ++d) x(b, 2, d) += 0.3f * static_cast<float>(d) - 1.0f;
    const Tensor<float> after = block.Forward(x, nullptr, 0, mask, false);

    for (size_t s : {0u, 1u, 3u, 4u}) ExpectNear(after.Slice(1, s, 1), before.Slice(1, s, 1));
}

TEST(DecoderBlockForward, BatchesAreIndependent) {
    DecoderBlock<float> block(LlamaStyle(), RopeFor(LlamaStyle()));
    SeedModule(block);
    Tensor<float> x = Input();
    const Tensor<float> before = block.Forward(x);

    for (size_t s = 0; s < kSeq; ++s)
        for (size_t d = 0; d < kModel; ++d) x(1, s, d) += 1.0f;
    const Tensor<float> after = block.Forward(x);

    ExpectNear(after.Slice(0, 0, 1), before.Slice(0, 0, 1));
}

TEST(DecoderBlockForward, DoesNotMutateTheInput) {
    DecoderBlock<float> block(LlamaStyle(), RopeFor(LlamaStyle()));
    SeedModule(block);
    const Tensor<float> x = Input();
    const Tensor<float> copy = x.Clone();

    block.Forward(x);

    ExpectNear(x, copy);
}
