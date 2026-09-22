#include <gtest/gtest.h>

#include <NeuralNet/Transformer/KVCache.h>
#include <NeuralNet/Transformer/MultiHeadAttention.h>
#include <NeuralNet/Transformer/Transformer.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using math::Tensor;
using nn::Module;
using transformer::KVCache;
using transformer::MultiHeadAttention;
using transformer::Transformer;
using transformer::TransformerBlock;
using transformer::TransformerConfig;

namespace {

constexpr size_t kDModel  = 8;
constexpr size_t kHeads   = 2;
constexpr size_t kHeadDim = kDModel / kHeads;
constexpr size_t kMaxSeq  = 6;
constexpr size_t kLayers  = 3;   // > 2 so a per-layer Advance bug is unmistakable

// Cached and uncached decoding run the same operations over the same values in
// the same order, so the results should be bit-identical. The bound is a few
// ULP rather than exactly zero only to stay robust to FMA contraction.
constexpr double kExact = 1e-6;

TransformerConfig TinyConfig() {
    TransformerConfig c;
    c.vocab_size  = 11;
    c.d_model     = kDModel;
    c.num_layers  = kLayers;
    c.num_heads   = kHeads;
    c.ffn_dim     = 16;
    c.max_seq_len = kMaxSeq;
    return c;
}

// Every Linear starts at zero, which makes attention output all zeros and every
// equivalence assertion below hold vacuously. Seed deterministically instead.
template <typename T>
void Seed(Module<T>& module) {
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
        data[i] = static_cast<T>(0.4 * std::sin(0.9 * static_cast<double>(i + 1) + phase));
    return t;
}

// Positions [first, first + count) of a {B, S, D} tensor, as its own tensor --
// the prompt slice or the single token a decode step feeds in.
Tensor<float> Window(const Tensor<float>& x, size_t first, size_t count) {
    Tensor<float> out({x.Shape()[0], count, x.Shape()[2]});
    for (size_t b = 0; b < x.Shape()[0]; ++b)
        for (size_t s = 0; s < count; ++s)
            for (size_t d = 0; d < x.Shape()[2]; ++d) out(b, s, d) = x(b, first + s, d);
    return out;
}

// Worst difference between position `sa` of `a` and position `sb` of `b`.
double RowDiff(const Tensor<float>& a, size_t sa, const Tensor<float>& b, size_t sb) {
    double worst = 0.0;
    for (size_t bt = 0; bt < a.Shape()[0]; ++bt)
        for (size_t d = 0; d < a.Shape()[2]; ++d)
            worst = std::max(worst, std::abs(static_cast<double>(a(bt, sa, d)) -
                                             static_cast<double>(b(bt, sb, d))));
    return worst;
}

double MaxAbsDiff(const Tensor<float>& a, const Tensor<float>& b) {
    double worst = 0.0;
    for (size_t s = 0; s < a.Shape()[1]; ++s) worst = std::max(worst, RowDiff(a, s, b, s));
    return worst;
}

size_t NonZeroCount(const Tensor<float>& t, size_t position) {
    size_t count = 0;
    for (size_t b = 0; b < t.Shape()[0]; ++b)
        for (size_t h = 0; h < t.Shape()[1]; ++h)
            for (size_t d = 0; d < t.Shape()[3]; ++d)
                if (t(b, h, position, d) != 0.0f) ++count;
    return count;
}

}  // namespace

// --- MultiHeadAttention: the cache is optional -------------------------------

TEST(KVCacheAttention, ANullCacheLeavesTheResultUnchanged) {
    MultiHeadAttention<float> attn(kDModel, kHeads);
    Seed(attn);
    const Tensor<float> x = Wave<float>({2, 4, kDModel});

    const Tensor<float> a = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);
    const Tensor<float> b = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);

    EXPECT_EQ(a.Shape(), (std::vector<size_t>{2, 4, kDModel}));
    EXPECT_EQ(MaxAbsDiff(a, b), 0.0) << "the uncached path must stay deterministic";
}

TEST(KVCacheAttention, PassingACacheDoesNotChangeTheOutputShape) {
    MultiHeadAttention<float> attn(kDModel, kHeads);
    Seed(attn);
    KVCache<float> cache(1, 2, kHeads, kMaxSeq, kHeadDim);

    const Tensor<float> out =
        attn.Forward(Wave<float>({2, 3, kDModel}), &cache.Layer(0), 0, Tensor<bool>(), true);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{2, 3, kDModel}));
}

// --- MultiHeadAttention: equivalence ----------------------------------------

// The whole point of the cache. If this fails, nothing else matters.
TEST(KVCacheAttention, DecodingOneTokenAtATimeMatchesTheFullSequence) {
    MultiHeadAttention<float> attn(kDModel, kHeads);
    Seed(attn);

    constexpr size_t kBatch = 2, kSeq = 5;
    const Tensor<float> x = Wave<float>({kBatch, kSeq, kDModel});
    const Tensor<float> reference = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);

    KVCache<float> cache(1, kBatch, kHeads, kMaxSeq, kHeadDim);

    for (size_t s = 0; s < kSeq; ++s) {
        const size_t cache_len = cache.Room(1);
        const Tensor<float> out =
            attn.Forward(Window(x, s, 1), &cache.Layer(0), cache_len, Tensor<bool>(), true);
        cache.Advance(1);

        ASSERT_EQ(out.Shape(), (std::vector<size_t>{kBatch, 1, kDModel})) << "step " << s;
        EXPECT_LT(RowDiff(out, 0, reference, s), kExact) << "position " << s;
    }
}

TEST(KVCacheAttention, PrefillThenDecodeMatchesTheFullSequence) {
    MultiHeadAttention<float> attn(kDModel, kHeads);
    Seed(attn);

    constexpr size_t kSeq = 5, kPrompt = 3;
    const Tensor<float> x = Wave<float>({1, kSeq, kDModel});
    const Tensor<float> reference = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);

    KVCache<float> cache(1, 1, kHeads, kMaxSeq, kHeadDim);

    // Prefill and decode are the same call: only `seq` differs.
    const Tensor<float> prefill =
        attn.Forward(Window(x, 0, kPrompt), &cache.Layer(0), cache.Room(kPrompt),
                     Tensor<bool>(), true);
    cache.Advance(kPrompt);

    ASSERT_EQ(prefill.Shape(), (std::vector<size_t>{1, kPrompt, kDModel}));
    for (size_t s = 0; s < kPrompt; ++s)
        EXPECT_LT(RowDiff(prefill, s, reference, s), kExact) << "prefill position " << s;

    for (size_t s = kPrompt; s < kSeq; ++s) {
        const Tensor<float> out =
            attn.Forward(Window(x, s, 1), &cache.Layer(0), cache.Room(1), Tensor<bool>(), true);
        cache.Advance(1);
        EXPECT_LT(RowDiff(out, 0, reference, s), kExact) << "decode position " << s;
    }
}

TEST(KVCacheAttention, TwoTokensAtATimeMatchesTheFullSequence) {
    MultiHeadAttention<float> attn(kDModel, kHeads);
    Seed(attn);

    const Tensor<float> x = Wave<float>({1, 4, kDModel});
    const Tensor<float> reference = attn.Forward(x, nullptr, 0, Tensor<bool>(), true);

    KVCache<float> cache(1, 1, kHeads, kMaxSeq, kHeadDim);

    // A chunk of 2 exercises the causal mask with 1 < seq_q < seq_k, which
    // neither a full pass nor a single-token step reaches.
    for (size_t s = 0; s < 4; s += 2) {
        const Tensor<float> out =
            attn.Forward(Window(x, s, 2), &cache.Layer(0), cache.Room(2), Tensor<bool>(), true);
        cache.Advance(2);

        EXPECT_LT(RowDiff(out, 0, reference, s), kExact) << "position " << s;
        EXPECT_LT(RowDiff(out, 1, reference, s + 1), kExact) << "position " << s + 1;
    }
}

// --- MultiHeadAttention: what lands in the cache -----------------------------

TEST(KVCacheAttention, CacheHoldsExactlyTheProjectedKeysAndValues) {
    MultiHeadAttention<float> attn(kDModel, kHeads);
    Seed(attn);

    const Tensor<float> x = Wave<float>({1, 3, kDModel});
    KVCache<float> cache(1, 1, kHeads, kMaxSeq, kHeadDim);
    attn.Forward(x, &cache.Layer(0), 0, Tensor<bool>(), true);
    cache.Advance(3);

    const Tensor<float> K = attn.SplitHeads(attn.key->Forward(x));
    const Tensor<float> V = attn.SplitHeads(attn.value->Forward(x));

    for (size_t h = 0; h < kHeads; ++h)
        for (size_t s = 0; s < 3; ++s)
            for (size_t d = 0; d < kHeadDim; ++d) {
                EXPECT_FLOAT_EQ(cache.Layer(0).k(0, h, s, d), K(0, h, s, d))
                    << "K at h" << h << " s" << s << " d" << d;
                EXPECT_FLOAT_EQ(cache.Layer(0).v(0, h, s, d), V(0, h, s, d))
                    << "V at h" << h << " s" << s << " d" << d;
            }
}

TEST(KVCacheAttention, KeysAndValuesGoToSeparateBuffers) {
    MultiHeadAttention<float> attn(kDModel, kHeads);
    Seed(attn);

    KVCache<float> cache(1, 1, kHeads, kMaxSeq, kHeadDim);
    attn.Forward(Wave<float>({1, 2, kDModel}), &cache.Layer(0), 0, Tensor<bool>(), true);

    // Swapping k and v in the append would make these identical.
    double worst = 0.0;
    for (size_t h = 0; h < kHeads; ++h)
        for (size_t s = 0; s < 2; ++s)
            for (size_t d = 0; d < kHeadDim; ++d)
                worst = std::max(worst, std::abs(static_cast<double>(cache.Layer(0).k(0, h, s, d)) -
                                                 static_cast<double>(cache.Layer(0).v(0, h, s, d))));
    EXPECT_GT(worst, 1e-3) << "K and V must not be written to the same buffer";
}

TEST(KVCacheAttention, AppendingLeavesTheUnusedTailZero) {
    MultiHeadAttention<float> attn(kDModel, kHeads);
    Seed(attn);

    KVCache<float> cache(1, 1, kHeads, kMaxSeq, kHeadDim);
    attn.Forward(Wave<float>({1, 2, kDModel}), &cache.Layer(0), 0, Tensor<bool>(), true);
    cache.Advance(2);

    for (size_t s = cache.Length(); s < kMaxSeq; ++s) {
        EXPECT_EQ(NonZeroCount(cache.Layer(0).k, s), 0u) << "K spilled into position " << s;
        EXPECT_EQ(NonZeroCount(cache.Layer(0).v, s), 0u) << "V spilled into position " << s;
    }
}

TEST(KVCacheAttention, AppendingAtAnOffsetDoesNotDisturbEarlierPositions) {
    MultiHeadAttention<float> attn(kDModel, kHeads);
    Seed(attn);

    KVCache<float> cache(1, 1, kHeads, kMaxSeq, kHeadDim);
    attn.Forward(Wave<float>({1, 2, kDModel}), &cache.Layer(0), 0, Tensor<bool>(), true);
    cache.Advance(2);

    const Tensor<float> before = cache.Layer(0).k.Slice(2, 0, 2).Clone();

    attn.Forward(Wave<float>({1, 1, kDModel}, 1.3), &cache.Layer(0), cache.Room(1),
                 Tensor<bool>(), true);
    cache.Advance(1);

    for (size_t h = 0; h < kHeads; ++h)
        for (size_t s = 0; s < 2; ++s)
            for (size_t d = 0; d < kHeadDim; ++d)
                EXPECT_FLOAT_EQ(cache.Layer(0).k(0, h, s, d), before(0, h, s, d))
                    << "position " << s << " was overwritten by a later append";
}

// --- TransformerBlock --------------------------------------------------------

TEST(KVCacheBlock, DecodingOneTokenAtATimeMatchesTheFullSequence) {
    TransformerBlock<float> block(kDModel, kHeads, 16);
    Seed(block);

    constexpr size_t kSeq = 5;
    const Tensor<float> x = Wave<float>({1, kSeq, kDModel});
    const Tensor<float> reference = block.Forward(x, nullptr, 0, Tensor<bool>(), true);

    KVCache<float> cache(1, 1, kHeads, kMaxSeq, kHeadDim);

    for (size_t s = 0; s < kSeq; ++s) {
        const Tensor<float> out =
            block.Forward(Window(x, s, 1), &cache.Layer(0), cache.Room(1), Tensor<bool>(), true);
        cache.Advance(1);
        EXPECT_LT(RowDiff(out, 0, reference, s), kExact) << "position " << s;
    }
}

TEST(KVCacheBlock, ANullCacheLeavesTheResultUnchanged) {
    TransformerBlock<float> block(kDModel, kHeads, 16);
    Seed(block);
    const Tensor<float> x = Wave<float>({1, 3, kDModel});

    EXPECT_EQ(MaxAbsDiff(block.Forward(x, nullptr, 0, Tensor<bool>(), true),
                         block.Forward(x, nullptr, 0, Tensor<bool>(), true)),
              0.0);
}

// --- Transformer: the multi-layer bookkeeping --------------------------------

TEST(KVCacheTransformer, DecodingOneTokenAtATimeMatchesTheFullSequence) {
    const TransformerConfig config = TinyConfig();
    Transformer<float> model(config);
    Seed(model);

    constexpr size_t kSeq = 5;
    const Tensor<float> x = Wave<float>({1, kSeq, kDModel});
    const Tensor<float> reference = model.Forward(x, nullptr, Tensor<bool>(), true);

    KVCache<float> cache(config.num_layers, 1, config.KVHeads(), config.max_seq_len,
                         config.HeadDim());

    for (size_t s = 0; s < kSeq; ++s) {
        const Tensor<float> out = model.Forward(Window(x, s, 1), &cache, Tensor<bool>(), true);
        EXPECT_LT(RowDiff(out, 0, reference, s), kExact) << "position " << s;
    }
}

TEST(KVCacheTransformer, PrefillThenDecodeMatchesTheFullSequence) {
    const TransformerConfig config = TinyConfig();
    Transformer<float> model(config);
    Seed(model);

    constexpr size_t kSeq = 5, kPrompt = 3;
    const Tensor<float> x = Wave<float>({1, kSeq, kDModel});
    const Tensor<float> reference = model.Forward(x, nullptr, Tensor<bool>(), true);

    KVCache<float> cache(config.num_layers, 1, config.KVHeads(), config.max_seq_len,
                         config.HeadDim());

    const Tensor<float> prefill = model.Forward(Window(x, 0, kPrompt), &cache, Tensor<bool>(), true);
    for (size_t s = 0; s < kPrompt; ++s)
        EXPECT_LT(RowDiff(prefill, s, reference, s), kExact) << "prefill position " << s;

    for (size_t s = kPrompt; s < kSeq; ++s) {
        const Tensor<float> out = model.Forward(Window(x, s, 1), &cache, Tensor<bool>(), true);
        EXPECT_LT(RowDiff(out, 0, reference, s), kExact) << "decode position " << s;
    }
}

TEST(KVCacheTransformer, OnePassAdvancesTheLengthOnceNotOncePerLayer) {
    const TransformerConfig config = TinyConfig();
    Transformer<float> model(config);
    Seed(model);

    KVCache<float> cache(config.num_layers, 1, config.KVHeads(), config.max_seq_len,
                         config.HeadDim());

    model.Forward(Wave<float>({1, 2, kDModel}), &cache, Tensor<bool>(), true);
    EXPECT_EQ(cache.Length(), 2u) << "advanced " << cache.Length() << " for a 2-token pass over "
                                  << config.num_layers << " layers";

    model.Forward(Wave<float>({1, 1, kDModel}), &cache, Tensor<bool>(), true);
    EXPECT_EQ(cache.Length(), 3u);
}

// Every layer must append at the SAME offset. If the length were advanced
// inside the block loop, layer i would write at position start + i.
TEST(KVCacheTransformer, EveryLayerAppendsAtTheSameOffset) {
    const TransformerConfig config = TinyConfig();
    Transformer<float> model(config);
    Seed(model);

    KVCache<float> cache(config.num_layers, 1, config.KVHeads(), config.max_seq_len,
                         config.HeadDim());
    model.Forward(Wave<float>({1, 1, kDModel}), &cache, Tensor<bool>(), true);

    ASSERT_EQ(cache.Length(), 1u);

    for (size_t l = 0; l < config.num_layers; ++l) {
        EXPECT_GT(NonZeroCount(cache.Layer(l).k, 0), 0u)
            << "layer " << l << " never wrote position 0";

        for (size_t s = 1; s < config.max_seq_len; ++s)
            EXPECT_EQ(NonZeroCount(cache.Layer(l).k, s), 0u)
                << "layer " << l << " wrote position " << s << " instead of 0";
    }
}

TEST(KVCacheTransformer, LayersHoldDifferentKeys) {
    const TransformerConfig config = TinyConfig();
    Transformer<float> model(config);
    Seed(model);

    KVCache<float> cache(config.num_layers, 1, config.KVHeads(), config.max_seq_len,
                         config.HeadDim());
    model.Forward(Wave<float>({1, 2, kDModel}), &cache, Tensor<bool>(), true);

    // Passing the same slot to every block would make these identical.
    double worst = 0.0;
    for (size_t h = 0; h < config.KVHeads(); ++h)
        for (size_t d = 0; d < config.HeadDim(); ++d)
            worst = std::max(worst, std::abs(static_cast<double>(cache.Layer(0).k(0, h, 0, d)) -
                                             static_cast<double>(cache.Layer(1).k(0, h, 0, d))));
    EXPECT_GT(worst, 1e-4) << "each block must be given its own cache slot";
}

TEST(KVCacheTransformer, ResetLetsTheSameCacheServeANewSequence) {
    const TransformerConfig config = TinyConfig();
    Transformer<float> model(config);
    Seed(model);

    KVCache<float> cache(config.num_layers, 1, config.KVHeads(), config.max_seq_len,
                         config.HeadDim());

    const Tensor<float> x = Wave<float>({1, 3, kDModel});
    const Tensor<float> first = model.Forward(x, &cache, Tensor<bool>(), true);

    cache.Reset();
    const Tensor<float> second = model.Forward(x, &cache, Tensor<bool>(), true);

    EXPECT_EQ(cache.Length(), 3u);
    EXPECT_LT(MaxAbsDiff(first, second), kExact)
        << "stale values past Length() must not affect the result";
}

TEST(KVCacheTransformer, OverflowThrowsBeforeAnyLayerWrites) {
    const TransformerConfig config = TinyConfig();
    Transformer<float> model(config);
    Seed(model);

    KVCache<float> cache(config.num_layers, 1, config.KVHeads(), config.max_seq_len,
                         config.HeadDim());
    model.Forward(Wave<float>({1, kMaxSeq, kDModel}), &cache, Tensor<bool>(), true);
    ASSERT_EQ(cache.Length(), kMaxSeq);

    EXPECT_THROW(model.Forward(Wave<float>({1, 1, kDModel}), &cache, Tensor<bool>(), true),
                 std::out_of_range);
    EXPECT_EQ(cache.Length(), kMaxSeq) << "a rejected pass must not advance the length";
}

TEST(KVCacheTransformer, ANullCacheLeavesTheResultUnchanged) {
    const TransformerConfig config = TinyConfig();
    Transformer<float> model(config);
    Seed(model);

    const Tensor<float> x = Wave<float>({1, 4, kDModel});
    EXPECT_EQ(MaxAbsDiff(model.Forward(x, nullptr, Tensor<bool>(), true),
                         model.Forward(x, nullptr, Tensor<bool>(), true)),
              0.0);
}
