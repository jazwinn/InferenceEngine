#include <gtest/gtest.h>

#include <Models/gpt2/GPT2.h>
#include <NeuralNet/Transformer/KVCache.h>
#include <NeuralNet/Transformer/MultiHeadAttention.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using models::gpt2::GPT2;
using math::Tensor;
using nn::Module;
using nn::transformer::KVCache;
using nn::transformer::MultiHeadAttention;
using nn::transformer::config::TransformerConfig;

namespace {
constexpr size_t kDModel  = 8;
constexpr size_t kHeads   = 2;
constexpr size_t kHeadDim = kDModel / kHeads;
constexpr size_t kMaxSeq  = 6;
constexpr size_t kLayers  = 3;

constexpr double kExact = 1e-6;

TransformerConfig TinyConfig() {
    TransformerConfig c;
    c.vocab_size  = 11;
    c.d_model     = kDModel;
    c.num_layers  = kLayers;
    c.num_heads   = kHeads;
    c.ffn_dim     = 16;
    c.max_seq_len = kMaxSeq;
    c.activation  = nn::transformer::config::Activation::GELUTanh;
    return c;
}

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

Tensor<float> Window(const Tensor<float>& x, size_t first, size_t count) {
    Tensor<float> out({x.Shape()[0], count, x.Shape()[2]});
    for (size_t b = 0; b < x.Shape()[0]; ++b)
        for (size_t s = 0; s < count; ++s)
            for (size_t d = 0; d < x.Shape()[2]; ++d) out(b, s, d) = x(b, first + s, d);
    return out;
}

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

}

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

    for (size_t s = 0; s < 4; s += 2) {
        const Tensor<float> out =
            attn.Forward(Window(x, s, 2), &cache.Layer(0), cache.Room(2), Tensor<bool>(), true);
        cache.Advance(2);

        EXPECT_LT(RowDiff(out, 0, reference, s), kExact) << "position " << s;
        EXPECT_LT(RowDiff(out, 1, reference, s + 1), kExact) << "position " << s + 1;
    }
}

TEST(KVCacheAttention, CacheHoldsExactlyTheProjectedKeysAndValues) {
    MultiHeadAttention<float> attn(kDModel, kHeads);
    Seed(attn);

    const Tensor<float> x = Wave<float>({1, 3, kDModel});
    KVCache<float> cache(1, 1, kHeads, kMaxSeq, kHeadDim);
    attn.Forward(x, &cache.Layer(0), 0, Tensor<bool>(), true);
    cache.Advance(3);

    const Tensor<float> K = nn::transformer::SplitHeads(attn.key->Forward(x), attn.num_heads, attn.head_dim);
    const Tensor<float> V = nn::transformer::SplitHeads(attn.value->Forward(x), attn.num_heads, attn.head_dim);

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

namespace {
TransformerConfig TinyGPT2Config() {
    TransformerConfig c = TinyConfig();
    c.position = nn::transformer::config::PositionEncoding::Learned;
    return c;
}

Tensor<int64_t> Ids(size_t seq) {
    Tensor<int64_t> t({1, seq});
    for (size_t s = 0; s < seq; ++s) t(0, s) = static_cast<int64_t>((s * 3 + 1) % 11);
    return t;
}

int64_t ArgMax(const Tensor<float>& logits, size_t position, size_t vocab) {
    int64_t best = 0;
    for (size_t v = 1; v < vocab; ++v)
        if (logits(0, position, v) > logits(0, position, best)) best = static_cast<int64_t>(v);
    return best;
}

}

TEST(KVCacheGPT2, DecodingOneTokenAtATimeMatchesTheFullSequence) {
    const TransformerConfig config = TinyGPT2Config();
    GPT2<float> model(config);
    Seed(model);
    ASSERT_NE(model.wpe, nullptr) << "this test is meaningless without learned positions";

    constexpr size_t kSeq = 5;
    const Tensor<int64_t> ids = Ids(kSeq);
    const Tensor<float> reference = model.Forward(ids, nullptr, 0);

    KVCache<float> cache(config.num_layers, 1, config.KVHeads(), config.max_seq_len,
                         config.HeadDim());

    for (size_t s = 0; s < kSeq; ++s) {
        Tensor<int64_t> one({1, 1});
        one(0, 0) = ids(0, s);
        const Tensor<float> out = model.Forward(one, &cache);
        EXPECT_LT(RowDiff(out, 0, reference, s), kExact) << "position " << s;
    }
}

TEST(KVCacheGPT2, PrefillThenDecodeMatchesTheFullSequence) {
    const TransformerConfig config = TinyGPT2Config();
    GPT2<float> model(config);
    Seed(model);

    constexpr size_t kSeq = 5, kPrompt = 3;
    const Tensor<int64_t> ids = Ids(kSeq);
    const Tensor<float> reference = model.Forward(ids, nullptr, 0);

    KVCache<float> cache(config.num_layers, 1, config.KVHeads(), config.max_seq_len,
                         config.HeadDim());

    Tensor<int64_t> prompt({1, kPrompt});
    for (size_t s = 0; s < kPrompt; ++s) prompt(0, s) = ids(0, s);
    const Tensor<float> prefill = model.Forward(prompt, &cache);

    for (size_t s = 0; s < kPrompt; ++s)
        EXPECT_LT(RowDiff(prefill, s, reference, s), kExact) << "prefill position " << s;

    for (size_t s = kPrompt; s < kSeq; ++s) {
        Tensor<int64_t> one({1, 1});
        one(0, 0) = ids(0, s);
        EXPECT_LT(RowDiff(model.Forward(one, &cache), 0, reference, s), kExact)
            << "decode position " << s;
    }
}

TEST(KVCacheGPT2, PositionActuallyChangesTheLogits) {
    const TransformerConfig config = TinyGPT2Config();
    GPT2<float> model(config);
    Seed(model);

    Tensor<int64_t> same({1, 2});
    same(0, 0) = 4;
    same(0, 1) = 4;

    const Tensor<float> out = model.Forward(same, nullptr, 0);
    EXPECT_GT(RowDiff(out, 0, out, 1), 1e-3)
        << "the same token at positions 0 and 1 should not give identical logits";
}

TEST(KVCacheGPT2, GreedyDecodingPicksTheSameTokensWithAndWithoutACache) {
    const TransformerConfig config = TinyGPT2Config();
    GPT2<float> model(config);
    Seed(model);

    const size_t prompt_len = 2;
    const size_t steps = config.max_seq_len - prompt_len;

    std::vector<int64_t> slow{Ids(prompt_len)(0, 0), Ids(prompt_len)(0, 1)};
    std::vector<int64_t> fast = slow;

    for (size_t i = 0; i < steps; ++i) {
        Tensor<int64_t> all({1, slow.size()});
        for (size_t s = 0; s < slow.size(); ++s) all(0, s) = slow[s];
        slow.push_back(ArgMax(model.Forward(all, nullptr, 0), slow.size() - 1, config.vocab_size));
    }

    KVCache<float> cache(config.num_layers, 1, config.KVHeads(), config.max_seq_len,
                         config.HeadDim());
    Tensor<int64_t> prompt({1, prompt_len});
    for (size_t s = 0; s < prompt_len; ++s) prompt(0, s) = fast[s];
    Tensor<float> logits = model.Forward(prompt, &cache);
    size_t last = prompt_len - 1;

    for (size_t i = 0; i < steps; ++i) {
        const int64_t best = ArgMax(logits, last, config.vocab_size);
        fast.push_back(best);
        if (i + 1 == steps) break;
        Tensor<int64_t> next({1, 1});
        next(0, 0) = best;
        logits = model.Forward(next, &cache);
        last = 0;
    }

    ASSERT_EQ(slow.size(), fast.size());
    for (size_t i = 0; i < slow.size(); ++i)
        EXPECT_EQ(slow[i], fast[i]) << "token " << i << " differs";
}
