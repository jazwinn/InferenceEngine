#include <gtest/gtest.h>

#include <NeuralNet/Transformer/KVCache.h>
#include <NeuralNet/Transformer/MultiHeadAttention.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using tensor::Tensor;
using nn::Module;
using nn::transformer::KVCache;
using nn::transformer::MultiHeadAttention;

namespace {
constexpr size_t kDModel  = 8;
constexpr size_t kHeads   = 2;
constexpr size_t kHeadDim = kDModel / kHeads;
constexpr size_t kMaxSeq  = 6;

constexpr double kExact = 1e-6;

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
