#include <gtest/gtest.h>

#include <NeuralNet/Transformer/KVCache.h>

#include <stdexcept>
#include <vector>

using math::Tensor;
using nn::transformer::KVCache;

namespace {

constexpr size_t kLayers  = 3;
constexpr size_t kBatch   = 2;
constexpr size_t kHeads   = 4;
constexpr size_t kMaxSeq  = 8;
constexpr size_t kHeadDim = 5;

KVCache<float> MakeCache() {
    return KVCache<float>(kLayers, kBatch, kHeads, kMaxSeq, kHeadDim);
}

// A value unique to (layer, which, b, h, s, d) so a misplaced write is visible.
float Pattern(size_t layer, size_t which, size_t b, size_t h, size_t s, size_t d) {
    return static_cast<float>(1 + layer * 100000 + which * 10000 + b * 1000 + h * 100 + s * 10 + d);
}

// Write `seq` positions starting at `start` into one layer's K and V,
// the way MultiHeadAttention appends freshly projected keys and values.
void WriteWindow(KVCache<float>& cache, size_t layer, size_t start, size_t seq) {
    Tensor<float> k = cache.Layer(layer).k.Slice(2, start, seq);
    Tensor<float> v = cache.Layer(layer).v.Slice(2, start, seq);

    for (size_t b = 0; b < kBatch; ++b)
        for (size_t h = 0; h < kHeads; ++h)
            for (size_t s = 0; s < seq; ++s)
                for (size_t d = 0; d < kHeadDim; ++d) {
                    k(b, h, s, d) = Pattern(layer, 0, b, h, start + s, d);
                    v(b, h, s, d) = Pattern(layer, 1, b, h, start + s, d);
                }
}

// ---------------------------------------------------------------- construction

TEST(KVCache, ReportsItsDimensions) {
    KVCache<float> cache = MakeCache();
    EXPECT_EQ(cache.Layers(), kLayers);
    EXPECT_EQ(cache.MaxLength(), kMaxSeq);
    EXPECT_EQ(cache.Length(), 0u);
}

TEST(KVCache, AllocatesOneKAndVPerLayerInHeadLayout) {
    KVCache<float> cache = MakeCache();
    const std::vector<size_t> expected{kBatch, kHeads, kMaxSeq, kHeadDim};

    for (size_t i = 0; i < kLayers; ++i) {
        EXPECT_EQ(cache.Layer(i).k.Shape(), expected) << "layer " << i;
        EXPECT_EQ(cache.Layer(i).v.Shape(), expected) << "layer " << i;
    }
}

TEST(KVCache, StartsZeroed) {
    KVCache<float> cache = MakeCache();
    for (size_t i = 0; i < kLayers; ++i)
        for (size_t b = 0; b < kBatch; ++b)
            for (size_t h = 0; h < kHeads; ++h)
                for (size_t s = 0; s < kMaxSeq; ++s)
                    for (size_t d = 0; d < kHeadDim; ++d) {
                        EXPECT_EQ(cache.Layer(i).k(b, h, s, d), 0.0f) << "layer " << i;
                        EXPECT_EQ(cache.Layer(i).v(b, h, s, d), 0.0f) << "layer " << i;
                    }
}

TEST(KVCache, RejectsZeroDimensions) {
    EXPECT_THROW(KVCache<float>(0, kBatch, kHeads, kMaxSeq, kHeadDim), std::invalid_argument);
    EXPECT_THROW(KVCache<float>(kLayers, 0, kHeads, kMaxSeq, kHeadDim), std::invalid_argument);
    EXPECT_THROW(KVCache<float>(kLayers, kBatch, 0, kMaxSeq, kHeadDim), std::invalid_argument);
    EXPECT_THROW(KVCache<float>(kLayers, kBatch, kHeads, 0, kHeadDim), std::invalid_argument);
    EXPECT_THROW(KVCache<float>(kLayers, kBatch, kHeads, kMaxSeq, 0), std::invalid_argument);
}

// ---------------------------------------------------------------- layer access

TEST(KVCache, LayerIndexIsBoundsChecked) {
    KVCache<float> cache = MakeCache();
    EXPECT_NO_THROW(cache.Layer(kLayers - 1));
    EXPECT_THROW(cache.Layer(kLayers), std::out_of_range);
}

TEST(KVCache, LayerReturnsTheSameStorageEveryCall) {
    KVCache<float> cache = MakeCache();
    cache.Layer(1).k(0, 0, 0, 0) = 42.0f;
    EXPECT_EQ(cache.Layer(1).k(0, 0, 0, 0), 42.0f);
    EXPECT_EQ(&cache.Layer(1), &cache.Layer(1));
}

TEST(KVCache, LayersAndKVDoNotShareStorage) {
    KVCache<float> cache = MakeCache();

    for (size_t i = 0; i < kLayers; ++i) {
        EXPECT_FALSE(cache.Layer(i).k.SharesStorage(cache.Layer(i).v)) << "layer " << i;
        for (size_t j = i + 1; j < kLayers; ++j) {
            EXPECT_FALSE(cache.Layer(i).k.SharesStorage(cache.Layer(j).k));
            EXPECT_FALSE(cache.Layer(i).v.SharesStorage(cache.Layer(j).v));
        }
    }

    // Behavioural version of the same claim.
    cache.Layer(0).k(0, 0, 0, 0) = 7.0f;
    EXPECT_EQ(cache.Layer(0).v(0, 0, 0, 0), 0.0f);
    EXPECT_EQ(cache.Layer(1).k(0, 0, 0, 0), 0.0f);
}

// ------------------------------------------------------------- length tracking

TEST(KVCache, AdvanceAccumulates) {
    KVCache<float> cache = MakeCache();
    cache.Advance(3);
    EXPECT_EQ(cache.Length(), 3u);
    cache.Advance(1);
    EXPECT_EQ(cache.Length(), 4u);
    cache.Advance(1);
    EXPECT_EQ(cache.Length(), 5u);
}

TEST(KVCache, AdvanceMayFillExactlyToMax) {
    KVCache<float> cache = MakeCache();
    EXPECT_NO_THROW(cache.Advance(kMaxSeq));
    EXPECT_EQ(cache.Length(), kMaxSeq);
    EXPECT_THROW(cache.Advance(1), std::out_of_range);
}

TEST(KVCache, AdvanceBeyondMaxThrowsAndLeavesLengthUnchanged) {
    KVCache<float> cache = MakeCache();
    cache.Advance(6);
    EXPECT_THROW(cache.Advance(3), std::out_of_range);
    EXPECT_EQ(cache.Length(), 6u) << "a rejected Advance must not partially apply";
}

TEST(KVCache, AdvanceOverflowCannotWrapAround) {
    KVCache<float> cache = MakeCache();
    cache.Advance(1);
    // m_length + seq overflows size_t if the bound is written as an addition.
    EXPECT_THROW(cache.Advance(static_cast<size_t>(-1)), std::out_of_range);
    EXPECT_EQ(cache.Length(), 1u);
}

TEST(KVCache, ResetRewindsLengthButKeepsTheAllocation) {
    KVCache<float> cache = MakeCache();
    cache.Advance(5);
    cache.Reset();

    EXPECT_EQ(cache.Length(), 0u);
    EXPECT_EQ(cache.MaxLength(), kMaxSeq);
    EXPECT_EQ(cache.Layers(), kLayers);
    EXPECT_EQ(cache.Layer(0).k.Shape(), (std::vector<size_t>{kBatch, kHeads, kMaxSeq, kHeadDim}));
    EXPECT_NO_THROW(cache.Advance(kMaxSeq));
}

// ------------------------------------------------- the append / read-back cycle

TEST(KVCache, SliceIsAWritableViewIntoTheCache) {
    KVCache<float> cache = MakeCache();
    Tensor<float> window = cache.Layer(0).k.Slice(2, 2, 3);

    EXPECT_TRUE(window.SharesStorage(cache.Layer(0).k))
        << "Slice must alias the cache, not copy it";

    window(0, 0, 0, 0) = 9.0f;
    EXPECT_EQ(cache.Layer(0).k(0, 0, 2, 0), 9.0f) << "window row 0 is absolute position 2";
}

TEST(KVCache, TwoAppendsReadBackAsOneContiguousSequence) {
    KVCache<float> cache = MakeCache();

    // A prefill of 3, then a decode step of 2 -- the same code path either way.
    const size_t prefill = 3;
    const size_t decode  = 2;

    for (size_t layer = 0; layer < kLayers; ++layer) WriteWindow(cache, layer, cache.Length(), prefill);
    cache.Advance(prefill);
    EXPECT_EQ(cache.Length(), prefill);

    for (size_t layer = 0; layer < kLayers; ++layer) WriteWindow(cache, layer, cache.Length(), decode);
    cache.Advance(decode);
    EXPECT_EQ(cache.Length(), prefill + decode);

    for (size_t layer = 0; layer < kLayers; ++layer) {
        Tensor<float> k = cache.Layer(layer).k.Slice(2, 0, cache.Length());
        Tensor<float> v = cache.Layer(layer).v.Slice(2, 0, cache.Length());

        ASSERT_EQ(k.Shape()[2], prefill + decode);

        for (size_t b = 0; b < kBatch; ++b)
            for (size_t h = 0; h < kHeads; ++h)
                for (size_t s = 0; s < cache.Length(); ++s)
                    for (size_t d = 0; d < kHeadDim; ++d) {
                        EXPECT_EQ(k(b, h, s, d), Pattern(layer, 0, b, h, s, d))
                            << "K layer " << layer << " pos " << s;
                        EXPECT_EQ(v(b, h, s, d), Pattern(layer, 1, b, h, s, d))
                            << "V layer " << layer << " pos " << s;
                    }
    }
}

TEST(KVCache, AppendingLeavesTheUnusedTailUntouched) {
    KVCache<float> cache = MakeCache();
    WriteWindow(cache, 0, 0, 3);
    cache.Advance(3);

    for (size_t b = 0; b < kBatch; ++b)
        for (size_t h = 0; h < kHeads; ++h)
            for (size_t s = cache.Length(); s < kMaxSeq; ++s)
                for (size_t d = 0; d < kHeadDim; ++d) {
                    EXPECT_EQ(cache.Layer(0).k(b, h, s, d), 0.0f) << "pos " << s;
                    EXPECT_EQ(cache.Layer(0).v(b, h, s, d), 0.0f) << "pos " << s;
                }
}

TEST(KVCache, ReadBackViewIsNonContiguousButCorrectlyStrided) {
    KVCache<float> cache = MakeCache();
    WriteWindow(cache, 0, 0, 3);
    cache.Advance(3);

    Tensor<float> k = cache.Layer(0).k.Slice(2, 0, cache.Length());

    // MatMul consumes this view directly, so it must survive being non-contiguous.
    EXPECT_FALSE(k.IsContiguous());
    EXPECT_EQ(k.Strides(), cache.Layer(0).k.Strides());
    EXPECT_EQ(k(1, 2, 1, 3), Pattern(0, 0, 1, 2, 1, 3));
}

TEST(KVCache, ResetMakesPositionZeroWritableAgain) {
    KVCache<float> cache = MakeCache();
    WriteWindow(cache, 0, 0, 3);
    cache.Advance(3);
    cache.Reset();

    WriteWindow(cache, 0, 0, 1);
    cache.Advance(1);

    EXPECT_EQ(cache.Length(), 1u);
    EXPECT_EQ(cache.Layer(0).k(0, 0, 0, 0), Pattern(0, 0, 0, 0, 0, 0));
    // Stale values past the new length are fine: nothing reads beyond Length().
}

// ------------------------------------------------------------------ other types

TEST(KVCache, WorksForOtherScalarTypes) {
    KVCache<double> cache(1, 1, 1, 4, 2);
    EXPECT_EQ(cache.Length(), 0u);
    cache.Layer(0).k(0, 0, 0, 0) = 0.5;
    EXPECT_DOUBLE_EQ(cache.Layer(0).k(0, 0, 0, 0), 0.5);
}

}  // namespace
