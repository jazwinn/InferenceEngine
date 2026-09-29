#include <gtest/gtest.h>

#include <NeuralNet/Transformer/RoPE.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using math::Tensor;
using nn::transformer::RotaryEmbedding;

namespace {

constexpr float kTol = 1e-5f;

void WaveFill(Tensor<float>& t, double phase) {
    float* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i)
        data[i] = static_cast<float>(0.5 * std::sin(phase + 0.37 * static_cast<double>(i)));
}

Tensor<float> Vector4(float a, float b, float c, float d) {
    Tensor<float> t({1, 1, 1, 4});
    t(0, 0, 0, 0) = a;
    t(0, 0, 0, 1) = b;
    t(0, 0, 0, 2) = c;
    t(0, 0, 0, 3) = d;
    return t;
}

double Dot(const Tensor<float>& a, const Tensor<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.Shape()[3]; ++i) sum += static_cast<double>(a(0, 0, 0, i)) * b(0, 0, 0, i);
    return sum;
}

Tensor<float> ReferenceRotate(const Tensor<float>& x, size_t start_pos, double theta) {
    const std::vector<size_t>& s = x.Shape();
    const size_t d = s[3], half = d / 2;
    Tensor<float> out(s);
    for (size_t b = 0; b < s[0]; ++b)
        for (size_t h = 0; h < s[1]; ++h)
            for (size_t t = 0; t < s[2]; ++t)
                for (size_t i = 0; i < half; ++i) {
                    const double angle = static_cast<double>(start_pos + t) *
                                         std::pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(d));
                    const double x1 = x(b, h, t, i), x2 = x(b, h, t, i + half);
                    out(b, h, t, i) = static_cast<float>(x1 * std::cos(angle) - x2 * std::sin(angle));
                    out(b, h, t, i + half) = static_cast<float>(x2 * std::cos(angle) + x1 * std::sin(angle));
                }
    return out;
}

void ExpectNear(const Tensor<float>& actual, const Tensor<float>& expected) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    Tensor<float> a = actual.Contiguous();
    Tensor<float> e = expected.Contiguous();
    for (size_t i = 0; i < a.Size(); ++i) EXPECT_NEAR(a.Data()[i], e.Data()[i], kTol) << "at flat index " << i;
}

}  // namespace

TEST(RotaryEmbeddingConstruction, TablesHaveOneRowPerPositionAndOneColumnPerPair) {
    RotaryEmbedding<float> rope(8, 12);

    EXPECT_EQ(rope.headDim, 8u);
    EXPECT_EQ(rope.seq, 12u);
    EXPECT_EQ(rope.cos.Shape(), (std::vector<size_t>{12, 4}));
    EXPECT_EQ(rope.sin.Shape(), (std::vector<size_t>{12, 4}));
}

TEST(RotaryEmbeddingConstruction, TablesFollowTheFrequencyFormula) {
    for (double theta : {10000.0, 500000.0}) {
        SCOPED_TRACE(theta);
        RotaryEmbedding<float> rope(8, 20, theta);
        for (size_t p = 0; p < 20; ++p)
            for (size_t i = 0; i < 4; ++i) {
                const double angle = static_cast<double>(p) * std::pow(theta, -2.0 * static_cast<double>(i) / 8.0);
                EXPECT_NEAR(rope.cos(p, i), std::cos(angle), kTol);
                EXPECT_NEAR(rope.sin(p, i), std::sin(angle), kTol);
            }
    }
}

TEST(RotaryEmbeddingConstruction, FirstPairTurnsOneRadianPerPosition) {
    RotaryEmbedding<float> rope(4, 4);

    EXPECT_NEAR(rope.cos(1, 0), std::cos(1.0), kTol);
    EXPECT_NEAR(rope.sin(1, 0), std::sin(1.0), kTol);
    EXPECT_NEAR(rope.cos(1, 1), std::cos(0.01), kTol);
    EXPECT_NEAR(rope.sin(1, 1), std::sin(0.01), kTol);
}

TEST(RotaryEmbeddingConstruction, RejectsInvalidArguments) {
    EXPECT_THROW(RotaryEmbedding<float>(0, 8), std::invalid_argument);
    EXPECT_THROW(RotaryEmbedding<float>(3, 8), std::invalid_argument);
    EXPECT_THROW(RotaryEmbedding<float>(4, 0), std::invalid_argument);
    EXPECT_THROW(RotaryEmbedding<float>(4, 8, 0.0), std::invalid_argument);
    EXPECT_THROW(RotaryEmbedding<float>(4, 8, -1.0), std::invalid_argument);
}

TEST(RotaryEmbeddingApply, PositionZeroIsIdentity) {
    RotaryEmbedding<float> rope(4, 8);
    const Tensor<float> x = Vector4(0.3f, -1.2f, 0.7f, 2.0f);

    ExpectNear(rope.Apply(x, 0), x);
}

TEST(RotaryEmbeddingApply, PairsElementWithTheOneHalfAHeadAway) {
    RotaryEmbedding<float> rope(4, 8);

    const Tensor<float> a = rope.Apply(Vector4(1, 0, 0, 0), 1);
    EXPECT_NEAR(a(0, 0, 0, 0), std::cos(1.0), kTol);
    EXPECT_NEAR(a(0, 0, 0, 1), 0.0f, kTol);
    EXPECT_NEAR(a(0, 0, 0, 2), std::sin(1.0), kTol);
    EXPECT_NEAR(a(0, 0, 0, 3), 0.0f, kTol);

    const Tensor<float> b = rope.Apply(Vector4(0, 1, 0, 0), 2);
    EXPECT_NEAR(b(0, 0, 0, 0), 0.0f, kTol);
    EXPECT_NEAR(b(0, 0, 0, 1), std::cos(0.02), kTol);
    EXPECT_NEAR(b(0, 0, 0, 2), 0.0f, kTol);
    EXPECT_NEAR(b(0, 0, 0, 3), std::sin(0.02), kTol);
}

TEST(RotaryEmbeddingApply, PreservesVectorLength) {
    RotaryEmbedding<float> rope(4, 16);
    const Tensor<float> x = Vector4(0.3f, -1.2f, 0.7f, 2.0f);

    for (size_t p : {1u, 5u, 15u}) {
        const Tensor<float> r = rope.Apply(x, p);
        EXPECT_NEAR(Dot(r, r), Dot(x, x), 1e-4);
    }
}

TEST(RotaryEmbeddingApply, DotProductDependsOnlyOnRelativeDistance) {
    RotaryEmbedding<float> rope(4, 32);
    const Tensor<float> q = Vector4(0.3f, -1.2f, 0.7f, 2.0f);
    const Tensor<float> k = Vector4(-0.5f, 0.4f, 1.1f, -0.9f);

    const double reference = Dot(rope.Apply(q, 3), rope.Apply(k, 0));
    for (size_t shift : {4u, 11u, 25u})
        EXPECT_NEAR(Dot(rope.Apply(q, 3 + shift), rope.Apply(k, shift)), reference, 1e-4) << "shift " << shift;

    EXPECT_GT(std::fabs(Dot(rope.Apply(q, 5), rope.Apply(k, 0)) - reference), 1e-3);
}

TEST(RotaryEmbeddingApply, MatchesReferenceImplementation) {
    RotaryEmbedding<float> rope(8, 16, 500000.0);
    Tensor<float> x({2, 3, 5, 8});
    WaveFill(x, 0.4);

    ExpectNear(rope.Apply(x, 6), ReferenceRotate(x, 6, 500000.0));
}

TEST(RotaryEmbeddingApply, SingleTokenAtOffsetMatchesFullSequence) {
    RotaryEmbedding<float> rope(8, 16);
    Tensor<float> x({2, 3, 5, 8});
    WaveFill(x, 0.4);
    const Tensor<float> full = rope.Apply(x, 2);

    for (size_t t = 0; t < 5; ++t) {
        SCOPED_TRACE(t);
        ExpectNear(rope.Apply(x.Slice(2, t, 1), 2 + t), full.Slice(2, t, 1));
    }
}

TEST(RotaryEmbeddingApply, HandlesNonContiguousInput) {
    RotaryEmbedding<float> rope(8, 16);
    Tensor<float> base({2, 5, 3, 8});
    WaveFill(base, 1.1);
    const Tensor<float> view = base.Transpose(1, 2);
    ASSERT_FALSE(view.IsContiguous());

    ExpectNear(rope.Apply(view, 3), ReferenceRotate(view.Contiguous(), 3, 10000.0));
}

TEST(RotaryEmbeddingApply, ReturnsNewTensorAndLeavesInputUntouched) {
    RotaryEmbedding<float> rope(4, 8);
    const Tensor<float> x = Vector4(0.3f, -1.2f, 0.7f, 2.0f);
    const Tensor<float> before = x.Clone();

    const Tensor<float> r = rope.Apply(x, 3);

    EXPECT_FALSE(r.SharesStorage(x));
    ExpectNear(x, before);
}

TEST(RotaryEmbeddingApply, AcceptsPositionsUpToMaxSeq) {
    RotaryEmbedding<float> rope(4, 8);
    Tensor<float> x({1, 1, 3, 4}, 1.0f);

    EXPECT_NO_THROW(rope.Apply(x, 5));
    EXPECT_THROW(rope.Apply(x, 6), std::out_of_range);
    EXPECT_THROW(rope.Apply(x, 100), std::out_of_range);
}

TEST(RotaryEmbeddingApply, RejectsWrongShape) {
    RotaryEmbedding<float> rope(4, 8);

    EXPECT_THROW(rope.Apply(Tensor<float>({1, 3, 4}, 1.0f), 0), std::invalid_argument);
    EXPECT_THROW(rope.Apply(Tensor<float>({1, 1, 3, 6}, 1.0f), 0), std::invalid_argument);
}

TEST(RotaryEmbeddingApply, WorksInDoublePrecision) {
    RotaryEmbedding<double> rope(4, 8);
    Tensor<double> x({1, 1, 1, 4}, 0.0);
    x(0, 0, 0, 0) = 1.0;

    const Tensor<double> r = rope.Apply(x, 3);

    EXPECT_NEAR(r(0, 0, 0, 0), std::cos(3.0), 1e-12);
    EXPECT_NEAR(r(0, 0, 0, 2), std::sin(3.0), 1e-12);
}
