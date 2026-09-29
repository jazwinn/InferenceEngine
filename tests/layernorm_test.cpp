#include <gtest/gtest.h>

#include <NeuralNet/Transformer/Normal/LayerNorm.h>
#include <NeuralNet/Module.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using math::Tensor;
using nn::transformer::normal::LayerNorm;
using nn::Module;

namespace {

constexpr float kTol = 1e-4f;

// Mean and (biased) variance over one contiguous block of `count` elements
// starting at flat index `start`.
template <typename T>
void BlockStats(const Tensor<T>& t, size_t start, size_t count, T& mean, T& variance) {
    const T* p = t.Data() + start;
    mean = T(0);
    for (size_t i = 0; i < count; ++i) mean += p[i];
    mean /= static_cast<T>(count);

    variance = T(0);
    for (size_t i = 0; i < count; ++i) {
        const T d = p[i] - mean;
        variance += d * d;
    }
    variance /= static_cast<T>(count);
}

template <typename T>
void IotaFillFlat(Tensor<T>& t, T start = T(1)) {
    T* p = t.Data();
    for (size_t i = 0; i < t.Size(); ++i) p[i] = start + static_cast<T>(i);
}

}  // namespace

// --- construction -----------------------------------------------------------

TEST(LayerNormConstruction, ScalarFeatureSizeNormalisesTheLastAxis) {
    LayerNorm<float> ln(4);

    EXPECT_EQ(ln.NormalizedShape(), (std::vector<size_t>{4}));
    EXPECT_EQ(ln.NormalizedSize(), 4u);
}

TEST(LayerNormConstruction, MultiAxisShapeIsStoredAsGiven) {
    LayerNorm<float> ln(std::vector<size_t>{3, 4, 5});

    EXPECT_EQ(ln.NormalizedShape(), (std::vector<size_t>{3, 4, 5}));
    EXPECT_EQ(ln.NormalizedSize(), 60u);
}

TEST(LayerNormConstruction, ParametersMatchNormalizedShape) {
    LayerNorm<float> ln(std::vector<size_t>{2, 3});

    EXPECT_EQ(ln.gamma->Shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(ln.beta->Shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(ln.Parameters().size(), 2u);
}

TEST(LayerNormConstruction, GammaStartsAtOneAndBetaAtZero) {
    LayerNorm<float> ln(std::vector<size_t>{2, 3});

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((*ln.gamma)(i, j), 1.0f);
            EXPECT_FLOAT_EQ((*ln.beta)(i, j), 0.0f);
        }
    }
}

TEST(LayerNormConstruction, RejectsEmptyOrZeroShape) {
    EXPECT_THROW(LayerNorm<float>(std::vector<size_t>{}), std::invalid_argument);
    EXPECT_THROW(LayerNorm<float>(std::vector<size_t>{3, 0}), std::invalid_argument);
}

TEST(LayerNormConstruction, ExplicitParametersAreAdopted) {
    Tensor<float> g({3}, 2.0f);
    Tensor<float> b({3}, 0.5f);
    LayerNorm<float> ln(g, b);

    EXPECT_EQ(ln.NormalizedShape(), (std::vector<size_t>{3}));
    EXPECT_FLOAT_EQ((*ln.gamma)(0), 2.0f);
    EXPECT_FLOAT_EQ((*ln.beta)(0), 0.5f);
}

TEST(LayerNormConstruction, RejectsMismatchedGammaAndBeta) {
    Tensor<float> g({3}, 1.0f);
    Tensor<float> b({4}, 0.0f);

    EXPECT_THROW(LayerNorm<float>(g, b), std::invalid_argument);
}

// --- forward: single axis ---------------------------------------------------

TEST(LayerNormForward, PreservesShape) {
    LayerNorm<float> ln(4);
    Tensor<float> in({2, 4});
    IotaFillFlat(in);

    const Tensor<float> out = ln.Forward(in);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{2, 4}));
}

TEST(LayerNormForward, EachRowHasZeroMeanAndUnitVariance) {
    LayerNorm<float> ln(4);
    Tensor<float> in({3, 4});
    IotaFillFlat(in, 1.0f);
    in(1, 2) = 100.0f;   // make the rows genuinely different

    const Tensor<float> out = ln.Forward(in);

    for (size_t row = 0; row < 3; ++row) {
        float mean = 0.0f;
        float variance = 0.0f;
        BlockStats(out, row * 4, 4, mean, variance);
        EXPECT_NEAR(mean, 0.0f, kTol) << "row " << row;
        EXPECT_NEAR(variance, 1.0f, 1e-3f) << "row " << row;
    }
}

TEST(LayerNormForward, KnownValuesForASingleRow) {
    // {1, 2, 3, 4}: mean 2.5, biased variance 1.25, stddev ~1.118034
    LayerNorm<float> ln(4);
    Tensor<float> in({1, 4});
    IotaFillFlat(in, 1.0f);

    const Tensor<float> out = ln.Forward(in);

    EXPECT_NEAR(out(0, 0), -1.341641f, kTol);
    EXPECT_NEAR(out(0, 1), -0.447214f, kTol);
    EXPECT_NEAR(out(0, 2), 0.447214f, kTol);
    EXPECT_NEAR(out(0, 3), 1.341641f, kTol);
}

TEST(LayerNormForward, RowsAreNormalisedIndependently) {
    LayerNorm<float> ln(2);
    Tensor<float> in({2, 2});
    in(0, 0) = 0.0f;   in(0, 1) = 1.0f;
    in(1, 0) = 100.0f; in(1, 1) = 200.0f;

    const Tensor<float> out = ln.Forward(in);

    EXPECT_NEAR(out(0, 0), out(1, 0), kTol) << "scale must not leak between rows";
    EXPECT_NEAR(out(0, 1), out(1, 1), kTol);
}

TEST(LayerNormForward, ConstantRowIsFiniteAndZero) {
    LayerNorm<float> ln(3);
    Tensor<float> in({1, 3}, 5.0f);

    const Tensor<float> out = ln.Forward(in);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_FALSE(std::isnan(out(0, j))) << "epsilon must prevent 0/0";
        EXPECT_NEAR(out(0, j), 0.0f, kTol);
    }
}

TEST(LayerNormForward, GammaScalesAndBetaShifts) {
    LayerNorm<float> ln(4);
    ln.gamma->Fill(2.0f);
    ln.beta->Fill(10.0f);

    Tensor<float> in({1, 4});
    IotaFillFlat(in, 1.0f);

    const Tensor<float> out = ln.Forward(in);

    EXPECT_NEAR(out(0, 0), -1.341641f * 2.0f + 10.0f, kTol);
    EXPECT_NEAR(out(0, 3), 1.341641f * 2.0f + 10.0f, kTol);
}

TEST(LayerNormForward, GammaIsAppliedPerFeature) {
    LayerNorm<float> ln(3);
    (*ln.gamma)(0) = 0.0f;
    (*ln.gamma)(1) = 1.0f;
    (*ln.gamma)(2) = 0.0f;

    Tensor<float> in({2, 3});
    IotaFillFlat(in, 1.0f);

    const Tensor<float> out = ln.Forward(in);

    for (size_t row = 0; row < 2; ++row) {
        EXPECT_NEAR(out(row, 0), 0.0f, kTol);
        EXPECT_NEAR(out(row, 2), 0.0f, kTol);
    }
}

TEST(LayerNormForward, IsInvariantToShiftAndScaleOfTheInput) {
    LayerNorm<float> ln(4);

    Tensor<float> base({1, 4});
    IotaFillFlat(base, 1.0f);

    Tensor<float> transformed({1, 4});
    for (size_t j = 0; j < 4; ++j) {
        transformed(0, j) = base(0, j) * 7.0f + 100.0f;
    }

    const Tensor<float> a = ln.Forward(base);
    const Tensor<float> b = ln.Forward(transformed);

    for (size_t j = 0; j < 4; ++j) {
        EXPECT_NEAR(a(0, j), b(0, j), 1e-3f);
    }
}

// --- forward: multiple axes -------------------------------------------------

TEST(LayerNormMultiAxis, NormalisesTheLastThreeDimensionsJointly) {
    LayerNorm<float> ln(std::vector<size_t>{2, 3, 4});
    Tensor<float> in({5, 2, 3, 4});
    IotaFillFlat(in, 1.0f);

    const Tensor<float> out = ln.Forward(in);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{5, 2, 3, 4}));
    const size_t inner = 24;
    for (size_t batch = 0; batch < 5; ++batch) {
        float mean = 0.0f;
        float variance = 0.0f;
        BlockStats(out, batch * inner, inner, mean, variance);
        EXPECT_NEAR(mean, 0.0f, kTol) << "batch " << batch;
        EXPECT_NEAR(variance, 1.0f, 1e-3f) << "batch " << batch;
    }
}

TEST(LayerNormMultiAxis, DiffersFromNormalisingOnlyTheLastAxis) {
    Tensor<float> in({1, 2, 3});
    IotaFillFlat(in, 1.0f);

    LayerNorm<float> last_only(3);
    LayerNorm<float> last_two(std::vector<size_t>{2, 3});

    const Tensor<float> a = last_only.Forward(in);
    const Tensor<float> b = last_two.Forward(in);

    EXPECT_GT(std::abs(a(0, 0, 0) - b(0, 0, 0)), 1e-3f)
        << "normalising over 6 elements must differ from two groups of 3";
}

TEST(LayerNormMultiAxis, WholeTensorWhenShapeMatchesTheInput) {
    LayerNorm<float> ln(std::vector<size_t>{2, 3});
    Tensor<float> in({2, 3});
    IotaFillFlat(in, 1.0f);

    const Tensor<float> out = ln.Forward(in);

    float mean = 0.0f;
    float variance = 0.0f;
    BlockStats(out, 0, 6, mean, variance);
    EXPECT_NEAR(mean, 0.0f, kTol);
    EXPECT_NEAR(variance, 1.0f, 1e-3f);
}

TEST(LayerNormMultiAxis, GammaAppliesAcrossAllNormalisedAxes) {
    LayerNorm<float> ln(std::vector<size_t>{2, 2});
    ln.gamma->Fill(3.0f);
    ln.beta->Fill(1.0f);

    Tensor<float> in({2, 2, 2});
    IotaFillFlat(in, 1.0f);

    const Tensor<float> out = ln.Forward(in);

    // Every 2x2 block is {1,2,3,4}-like: normalised to +-1.341641, +-0.447214.
    EXPECT_NEAR(out(0, 0, 0), -1.341641f * 3.0f + 1.0f, kTol);
    EXPECT_NEAR(out(1, 1, 1), 1.341641f * 3.0f + 1.0f, kTol);
}

// --- validation -------------------------------------------------------------

TEST(LayerNormForward, RejectsRankSmallerThanNormalizedShape) {
    LayerNorm<float> ln(std::vector<size_t>{2, 3});
    Tensor<float> in({3}, 1.0f);

    EXPECT_THROW(ln.Forward(in), std::invalid_argument);
}

TEST(LayerNormForward, RejectsMismatchedTrailingDimensions) {
    LayerNorm<float> ln(std::vector<size_t>{2, 3});

    Tensor<float> wrong_last({4, 2, 4}, 1.0f);
    Tensor<float> wrong_middle({4, 5, 3}, 1.0f);

    EXPECT_THROW(ln.Forward(wrong_last), std::invalid_argument);
    EXPECT_THROW(ln.Forward(wrong_middle), std::invalid_argument);
}

// --- aliasing and reuse -----------------------------------------------------

TEST(LayerNormForward, DoesNotMutateOrAliasTheInput) {
    LayerNorm<float> ln(3);
    Tensor<float> in({2, 3});
    IotaFillFlat(in, 1.0f);
    const Tensor<float> before = in.Clone();

    const Tensor<float> out = ln.Forward(in);

    EXPECT_FALSE(out.SharesStorage(in));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(in(i, j), before(i, j));
        }
    }
}

TEST(LayerNormForward, HandlesANonContiguousInput) {
    LayerNorm<float> ln(2);
    Tensor<float> source({2, 3});
    IotaFillFlat(source, 1.0f);

    const Tensor<float> view = source.Transpose();
    ASSERT_FALSE(view.IsContiguous());
    ASSERT_EQ(view.Shape(), (std::vector<size_t>{3, 2}));

    const Tensor<float> out = ln.Forward(view);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{3, 2}));
    for (size_t row = 0; row < 3; ++row) {
        float mean = 0.0f;
        float variance = 0.0f;
        BlockStats(out, row * 2, 2, mean, variance);
        EXPECT_NEAR(mean, 0.0f, kTol) << "row " << row;
    }
}

TEST(LayerNormForward, IsRepeatable) {
    LayerNorm<float> ln(4);
    Tensor<float> in({2, 4});
    IotaFillFlat(in, 1.0f);

    const Tensor<float> first = ln.Forward(in);
    const Tensor<float> second = ln.Forward(in);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(first(i, j), second(i, j));
        }
    }
}

TEST(LayerNormModule, ReportsParametersThroughABasePointer) {
    std::shared_ptr<Module<float>> m =
        std::make_shared<LayerNorm<float>>(std::vector<size_t>{3});

    EXPECT_EQ(m->Parameters().size(), 2u);
}

// --- element type -----------------------------------------------------------

TEST(LayerNormElementType, WorksWithDouble) {
    LayerNorm<double> ln(4);
    Tensor<double> in({1, 4});
    IotaFillFlat(in, 1.0);

    const Tensor<double> out = ln.Forward(in);

    // {1,2,3,4}: mean 2.5, biased variance 1.25. Epsilon shifts the divisor,
    // so derive the expectation rather than hard-coding the eps-free constant.
    const double inv_stddev = 1.0 / std::sqrt(1.25 + ln.Epsilon());
    EXPECT_NEAR(out(0, 0), -1.5 * inv_stddev, 1e-12);
    EXPECT_NEAR(out(0, 3), 1.5 * inv_stddev, 1e-12);
}
