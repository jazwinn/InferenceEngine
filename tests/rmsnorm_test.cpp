#include <gtest/gtest.h>

#include <NeuralNet/Module.h>
#include <NeuralNet/Transformer/Normal/RMSNorm.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using tensor::Tensor;
using nn::Module;
using nn::transformer::norm::RMSNorm;

namespace {

constexpr float kTol = 1e-4f;

// sqrt(mean(x^2) + eps) for one row, computed independently of the layer.
float RootMeanSquare(std::initializer_list<float> values, float epsilon) {
    float sum = 0.0f;
    for (float v : values) sum += v * v;
    return std::sqrt(sum / static_cast<float>(values.size()) + epsilon);
}

template <typename T>
void IotaFill(Tensor<T>& t) {
    T next = T(1);
    for (size_t i = 0; i < t.Shape()[0]; ++i) {
        for (size_t j = 0; j < t.Shape()[1]; ++j) {
            t(i, j) = next++;
        }
    }
}

}  // namespace

// --- construction -----------------------------------------------------------

TEST(RMSNormConstruction, WeightHasOneEntryPerFeature) {
    RMSNorm<float> norm(4);

    ASSERT_TRUE(norm.weight);
    EXPECT_EQ(norm.weight->Shape(), (std::vector<size_t>{4}));
}

TEST(RMSNormConstruction, WeightStartsAtOne) {
    RMSNorm<float> norm(3);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ((*norm.weight)(i), 1.0f) << "at " << i;
    }
}

TEST(RMSNormConstruction, RegistersExactlyOneParameter) {
    RMSNorm<float> norm(3);

    EXPECT_EQ(norm.Parameters().size(), 1u) << "RMSNorm has no bias";
    EXPECT_EQ(norm.Parameters()[0], norm.weight);
}

TEST(RMSNormConstruction, EpsilonIsConfigurable) {
    RMSNorm<float> defaulted(3);
    RMSNorm<float> custom(3, 1e-3f);

    EXPECT_FLOAT_EQ(defaulted.epsilon, 1e-8f);
    EXPECT_FLOAT_EQ(custom.epsilon, 1e-3f);
}

// --- forward ----------------------------------------------------------------

TEST(RMSNormForward, PreservesShape) {
    RMSNorm<float> norm(3);
    Tensor<float> in({4, 3}, 1.0f);

    const Tensor<float> out = norm.Forward(in);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{4, 3}));
}

TEST(RMSNormForward, KnownValuesForASingleRow) {
    // {1, 2, 3}: mean square 14/3, rms = sqrt(14/3 + eps) ~ 2.160247
    RMSNorm<float> norm(3);
    Tensor<float> in({1, 3});
    in(0, 0) = 1.0f; in(0, 1) = 2.0f; in(0, 2) = 3.0f;

    const Tensor<float> out = norm.Forward(in);
    const float rms = RootMeanSquare({1.0f, 2.0f, 3.0f}, norm.epsilon);

    EXPECT_NEAR(out(0, 0), 1.0f / rms, kTol);
    EXPECT_NEAR(out(0, 1), 2.0f / rms, kTol);
    EXPECT_NEAR(out(0, 2), 3.0f / rms, kTol);
}

TEST(RMSNormForward, OutputHasUnitRootMeanSquare) {
    RMSNorm<float> norm(4);
    Tensor<float> in({3, 4});
    IotaFill(in);
    in(1, 2) = 50.0f;   // make the rows genuinely different

    const Tensor<float> out = norm.Forward(in);

    for (size_t row = 0; row < 3; ++row) {
        float sum = 0.0f;
        for (size_t j = 0; j < 4; ++j) sum += out(row, j) * out(row, j);
        EXPECT_NEAR(std::sqrt(sum / 4.0f), 1.0f, 1e-3f) << "row " << row;
    }
}

TEST(RMSNormForward, DoesNotCentreTheInput) {
    // Unlike LayerNorm, RMSNorm divides by scale only -- the row mean is not
    // forced to zero.
    RMSNorm<float> norm(3);
    Tensor<float> in({1, 3}, 5.0f);

    const Tensor<float> out = norm.Forward(in);

    float sum = 0.0f;
    for (size_t j = 0; j < 3; ++j) sum += out(0, j);
    EXPECT_GT(std::abs(sum), 1.0f) << "a constant positive row must stay positive";
}

TEST(RMSNormForward, RowsAreNormalisedIndependently) {
    RMSNorm<float> norm(2);
    Tensor<float> in({2, 2});
    in(0, 0) = 1.0f;   in(0, 1) = 2.0f;
    in(1, 0) = 100.0f; in(1, 1) = 200.0f;

    const Tensor<float> out = norm.Forward(in);

    EXPECT_NEAR(out(0, 0), out(1, 0), 1e-3f) << "scale must not leak between rows";
    EXPECT_NEAR(out(0, 1), out(1, 1), 1e-3f);
}

TEST(RMSNormForward, IsScaleEquivariant) {
    RMSNorm<float> norm(3);

    Tensor<float> base({1, 3});
    base(0, 0) = 1.0f; base(0, 1) = 2.0f; base(0, 2) = 3.0f;

    Tensor<float> scaled({1, 3});
    for (size_t j = 0; j < 3; ++j) scaled(0, j) = base(0, j) * 10.0f;

    const Tensor<float> a = norm.Forward(base);
    const Tensor<float> b = norm.Forward(scaled);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(a(0, j), b(0, j), 1e-3f) << "at " << j;
    }
}

TEST(RMSNormForward, WeightScalesPerFeature) {
    RMSNorm<float> norm(3);
    (*norm.weight)(0) = 0.0f;
    (*norm.weight)(1) = 2.0f;
    (*norm.weight)(2) = 1.0f;

    Tensor<float> in({2, 3});
    IotaFill(in);

    const Tensor<float> unweighted = RMSNorm<float>(3).Forward(in);
    const Tensor<float> out = norm.Forward(in);

    for (size_t row = 0; row < 2; ++row) {
        EXPECT_NEAR(out(row, 0), 0.0f, kTol) << "row " << row;
        EXPECT_NEAR(out(row, 1), unweighted(row, 1) * 2.0f, kTol) << "row " << row;
        EXPECT_NEAR(out(row, 2), unweighted(row, 2), kTol) << "row " << row;
    }
}

TEST(RMSNormForward, ZeroRowStaysFiniteThroughEpsilon) {
    RMSNorm<float> norm(3, 1e-6f);
    Tensor<float> in({1, 3}, 0.0f);

    const Tensor<float> out = norm.Forward(in);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_FALSE(std::isnan(out(0, j))) << "epsilon must prevent 0/0";
        EXPECT_NEAR(out(0, j), 0.0f, kTol);
    }
}

TEST(RMSNormForward, HandlesNegativeValues) {
    RMSNorm<float> norm(3);
    Tensor<float> in({1, 3});
    in(0, 0) = -1.0f; in(0, 1) = 2.0f; in(0, 2) = -3.0f;

    const Tensor<float> out = norm.Forward(in);
    const float rms = RootMeanSquare({-1.0f, 2.0f, -3.0f}, norm.epsilon);

    EXPECT_NEAR(out(0, 0), -1.0f / rms, kTol);
    EXPECT_NEAR(out(0, 2), -3.0f / rms, kTol);
    EXPECT_LT(out(0, 0), 0.0f) << "sign must be preserved";
}

TEST(RMSNormForward, DoesNotMutateOrAliasTheInput) {
    RMSNorm<float> norm(3);
    Tensor<float> in({2, 3});
    IotaFill(in);
    const Tensor<float> before = in.Clone();

    const Tensor<float> out = norm.Forward(in);

    EXPECT_FALSE(out.SharesStorage(in));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(in(i, j), before(i, j));
        }
    }
}

TEST(RMSNormForward, DoesNotMutateTheWeight) {
    RMSNorm<float> norm(3);
    Tensor<float> in({2, 3});
    IotaFill(in);

    norm.Forward(in);
    norm.Forward(in);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ((*norm.weight)(i), 1.0f);
    }
}

TEST(RMSNormForward, IsRepeatable) {
    RMSNorm<float> norm(3);
    Tensor<float> in({2, 3});
    IotaFill(in);

    const Tensor<float> first = norm.Forward(in);
    const Tensor<float> second = norm.Forward(in);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(first(i, j), second(i, j));
        }
    }
}

TEST(RMSNormForward, HandlesANonContiguousInput) {
    RMSNorm<float> norm(2);
    Tensor<float> source({2, 3});
    IotaFill(source);

    const Tensor<float> view = source.Transpose();   // {3,2}
    ASSERT_FALSE(view.IsContiguous());

    const Tensor<float> out = norm.Forward(view);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{3, 2}));
    for (size_t row = 0; row < 3; ++row) {
        float sum = 0.0f;
        for (size_t j = 0; j < 2; ++j) sum += out(row, j) * out(row, j);
        EXPECT_NEAR(std::sqrt(sum / 2.0f), 1.0f, 1e-3f) << "row " << row;
    }
}

TEST(RMSNormForward, HandlesRankOneInput) {
    // The reduced axis is the last one, so a bare feature vector works.
    RMSNorm<float> norm(3);
    Tensor<float> in({3});
    in(0) = 1.0f; in(1) = 2.0f; in(2) = 3.0f;

    const Tensor<float> out = norm.Forward(in);
    const float rms = RootMeanSquare({1.0f, 2.0f, 3.0f}, norm.epsilon);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{3}));
    EXPECT_NEAR(out(0), 1.0f / rms, kTol);
    EXPECT_NEAR(out(2), 3.0f / rms, kTol);
}

TEST(RMSNormForward, RankThreeNormalisesTheFeatureAxisNotTheSequenceAxis) {
    // {batch, seq, features}: each (batch, seq) position is normalised over
    // its own features, independently of every other position.
    RMSNorm<float> norm(3);
    Tensor<float> in({2, 2, 3});
    float next = 1.0f;
    for (size_t b = 0; b < 2; ++b)
        for (size_t s = 0; s < 2; ++s)
            for (size_t f = 0; f < 3; ++f) in(b, s, f) = next++;

    const Tensor<float> out = norm.Forward(in);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{2, 2, 3}));
    for (size_t b = 0; b < 2; ++b) {
        for (size_t s = 0; s < 2; ++s) {
            float sum = 0.0f;
            for (size_t f = 0; f < 3; ++f) sum += out(b, s, f) * out(b, s, f);
            EXPECT_NEAR(std::sqrt(sum / 3.0f), 1.0f, 1e-3f)
                << "at batch " << b << ", position " << s;
        }
    }
}

TEST(RMSNormForward, RankThreePositionsAreIndependent) {
    RMSNorm<float> norm(2);
    Tensor<float> in({1, 2, 2});
    in(0, 0, 0) = 1.0f;   in(0, 0, 1) = 2.0f;
    in(0, 1, 0) = 100.0f; in(0, 1, 1) = 200.0f;

    const Tensor<float> out = norm.Forward(in);

    EXPECT_NEAR(out(0, 0, 0), out(0, 1, 0), 1e-3f)
        << "a huge second position must not rescale the first";
    EXPECT_NEAR(out(0, 0, 1), out(0, 1, 1), 1e-3f);
}

// --- Module interface -------------------------------------------------------

TEST(RMSNormModule, ReportsParametersThroughABasePointer) {
    std::shared_ptr<Module<float>> m = std::make_shared<RMSNorm<float>>(3);

    EXPECT_EQ(m->Parameters().size(), 1u);
}

// --- element type -----------------------------------------------------------

TEST(RMSNormElementType, WorksWithDouble) {
    RMSNorm<double> norm(2);
    Tensor<double> in({1, 2});
    in(0, 0) = 3.0;
    in(0, 1) = 4.0;

    const Tensor<double> out = norm.Forward(in);

    // rms = sqrt((9 + 16) / 2 + eps) = sqrt(12.5 + eps)
    const double rms = std::sqrt(12.5 + norm.epsilon);
    EXPECT_NEAR(out(0, 0), 3.0 / rms, 1e-9);
    EXPECT_NEAR(out(0, 1), 4.0 / rms, 1e-9);
}
