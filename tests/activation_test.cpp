#include <gtest/gtest.h>

#include <NeuralNet/Activation.h>
#include <NeuralNet/Module.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using nn::GELU;
using nn::ReLU;
using nn::Sigmoid;
using nn::SiLU;
using nn::Softmax;
using nn::SwiGLU;
using nn::Tanh;
using math::Tensor;
using nn::Module;

namespace {

constexpr float kTol = 1e-5f;

Tensor<float> RowVector(std::initializer_list<float> values) {
    Tensor<float> t({1, values.size()});
    size_t j = 0;
    for (float v : values) {
        t(0, j++) = v;
    }
    return t;
}

float Silu(float x) { return x / (1.0f + std::exp(-x)); }

std::shared_ptr<Tensor<float>> NamedWeight(const Module<float> &module, const std::string &name) {
    for (const auto &param : module.NamedParameters()) {
        if (param.name == name) return param.tensor;
    }
    return nullptr;
}

void SetMatrix(Tensor<float> &w, std::initializer_list<std::initializer_list<float>> rows) {
    size_t i = 0;
    for (const auto &row : rows) {
        size_t j = 0;
        for (float v : row) w(i, j++) = v;
        ++i;
    }
}

}  // namespace

// --- ReLU -------------------------------------------------------------------

TEST(ReLUForward, ClampsNegativesAndKeepsPositives) {
    ReLU<float> relu;
    const Tensor<float> in = RowVector({-2.0f, -0.5f, 0.0f, 0.5f, 3.0f});

    const Tensor<float> out = relu.Forward(in);

    EXPECT_FLOAT_EQ(out(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(out(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(out(0, 2), 0.0f);
    EXPECT_FLOAT_EQ(out(0, 3), 0.5f);
    EXPECT_FLOAT_EQ(out(0, 4), 3.0f);
}

TEST(ReLUForward, PreservesShape) {
    ReLU<float> relu;
    Tensor<float> in({2, 3, 4}, -1.0f);

    const Tensor<float> out = relu.Forward(in);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_FLOAT_EQ(out(1, 2, 3), 0.0f);
}

TEST(ReLUForward, IsIdempotent) {
    ReLU<float> relu;
    const Tensor<float> in = RowVector({-1.0f, 2.0f, -3.0f});

    const Tensor<float> once = relu.Forward(in);
    const Tensor<float> twice = relu.Forward(once);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_FLOAT_EQ(once(0, j), twice(0, j));
    }
}

TEST(ReLUForward, DoesNotMutateOrAliasTheInput) {
    ReLU<float> relu;
    Tensor<float> in = RowVector({-1.0f, 2.0f});

    const Tensor<float> out = relu.Forward(in);

    EXPECT_FLOAT_EQ(in(0, 0), -1.0f) << "input must be untouched";
    EXPECT_FALSE(out.SharesStorage(in));
}

TEST(ReLUForward, HandlesANonContiguousInput) {
    ReLU<float> relu;
    Tensor<float> source({2, 3});
    source(0, 0) = -1.0f; source(0, 1) = 2.0f;  source(0, 2) = -3.0f;
    source(1, 0) = 4.0f;  source(1, 1) = -5.0f; source(1, 2) = 6.0f;

    const Tensor<float> view = source.Transpose();
    ASSERT_FALSE(view.IsContiguous());

    const Tensor<float> out = relu.Forward(view);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{3, 2}));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(out(i, j), std::max(0.0f, view(i, j)))
                << "at (" << i << ", " << j << ")";
        }
    }
}

// --- Sigmoid ----------------------------------------------------------------

TEST(SigmoidForward, KnownValues) {
    Sigmoid<float> sigmoid;
    const Tensor<float> in = RowVector({0.0f, 1.0f, -1.0f, 2.0f});

    const Tensor<float> out = sigmoid.Forward(in);

    EXPECT_NEAR(out(0, 0), 0.5f, kTol);
    EXPECT_NEAR(out(0, 1), 0.7310586f, kTol);
    EXPECT_NEAR(out(0, 2), 0.2689414f, kTol);
    EXPECT_NEAR(out(0, 3), 0.8807971f, kTol);
}

TEST(SigmoidForward, StaysStrictlyWithinZeroAndOne) {
    Sigmoid<float> sigmoid;
    const Tensor<float> in = RowVector({-50.0f, -5.0f, 0.0f, 5.0f, 50.0f});

    const Tensor<float> out = sigmoid.Forward(in);

    for (size_t j = 0; j < 5; ++j) {
        EXPECT_GE(out(0, j), 0.0f);
        EXPECT_LE(out(0, j), 1.0f);
        EXPECT_FALSE(std::isnan(out(0, j)));
    }
}

TEST(SigmoidForward, IsSymmetricAboutTheOrigin) {
    Sigmoid<float> sigmoid;
    const Tensor<float> in = RowVector({0.7f, -0.7f, 3.2f, -3.2f});

    const Tensor<float> out = sigmoid.Forward(in);

    EXPECT_NEAR(out(0, 0) + out(0, 1), 1.0f, kTol);
    EXPECT_NEAR(out(0, 2) + out(0, 3), 1.0f, kTol);
}

TEST(SigmoidForward, IsMonotonicIncreasing) {
    Sigmoid<float> sigmoid;
    const Tensor<float> in = RowVector({-3.0f, -1.0f, 0.0f, 1.0f, 3.0f});

    const Tensor<float> out = sigmoid.Forward(in);

    for (size_t j = 1; j < 5; ++j) {
        EXPECT_GT(out(0, j), out(0, j - 1));
    }
}

// --- Tanh -------------------------------------------------------------------

TEST(TanhForward, KnownValues) {
    Tanh<float> tanh_layer;
    const Tensor<float> in = RowVector({0.0f, 1.0f, -1.0f, 0.5f});

    const Tensor<float> out = tanh_layer.Forward(in);

    EXPECT_NEAR(out(0, 0), 0.0f, kTol);
    EXPECT_NEAR(out(0, 1), 0.7615942f, kTol);
    EXPECT_NEAR(out(0, 2), -0.7615942f, kTol);
    EXPECT_NEAR(out(0, 3), 0.4621172f, kTol);
}

TEST(TanhForward, IsAnOddFunction) {
    Tanh<float> tanh_layer;
    const Tensor<float> in = RowVector({2.5f, -2.5f});

    const Tensor<float> out = tanh_layer.Forward(in);

    EXPECT_NEAR(out(0, 0), -out(0, 1), kTol);
}

TEST(TanhForward, SaturatesWithinPlusMinusOne) {
    Tanh<float> tanh_layer;
    const Tensor<float> in = RowVector({-40.0f, -1.0f, 0.0f, 1.0f, 40.0f});

    const Tensor<float> out = tanh_layer.Forward(in);

    for (size_t j = 0; j < 5; ++j) {
        EXPECT_GE(out(0, j), -1.0f);
        EXPECT_LE(out(0, j), 1.0f);
        EXPECT_FALSE(std::isnan(out(0, j)));
    }
}

// --- GELU -------------------------------------------------------------------

TEST(GELUForward, MatchesTheTanhApproximation) {
    GELU<float> gelu;
    const Tensor<float> in = RowVector({-3.0f, -1.0f, 0.0f, 0.5f, 1.0f, 3.0f});

    const Tensor<float> out = gelu.Forward(in);

    // Reference values from torch.nn.functional.gelu(x, approximate="tanh").
    EXPECT_NEAR(out(0, 0), -0.0036374f, kTol);
    EXPECT_NEAR(out(0, 1), -0.1588080f, kTol);
    EXPECT_NEAR(out(0, 2), 0.0f, kTol);
    EXPECT_NEAR(out(0, 3), 0.3457140f, kTol);
    EXPECT_NEAR(out(0, 4), 0.8411920f, kTol);
    EXPECT_NEAR(out(0, 5), 2.9963626f, kTol);
}

TEST(GELUForward, PreservesShapeAndLeavesTheInputUntouched) {
    GELU<float> gelu;
    const Tensor<float> in({2, 3, 4}, 1.0f);

    const Tensor<float> out = gelu.Forward(in);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_NEAR(out(1, 2, 3), 0.8411920f, kTol);
    EXPECT_FLOAT_EQ(in(1, 2, 3), 1.0f);
}

TEST(GELUForward, MoreReferenceValues) {
    GELU<float> gelu;
    const Tensor<float> in = RowVector({-2.0f, -0.5f, 2.0f, 5.0f});

    const Tensor<float> out = gelu.Forward(in);

    // torch.nn.functional.gelu(x, approximate="tanh")
    EXPECT_NEAR(out(0, 0), -0.0454023f, kTol);
    EXPECT_NEAR(out(0, 1), -0.1542860f, kTol);
    EXPECT_NEAR(out(0, 2), 1.9545977f, kTol);
    EXPECT_NEAR(out(0, 3), 5.0000000f, kTol);
}

TEST(GELUForward, UsesTheTanhApproximationNotTheExactErfForm) {
    // GPT-2 was trained with the tanh form; the exact erf form gives
    // 0.8413447 at x = 1, which is 1.5e-4 away.
    GELU<float> gelu;
    const Tensor<float> out = gelu.Forward(RowVector({1.0f}));

    const float exact = 0.5f * (1.0f + std::erf(1.0f / std::sqrt(2.0f)));
    EXPECT_GT(std::fabs(out(0, 0) - exact), 1e-4f);
    EXPECT_NEAR(out(0, 0), 0.8411920f, kTol);
}

TEST(GELUForward, DifferenceWithItsMirrorIsTheInput) {
    // 0.5x(1 + tanh(u)) - 0.5(-x)(1 - tanh(u)) = x, for any x.
    GELU<float> gelu;
    const std::vector<float> xs = {-4.0f, -1.3f, -0.2f, 0.0f, 0.7f, 2.5f, 6.0f};
    Tensor<float> pos({1, xs.size()});
    Tensor<float> neg({1, xs.size()});
    for (size_t j = 0; j < xs.size(); ++j) {
        pos(0, j) = xs[j];
        neg(0, j) = -xs[j];
    }

    const Tensor<float> a = gelu.Forward(pos);
    const Tensor<float> b = gelu.Forward(neg);

    for (size_t j = 0; j < xs.size(); ++j)
        EXPECT_NEAR(a(0, j) - b(0, j), xs[j], kTol) << "x = " << xs[j];
}

TEST(GELUForward, LiesBetweenZeroAndTheInput) {
    GELU<float> gelu;
    Tensor<float> in({1, 81});
    for (size_t j = 0; j < 81; ++j) in(0, j) = -4.0f + 0.1f * static_cast<float>(j);

    const Tensor<float> out = gelu.Forward(in);

    for (size_t j = 0; j < 81; ++j) {
        const float x = in(0, j);
        EXPECT_LE(out(0, j), std::max(x, 0.0f) + kTol) << "x = " << x;
        EXPECT_GE(out(0, j), std::min(x, 0.0f) - kTol) << "x = " << x;
    }
}

TEST(GELUForward, NeverDipsBelowItsMinimum) {
    // The tanh form bottoms out at about -0.170 near x = -0.75.
    GELU<float> gelu;
    Tensor<float> in({1, 201});
    for (size_t j = 0; j < 201; ++j) in(0, j) = -5.0f + 0.05f * static_cast<float>(j);

    const Tensor<float> out = gelu.Forward(in);

    float lowest = out(0, 0);
    for (size_t j = 1; j < 201; ++j) lowest = std::min(lowest, out(0, j));
    EXPECT_NEAR(lowest, -0.170f, 1e-3f);
}

TEST(GELUForward, IsIncreasingAboveItsMinimum) {
    GELU<float> gelu;
    Tensor<float> in({1, 60});
    for (size_t j = 0; j < 60; ++j) in(0, j) = -0.7f + 0.1f * static_cast<float>(j);

    const Tensor<float> out = gelu.Forward(in);

    for (size_t j = 1; j < 60; ++j)
        EXPECT_GT(out(0, j), out(0, j - 1)) << "between x = " << in(0, j - 1) << " and " << in(0, j);
}

TEST(GELUForward, SaturatesWithoutNaN) {
    GELU<float> gelu;
    const Tensor<float> in = RowVector({-100.0f, -20.0f, 20.0f, 100.0f});

    const Tensor<float> out = gelu.Forward(in);

    for (size_t j = 0; j < 4; ++j) ASSERT_FALSE(std::isnan(out(0, j))) << "at " << j;
    EXPECT_NEAR(out(0, 0), 0.0f, kTol);
    EXPECT_NEAR(out(0, 1), 0.0f, kTol);
    EXPECT_FLOAT_EQ(out(0, 2), 20.0f);
    EXPECT_FLOAT_EQ(out(0, 3), 100.0f);
}

TEST(GELUForward, HandlesANonContiguousInput) {
    GELU<float> gelu;
    Tensor<float> source({2, 3});
    source(0, 0) = -1.0f; source(0, 1) = 0.5f;  source(0, 2) = 3.0f;
    source(1, 0) = 1.0f;  source(1, 1) = -3.0f; source(1, 2) = 0.0f;

    const Tensor<float> view = source.Transpose();
    ASSERT_FALSE(view.IsContiguous());

    const Tensor<float> out = gelu.Forward(view);
    const Tensor<float> expected = gelu.Forward(view.Contiguous());

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{3, 2}));
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_FLOAT_EQ(out(i, j), expected(i, j)) << "at (" << i << ", " << j << ")";
    EXPECT_NEAR(out(0, 0), -0.1588080f, kTol);   // source(0, 0) = -1
    EXPECT_NEAR(out(2, 0), 2.9963626f, kTol);    // source(0, 2) = 3
}

TEST(GELUForward, WorksWithDouble) {
    GELU<double> gelu;
    Tensor<double> in({1, 2});
    in(0, 0) = 1.0;
    in(0, 1) = -1.0;

    const Tensor<double> out = gelu.Forward(in);

    EXPECT_NEAR(out(0, 0), 0.8411919906082768, 1e-12);
    EXPECT_NEAR(out(0, 1), -0.1588080093917232, 1e-12);
}

TEST(GELUModule, HasNoParameters) {
    EXPECT_TRUE(GELU<float>().Parameters().empty());
}

// --- SiLU -------------------------------------------------------------------

TEST(SiLUForward, KnownValues) {
    SiLU<float> silu;
    const Tensor<float> in = RowVector({-3.0f, -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f});

    const Tensor<float> out = silu.Forward(in);

    EXPECT_NEAR(out(0, 0), -0.1422776f, kTol);
    EXPECT_NEAR(out(0, 1), -0.2384058f, kTol);
    EXPECT_NEAR(out(0, 2), -0.2689414f, kTol);
    EXPECT_NEAR(out(0, 3), 0.0f, kTol);
    EXPECT_NEAR(out(0, 4), 0.3112297f, kTol);
    EXPECT_NEAR(out(0, 5), 0.7310586f, kTol);
    EXPECT_NEAR(out(0, 6), 1.7615942f, kTol);
    EXPECT_NEAR(out(0, 7), 2.8577223f, kTol);
}

TEST(SiLUForward, EqualsInputTimesSigmoid) {
    SiLU<float> silu;
    Sigmoid<float> sigmoid;
    Tensor<float> in({1, 41});
    for (size_t j = 0; j < 41; ++j) in(0, j) = -4.0f + 0.2f * static_cast<float>(j);

    const Tensor<float> out = silu.Forward(in);
    const Tensor<float> sig = sigmoid.Forward(in);

    for (size_t j = 0; j < 41; ++j)
        EXPECT_NEAR(out(0, j), in(0, j) * sig(0, j), kTol) << "x = " << in(0, j);
}

TEST(SiLUForward, DifferenceWithItsMirrorIsTheInput) {
    SiLU<float> silu;
    const std::vector<float> xs = {-4.0f, -1.3f, -0.2f, 0.0f, 0.7f, 2.5f, 6.0f};
    Tensor<float> pos({1, xs.size()});
    Tensor<float> neg({1, xs.size()});
    for (size_t j = 0; j < xs.size(); ++j) {
        pos(0, j) = xs[j];
        neg(0, j) = -xs[j];
    }

    const Tensor<float> a = silu.Forward(pos);
    const Tensor<float> b = silu.Forward(neg);

    for (size_t j = 0; j < xs.size(); ++j)
        EXPECT_NEAR(a(0, j) - b(0, j), xs[j], kTol) << "x = " << xs[j];
}

TEST(SiLUForward, NeverDipsBelowItsMinimum) {
    SiLU<float> silu;
    Tensor<float> in({1, 201});
    for (size_t j = 0; j < 201; ++j) in(0, j) = -5.0f + 0.05f * static_cast<float>(j);

    const Tensor<float> out = silu.Forward(in);

    float lowest = out(0, 0);
    for (size_t j = 1; j < 201; ++j) lowest = std::min(lowest, out(0, j));
    EXPECT_NEAR(lowest, -0.2784645f, 1e-3f);
}

TEST(SiLUForward, SaturatesWithoutNaN) {
    SiLU<float> silu;
    const Tensor<float> in = RowVector({-100.0f, -20.0f, 20.0f, 100.0f});

    const Tensor<float> out = silu.Forward(in);

    for (size_t j = 0; j < 4; ++j) ASSERT_FALSE(std::isnan(out(0, j))) << "at " << j;
    EXPECT_NEAR(out(0, 0), 0.0f, kTol);
    EXPECT_NEAR(out(0, 1), 0.0f, kTol);
    EXPECT_FLOAT_EQ(out(0, 2), 20.0f);
    EXPECT_FLOAT_EQ(out(0, 3), 100.0f);
}

TEST(SiLUForward, PreservesShapeAndLeavesTheInputUntouched) {
    SiLU<float> silu;
    const Tensor<float> in({2, 3, 4}, 2.0f);

    const Tensor<float> out = silu.Forward(in);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_NEAR(out(1, 2, 3), 1.7615942f, kTol);
    EXPECT_FLOAT_EQ(in(1, 2, 3), 2.0f);
    EXPECT_FALSE(out.SharesStorage(in));
}

TEST(SiLUForward, HandlesANonContiguousInput) {
    SiLU<float> silu;
    Tensor<float> source({2, 3});
    source(0, 0) = -1.0f; source(0, 1) = 0.5f;  source(0, 2) = 3.0f;
    source(1, 0) = 1.0f;  source(1, 1) = -3.0f; source(1, 2) = 0.0f;

    const Tensor<float> view = source.Transpose();
    ASSERT_FALSE(view.IsContiguous());

    const Tensor<float> out = silu.Forward(view);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{3, 2}));
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_NEAR(out(i, j), Silu(view(i, j)), kTol) << "at (" << i << ", " << j << ")";
}

TEST(SiLUForward, WorksWithDouble) {
    SiLU<double> silu;
    Tensor<double> in({1, 2});
    in(0, 0) = 1.0;
    in(0, 1) = -1.0;

    const Tensor<double> out = silu.Forward(in);

    EXPECT_NEAR(out(0, 0), 0.7310585786300049, 1e-12);
    EXPECT_NEAR(out(0, 1), -0.2689414213699951, 1e-12);
}

TEST(SiLUModule, HasNoParameters) {
    EXPECT_TRUE(SiLU<float>().Parameters().empty());
}

// --- SwiGLU -----------------------------------------------------------------

TEST(SwiGLUModule, RegistersThreeBiasFreeProjections) {
    SwiGLU<float> swiglu(4, 6);

    EXPECT_EQ(swiglu.Parameters().size(), 3u);

    const auto gate = NamedWeight(swiglu, "gate_proj.weight");
    const auto up = NamedWeight(swiglu, "up_proj.weight");
    const auto down = NamedWeight(swiglu, "down_proj.weight");
    ASSERT_NE(gate, nullptr);
    ASSERT_NE(up, nullptr);
    ASSERT_NE(down, nullptr);

    EXPECT_EQ(gate->Shape(), (std::vector<size_t>{6, 4}));
    EXPECT_EQ(up->Shape(), (std::vector<size_t>{6, 4}));
    EXPECT_EQ(down->Shape(), (std::vector<size_t>{4, 6}));
}

TEST(SwiGLUModule, GateAndUpAreDistinctTensors) {
    SwiGLU<float> swiglu(2, 3);

    EXPECT_NE(NamedWeight(swiglu, "gate_proj.weight"), NamedWeight(swiglu, "up_proj.weight"));
}

TEST(SwiGLUForward, IdentityWeightsGiveSiluTimesInput) {
    SwiGLU<float> swiglu(2, 2);
    SetMatrix(*NamedWeight(swiglu, "gate_proj.weight"), {{1, 0}, {0, 1}});
    SetMatrix(*NamedWeight(swiglu, "up_proj.weight"), {{1, 0}, {0, 1}});
    SetMatrix(*NamedWeight(swiglu, "down_proj.weight"), {{1, 0}, {0, 1}});

    const Tensor<float> out = swiglu.Forward(RowVector({1.0f, 2.0f}));

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{1, 2}));
    EXPECT_NEAR(out(0, 0), 0.7310586f, kTol);
    EXPECT_NEAR(out(0, 1), 3.5231884f, kTol);
}

TEST(SwiGLUForward, SiluIsAppliedToTheGateNotTheUpProjection) {
    SwiGLU<float> swiglu(2, 2);
    SetMatrix(*NamedWeight(swiglu, "gate_proj.weight"), {{1, 0}, {0, 1}});
    SetMatrix(*NamedWeight(swiglu, "up_proj.weight"), {{2, 0}, {0, 2}});
    SetMatrix(*NamedWeight(swiglu, "down_proj.weight"), {{1, 0}, {0, 1}});

    const Tensor<float> out = swiglu.Forward(RowVector({1.0f, -1.0f}));

    EXPECT_NEAR(out(0, 0), Silu(1.0f) * 2.0f, kTol);
    EXPECT_NEAR(out(0, 1), Silu(-1.0f) * -2.0f, kTol);
    EXPECT_GT(std::fabs(out(0, 0) - Silu(2.0f) * 1.0f), 1e-2f) << "gate and up look swapped";
}

TEST(SwiGLUForward, DownProjectionMapsBackToModelWidth) {
    SwiGLU<float> swiglu(2, 2);
    SetMatrix(*NamedWeight(swiglu, "gate_proj.weight"), {{1, 0}, {0, 1}});
    SetMatrix(*NamedWeight(swiglu, "up_proj.weight"), {{1, 0}, {0, 1}});
    SetMatrix(*NamedWeight(swiglu, "down_proj.weight"), {{0, 1}, {1, 0}});

    const Tensor<float> out = swiglu.Forward(RowVector({1.0f, 2.0f}));

    EXPECT_NEAR(out(0, 0), 3.5231884f, kTol);
    EXPECT_NEAR(out(0, 1), 0.7310586f, kTol);
}

TEST(SwiGLUForward, HandComputedWithAWiderHiddenLayer) {
    SwiGLU<float> swiglu(2, 3);
    SetMatrix(*NamedWeight(swiglu, "gate_proj.weight"), {{1, 0}, {0, 1}, {1, 1}});
    SetMatrix(*NamedWeight(swiglu, "up_proj.weight"), {{1, 1}, {1, 0}, {0, 1}});
    SetMatrix(*NamedWeight(swiglu, "down_proj.weight"), {{1, 0, 0}, {0, 1, 1}});

    const Tensor<float> out = swiglu.Forward(RowVector({1.0f, 1.0f}));

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{1, 2}));
    EXPECT_NEAR(out(0, 0), 1.4621172f, kTol);
    EXPECT_NEAR(out(0, 1), 2.4926528f, kTol);
}

TEST(SwiGLUForward, ZeroInputGivesZeroOutput) {
    SwiGLU<float> swiglu(3, 5);
    for (const auto &param : swiglu.Parameters()) {
        std::fill(param->Data(), param->Data() + param->Size(), 0.37f);
    }

    const Tensor<float> out = swiglu.Forward(Tensor<float>({4, 3}, 0.0f));

    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 3; ++j) EXPECT_NEAR(out(i, j), 0.0f, kTol);
}

TEST(SwiGLUForward, ProcessesEachTokenIndependently) {
    SwiGLU<float> swiglu(2, 3);
    SetMatrix(*NamedWeight(swiglu, "gate_proj.weight"), {{1, 0}, {0, 1}, {1, 1}});
    SetMatrix(*NamedWeight(swiglu, "up_proj.weight"), {{1, 1}, {1, 0}, {0, 1}});
    SetMatrix(*NamedWeight(swiglu, "down_proj.weight"), {{1, 0, 0}, {0, 1, 1}});

    Tensor<float> batch({2, 3, 2});
    for (size_t b = 0; b < 2; ++b)
        for (size_t t = 0; t < 3; ++t) {
            batch(b, t, 0) = 0.3f * static_cast<float>(b + 1) - 0.2f * static_cast<float>(t);
            batch(b, t, 1) = -0.5f + 0.4f * static_cast<float>(t);
        }

    const Tensor<float> out = swiglu.Forward(batch);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{2, 3, 2}));
    for (size_t b = 0; b < 2; ++b)
        for (size_t t = 0; t < 3; ++t) {
            const Tensor<float> single = swiglu.Forward(RowVector({batch(b, t, 0), batch(b, t, 1)}));
            EXPECT_NEAR(out(b, t, 0), single(0, 0), kTol) << "at (" << b << ", " << t << ")";
            EXPECT_NEAR(out(b, t, 1), single(0, 1), kTol) << "at (" << b << ", " << t << ")";
        }
}

TEST(SwiGLUForward, LeavesTheInputUntouched) {
    SwiGLU<float> swiglu(2, 2);
    SetMatrix(*NamedWeight(swiglu, "gate_proj.weight"), {{1, 0}, {0, 1}});
    SetMatrix(*NamedWeight(swiglu, "up_proj.weight"), {{1, 0}, {0, 1}});
    SetMatrix(*NamedWeight(swiglu, "down_proj.weight"), {{1, 0}, {0, 1}});
    const Tensor<float> in = RowVector({1.0f, -2.0f});

    const Tensor<float> out = swiglu.Forward(in);

    EXPECT_FLOAT_EQ(in(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(in(0, 1), -2.0f);
    EXPECT_FALSE(out.SharesStorage(in));
}

// --- Softmax ----------------------------------------------------------------

TEST(SoftmaxForward, KnownDistribution) {
    Softmax<float> softmax;
    const Tensor<float> in = RowVector({1.0f, 2.0f, 3.0f});

    const Tensor<float> out = softmax.Forward(in);

    EXPECT_NEAR(out(0, 0), 0.0900306f, kTol);
    EXPECT_NEAR(out(0, 1), 0.2447285f, kTol);
    EXPECT_NEAR(out(0, 2), 0.6652409f, kTol);
}

TEST(SoftmaxForward, EachRowSumsToOne) {
    Softmax<float> softmax;
    Tensor<float> in({3, 4});
    float next = -3.0f;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            in(i, j) = next;
            next += 0.7f;
        }
    }

    const Tensor<float> out = softmax.Forward(in);

    for (size_t i = 0; i < 3; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < 4; ++j) {
            sum += out(i, j);
        }
        EXPECT_NEAR(sum, 1.0f, kTol) << "row " << i;
    }
}

TEST(SoftmaxForward, NormalisesRowsIndependently) {
    Softmax<float> softmax;
    Tensor<float> in({2, 2});
    in(0, 0) = 0.0f;   in(0, 1) = 0.0f;
    in(1, 0) = 100.0f; in(1, 1) = 0.0f;

    const Tensor<float> out = softmax.Forward(in);

    EXPECT_NEAR(out(0, 0), 0.5f, kTol) << "first row must not see the second";
    EXPECT_NEAR(out(0, 1), 0.5f, kTol);
    EXPECT_NEAR(out(1, 0), 1.0f, kTol);
    EXPECT_NEAR(out(1, 1), 0.0f, kTol);
}

TEST(SoftmaxForward, AllOutputsArePositive) {
    Softmax<float> softmax;
    const Tensor<float> in = RowVector({-20.0f, 0.0f, 20.0f});

    const Tensor<float> out = softmax.Forward(in);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_GT(out(0, j), 0.0f);
        EXPECT_LE(out(0, j), 1.0f);
    }
}

TEST(SoftmaxForward, PreservesOrdering) {
    Softmax<float> softmax;
    const Tensor<float> in = RowVector({0.5f, 2.0f, -1.0f, 1.0f});

    const Tensor<float> out = softmax.Forward(in);

    EXPECT_GT(out(0, 1), out(0, 3));
    EXPECT_GT(out(0, 3), out(0, 0));
    EXPECT_GT(out(0, 0), out(0, 2));
}

TEST(SoftmaxForward, UniformLogitsGiveAUniformDistribution) {
    Softmax<float> softmax;
    Tensor<float> in({2, 5}, 7.0f);

    const Tensor<float> out = softmax.Forward(in);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            EXPECT_NEAR(out(i, j), 0.2f, kTol);
        }
    }
}

TEST(SoftmaxForward, IsInvariantToAConstantShift) {
    Softmax<float> softmax;
    const Tensor<float> base = RowVector({1.0f, 2.0f, 3.0f});
    const Tensor<float> shifted = RowVector({101.0f, 102.0f, 103.0f});

    const Tensor<float> a = softmax.Forward(base);
    const Tensor<float> b = softmax.Forward(shifted);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(a(0, j), b(0, j), kTol);
    }
}

TEST(SoftmaxForward, SurvivesLargeLogitsWithoutOverflow) {
    Softmax<float> softmax;
    const Tensor<float> in = RowVector({1000.0f, 1000.0f, 999.0f});

    const Tensor<float> out = softmax.Forward(in);

    float sum = 0.0f;
    for (size_t j = 0; j < 3; ++j) {
        EXPECT_FALSE(std::isnan(out(0, j))) << "column " << j;
        EXPECT_FALSE(std::isinf(out(0, j))) << "column " << j;
        sum += out(0, j);
    }
    EXPECT_NEAR(sum, 1.0f, kTol);
}

TEST(SoftmaxForward, SurvivesLargeNegativeLogits) {
    Softmax<float> softmax;
    const Tensor<float> in = RowVector({-1000.0f, -1001.0f});

    const Tensor<float> out = softmax.Forward(in);

    EXPECT_FALSE(std::isnan(out(0, 0)));
    EXPECT_NEAR(out(0, 0) + out(0, 1), 1.0f, kTol);
}

TEST(SoftmaxForward, SingleClassCollapsesToOne) {
    Softmax<float> softmax;
    Tensor<float> in({3, 1}, 4.0f);

    const Tensor<float> out = softmax.Forward(in);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(out(i, 0), 1.0f, kTol);
    }
}

TEST(SoftmaxForward, DefaultsToTheLastAxis) {
    Softmax<float> softmax;
    EXPECT_EQ(softmax.Axis(), -1);
}

TEST(SoftmaxForward, HandlesRankOneInput) {
    Softmax<float> softmax;
    Tensor<float> in({3});
    in(0) = 1.0f; in(1) = 2.0f; in(2) = 3.0f;

    const Tensor<float> out = softmax.Forward(in);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{3}));
    EXPECT_NEAR(out(0), 0.0900306f, kTol);
    EXPECT_NEAR(out(2), 0.6652409f, kTol);
    EXPECT_NEAR(out(0) + out(1) + out(2), 1.0f, kTol);
}

TEST(SoftmaxForward, HandlesRankThreeOverTheLastAxis) {
    Softmax<float> softmax;
    Tensor<float> in({2, 3, 4});
    float next = 0.0f;
    for (size_t h = 0; h < 2; ++h)
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j) in(h, i, j) = next++ * 0.1f;

    const Tensor<float> out = softmax.Forward(in);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{2, 3, 4}));
    for (size_t h = 0; h < 2; ++h) {
        for (size_t i = 0; i < 3; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < 4; ++j) sum += out(h, i, j);
            EXPECT_NEAR(sum, 1.0f, kTol) << "at (" << h << ", " << i << ")";
        }
    }
}

TEST(SoftmaxForward, AxisZeroNormalisesDownColumns) {
    Softmax<float> softmax(0);
    Tensor<float> in({2, 3});
    in(0, 0) = 1.0f; in(0, 1) = 2.0f; in(0, 2) = 3.0f;
    in(1, 0) = 1.0f; in(1, 1) = 2.0f; in(1, 2) = 3.0f;

    const Tensor<float> out = softmax.Forward(in);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(out(0, j) + out(1, j), 1.0f, kTol) << "column " << j;
        EXPECT_NEAR(out(0, j), 0.5f, kTol) << "identical rows split evenly";
    }
}

TEST(SoftmaxForward, NegativeAndPositiveAxesAgree) {
    Tensor<float> in({2, 3, 4});
    float next = 0.0f;
    for (size_t h = 0; h < 2; ++h)
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j) in(h, i, j) = next++ * 0.3f;

    const Tensor<float> negative = Softmax<float>(-1).Forward(in);
    const Tensor<float> positive = Softmax<float>(2).Forward(in);

    for (size_t h = 0; h < 2; ++h)
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                EXPECT_FLOAT_EQ(negative(h, i, j), positive(h, i, j));
}

TEST(SoftmaxForward, MiddleAxisNormalisesAcrossThatAxisOnly) {
    Softmax<float> softmax(1);
    Tensor<float> in({2, 3, 2}, 0.0f);

    const Tensor<float> out = softmax.Forward(in);

    // Equal logits along axis 1 (length 3) give 1/3 each.
    for (size_t h = 0; h < 2; ++h) {
        for (size_t j = 0; j < 2; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < 3; ++i) sum += out(h, i, j);
            EXPECT_NEAR(sum, 1.0f, kTol);
            EXPECT_NEAR(out(h, 0, j), 1.0f / 3.0f, kTol);
        }
    }
}

TEST(SoftmaxForward, AttentionScoreShapeNormalisesPerQuery) {
    // {heads, queries, keys}: each query's distribution over keys sums to 1.
    Softmax<float> softmax(-1);
    Tensor<float> scores({2, 3, 3});
    float next = 1.0f;
    for (size_t h = 0; h < 2; ++h)
        for (size_t q = 0; q < 3; ++q)
            for (size_t k = 0; k < 3; ++k) scores(h, q, k) = next++ * 0.25f;

    const Tensor<float> weights = softmax.Forward(scores);

    for (size_t h = 0; h < 2; ++h) {
        for (size_t q = 0; q < 3; ++q) {
            float sum = 0.0f;
            for (size_t k = 0; k < 3; ++k) sum += weights(h, q, k);
            EXPECT_NEAR(sum, 1.0f, kTol) << "head " << h << " query " << q;
        }
    }
}

TEST(SoftmaxForward, RejectsAnOutOfRangeAxis) {
    Tensor<float> in({2, 3}, 1.0f);

    EXPECT_THROW(Softmax<float>(2).Forward(in), std::invalid_argument);
    EXPECT_THROW(Softmax<float>(-3).Forward(in), std::invalid_argument);
}

TEST(SoftmaxForward, DoesNotMutateOrAliasTheInput) {
    Softmax<float> softmax;
    Tensor<float> in = RowVector({1.0f, 2.0f});

    const Tensor<float> out = softmax.Forward(in);

    EXPECT_FLOAT_EQ(in(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(in(0, 1), 2.0f);
    EXPECT_FALSE(out.SharesStorage(in));
}

TEST(SoftmaxForward, HandlesANonContiguousInput) {
    Softmax<float> softmax;
    Tensor<float> source({2, 2});
    source(0, 0) = 1.0f; source(0, 1) = 3.0f;
    source(1, 0) = 2.0f; source(1, 1) = 4.0f;

    const Tensor<float> view = source.Transpose();
    ASSERT_FALSE(view.IsContiguous());

    const Tensor<float> out = softmax.Forward(view);

    for (size_t i = 0; i < 2; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < 2; ++j) {
            sum += out(i, j);
        }
        EXPECT_NEAR(sum, 1.0f, kTol) << "row " << i;
    }
    EXPECT_GT(out(0, 1), out(0, 0)) << "view row 0 is (1, 2)";
}

// --- Module interface -------------------------------------------------------

TEST(ActivationModule, HaveNoParameters) {
    EXPECT_TRUE(ReLU<float>().Parameters().empty());
    EXPECT_TRUE(Sigmoid<float>().Parameters().empty());
    EXPECT_TRUE(Tanh<float>().Parameters().empty());
    EXPECT_TRUE(Softmax<float>().Parameters().empty());
}

TEST(ActivationModule, ComposeAsASubmodule) {
    ReLU<float> relu;
    auto inner = std::make_shared<Sigmoid<float>>();
    relu.RegisterSubmodule("inner", inner);

    EXPECT_TRUE(relu.Parameters().empty()) << "activations contribute no parameters";
}

// --- element type -----------------------------------------------------------

TEST(ActivationElementType, WorksWithDouble) {
    ReLU<double> relu;
    Softmax<double> softmax;

    Tensor<double> in({1, 2});
    in(0, 0) = -1.0;
    in(0, 1) = 2.0;

    EXPECT_DOUBLE_EQ(relu.Forward(in)(0, 0), 0.0);
    EXPECT_NEAR(softmax.Forward(in)(0, 0) + softmax.Forward(in)(0, 1), 1.0, 1e-12);
}
