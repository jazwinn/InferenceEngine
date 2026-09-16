#include <gtest/gtest.h>

#include <NeuralNet/Activation.h>
#include <NeuralNet/Module.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using activation::ReLU;
using activation::Sigmoid;
using activation::Softmax;
using activation::Tanh;
using math::Tensor;
using nn::Module;

namespace {

constexpr float kTol = 1e-5f;

Tensor<float> row_vector(std::initializer_list<float> values) {
    Tensor<float> t({1, values.size()});
    size_t j = 0;
    for (float v : values) {
        t(0, j++) = v;
    }
    return t;
}

}  // namespace

// --- ReLU -------------------------------------------------------------------

TEST(ReLUForward, ClampsNegativesAndKeepsPositives) {
    ReLU<float> relu;
    const Tensor<float> in = row_vector({-2.0f, -0.5f, 0.0f, 0.5f, 3.0f});

    const Tensor<float> out = relu.forward(in);

    EXPECT_FLOAT_EQ(out(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(out(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(out(0, 2), 0.0f);
    EXPECT_FLOAT_EQ(out(0, 3), 0.5f);
    EXPECT_FLOAT_EQ(out(0, 4), 3.0f);
}

TEST(ReLUForward, PreservesShape) {
    ReLU<float> relu;
    Tensor<float> in({2, 3, 4}, -1.0f);

    const Tensor<float> out = relu.forward(in);

    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_FLOAT_EQ(out(1, 2, 3), 0.0f);
}

TEST(ReLUForward, IsIdempotent) {
    ReLU<float> relu;
    const Tensor<float> in = row_vector({-1.0f, 2.0f, -3.0f});

    const Tensor<float> once = relu.forward(in);
    const Tensor<float> twice = relu.forward(once);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_FLOAT_EQ(once(0, j), twice(0, j));
    }
}

TEST(ReLUForward, DoesNotMutateOrAliasTheInput) {
    ReLU<float> relu;
    Tensor<float> in = row_vector({-1.0f, 2.0f});

    const Tensor<float> out = relu.forward(in);

    EXPECT_FLOAT_EQ(in(0, 0), -1.0f) << "input must be untouched";
    EXPECT_FALSE(out.shares_storage(in));
}

TEST(ReLUForward, HandlesANonContiguousInput) {
    ReLU<float> relu;
    Tensor<float> source({2, 3});
    source(0, 0) = -1.0f; source(0, 1) = 2.0f;  source(0, 2) = -3.0f;
    source(1, 0) = 4.0f;  source(1, 1) = -5.0f; source(1, 2) = 6.0f;

    const Tensor<float> view = source.transpose();
    ASSERT_FALSE(view.is_contiguous());

    const Tensor<float> out = relu.forward(view);

    ASSERT_EQ(out.shape(), (std::vector<size_t>{3, 2}));
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
    const Tensor<float> in = row_vector({0.0f, 1.0f, -1.0f, 2.0f});

    const Tensor<float> out = sigmoid.forward(in);

    EXPECT_NEAR(out(0, 0), 0.5f, kTol);
    EXPECT_NEAR(out(0, 1), 0.7310586f, kTol);
    EXPECT_NEAR(out(0, 2), 0.2689414f, kTol);
    EXPECT_NEAR(out(0, 3), 0.8807971f, kTol);
}

TEST(SigmoidForward, StaysStrictlyWithinZeroAndOne) {
    Sigmoid<float> sigmoid;
    const Tensor<float> in = row_vector({-50.0f, -5.0f, 0.0f, 5.0f, 50.0f});

    const Tensor<float> out = sigmoid.forward(in);

    for (size_t j = 0; j < 5; ++j) {
        EXPECT_GE(out(0, j), 0.0f);
        EXPECT_LE(out(0, j), 1.0f);
        EXPECT_FALSE(std::isnan(out(0, j)));
    }
}

TEST(SigmoidForward, IsSymmetricAboutTheOrigin) {
    Sigmoid<float> sigmoid;
    const Tensor<float> in = row_vector({0.7f, -0.7f, 3.2f, -3.2f});

    const Tensor<float> out = sigmoid.forward(in);

    EXPECT_NEAR(out(0, 0) + out(0, 1), 1.0f, kTol);
    EXPECT_NEAR(out(0, 2) + out(0, 3), 1.0f, kTol);
}

TEST(SigmoidForward, IsMonotonicIncreasing) {
    Sigmoid<float> sigmoid;
    const Tensor<float> in = row_vector({-3.0f, -1.0f, 0.0f, 1.0f, 3.0f});

    const Tensor<float> out = sigmoid.forward(in);

    for (size_t j = 1; j < 5; ++j) {
        EXPECT_GT(out(0, j), out(0, j - 1));
    }
}

// --- Tanh -------------------------------------------------------------------

TEST(TanhForward, KnownValues) {
    Tanh<float> tanh_layer;
    const Tensor<float> in = row_vector({0.0f, 1.0f, -1.0f, 0.5f});

    const Tensor<float> out = tanh_layer.forward(in);

    EXPECT_NEAR(out(0, 0), 0.0f, kTol);
    EXPECT_NEAR(out(0, 1), 0.7615942f, kTol);
    EXPECT_NEAR(out(0, 2), -0.7615942f, kTol);
    EXPECT_NEAR(out(0, 3), 0.4621172f, kTol);
}

TEST(TanhForward, IsAnOddFunction) {
    Tanh<float> tanh_layer;
    const Tensor<float> in = row_vector({2.5f, -2.5f});

    const Tensor<float> out = tanh_layer.forward(in);

    EXPECT_NEAR(out(0, 0), -out(0, 1), kTol);
}

TEST(TanhForward, SaturatesWithinPlusMinusOne) {
    Tanh<float> tanh_layer;
    const Tensor<float> in = row_vector({-40.0f, -1.0f, 0.0f, 1.0f, 40.0f});

    const Tensor<float> out = tanh_layer.forward(in);

    for (size_t j = 0; j < 5; ++j) {
        EXPECT_GE(out(0, j), -1.0f);
        EXPECT_LE(out(0, j), 1.0f);
        EXPECT_FALSE(std::isnan(out(0, j)));
    }
}

// --- Softmax ----------------------------------------------------------------

TEST(SoftmaxForward, KnownDistribution) {
    Softmax<float> softmax;
    const Tensor<float> in = row_vector({1.0f, 2.0f, 3.0f});

    const Tensor<float> out = softmax.forward(in);

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

    const Tensor<float> out = softmax.forward(in);

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

    const Tensor<float> out = softmax.forward(in);

    EXPECT_NEAR(out(0, 0), 0.5f, kTol) << "first row must not see the second";
    EXPECT_NEAR(out(0, 1), 0.5f, kTol);
    EXPECT_NEAR(out(1, 0), 1.0f, kTol);
    EXPECT_NEAR(out(1, 1), 0.0f, kTol);
}

TEST(SoftmaxForward, AllOutputsArePositive) {
    Softmax<float> softmax;
    const Tensor<float> in = row_vector({-20.0f, 0.0f, 20.0f});

    const Tensor<float> out = softmax.forward(in);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_GT(out(0, j), 0.0f);
        EXPECT_LE(out(0, j), 1.0f);
    }
}

TEST(SoftmaxForward, PreservesOrdering) {
    Softmax<float> softmax;
    const Tensor<float> in = row_vector({0.5f, 2.0f, -1.0f, 1.0f});

    const Tensor<float> out = softmax.forward(in);

    EXPECT_GT(out(0, 1), out(0, 3));
    EXPECT_GT(out(0, 3), out(0, 0));
    EXPECT_GT(out(0, 0), out(0, 2));
}

TEST(SoftmaxForward, UniformLogitsGiveAUniformDistribution) {
    Softmax<float> softmax;
    Tensor<float> in({2, 5}, 7.0f);

    const Tensor<float> out = softmax.forward(in);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            EXPECT_NEAR(out(i, j), 0.2f, kTol);
        }
    }
}

TEST(SoftmaxForward, IsInvariantToAConstantShift) {
    Softmax<float> softmax;
    const Tensor<float> base = row_vector({1.0f, 2.0f, 3.0f});
    const Tensor<float> shifted = row_vector({101.0f, 102.0f, 103.0f});

    const Tensor<float> a = softmax.forward(base);
    const Tensor<float> b = softmax.forward(shifted);

    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(a(0, j), b(0, j), kTol);
    }
}

TEST(SoftmaxForward, SurvivesLargeLogitsWithoutOverflow) {
    Softmax<float> softmax;
    const Tensor<float> in = row_vector({1000.0f, 1000.0f, 999.0f});

    const Tensor<float> out = softmax.forward(in);

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
    const Tensor<float> in = row_vector({-1000.0f, -1001.0f});

    const Tensor<float> out = softmax.forward(in);

    EXPECT_FALSE(std::isnan(out(0, 0)));
    EXPECT_NEAR(out(0, 0) + out(0, 1), 1.0f, kTol);
}

TEST(SoftmaxForward, SingleClassCollapsesToOne) {
    Softmax<float> softmax;
    Tensor<float> in({3, 1}, 4.0f);

    const Tensor<float> out = softmax.forward(in);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(out(i, 0), 1.0f, kTol);
    }
}

TEST(SoftmaxForward, RejectsNonMatrixInput) {
    Softmax<float> softmax;
    Tensor<float> vec({4}, 1.0f);
    Tensor<float> cube({2, 2, 2}, 1.0f);

    EXPECT_THROW(softmax.forward(vec), std::invalid_argument);
    EXPECT_THROW(softmax.forward(cube), std::invalid_argument);
}

TEST(SoftmaxForward, DoesNotMutateOrAliasTheInput) {
    Softmax<float> softmax;
    Tensor<float> in = row_vector({1.0f, 2.0f});

    const Tensor<float> out = softmax.forward(in);

    EXPECT_FLOAT_EQ(in(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(in(0, 1), 2.0f);
    EXPECT_FALSE(out.shares_storage(in));
}

TEST(SoftmaxForward, HandlesANonContiguousInput) {
    Softmax<float> softmax;
    Tensor<float> source({2, 2});
    source(0, 0) = 1.0f; source(0, 1) = 3.0f;
    source(1, 0) = 2.0f; source(1, 1) = 4.0f;

    const Tensor<float> view = source.transpose();
    ASSERT_FALSE(view.is_contiguous());

    const Tensor<float> out = softmax.forward(view);

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
    EXPECT_TRUE(ReLU<float>().parameters().empty());
    EXPECT_TRUE(Sigmoid<float>().parameters().empty());
    EXPECT_TRUE(Tanh<float>().parameters().empty());
    EXPECT_TRUE(Softmax<float>().parameters().empty());
}

TEST(ActivationModule, DispatchThroughABasePointer) {
    std::vector<std::shared_ptr<Module<float>>> layers{
        std::make_shared<ReLU<float>>(),
        std::make_shared<Sigmoid<float>>(),
        std::make_shared<Tanh<float>>(),
    };

    Tensor<float> input({2, 2}, 0.5f);
    for (const auto& layer : layers) {
        const Tensor<float> out = layer->forward(input);
        EXPECT_EQ(out.shape(), input.shape());
    }
}

TEST(ActivationModule, ComposeAsASubmodule) {
    ReLU<float> relu;
    auto inner = std::make_shared<Sigmoid<float>>();
    relu.register_submodule("inner", inner);

    EXPECT_TRUE(relu.parameters().empty()) << "activations contribute no parameters";
}

// --- element type -----------------------------------------------------------

TEST(ActivationElementType, WorksWithDouble) {
    ReLU<double> relu;
    Softmax<double> softmax;

    Tensor<double> in({1, 2});
    in(0, 0) = -1.0;
    in(0, 1) = 2.0;

    EXPECT_DOUBLE_EQ(relu.forward(in)(0, 0), 0.0);
    EXPECT_NEAR(softmax.forward(in)(0, 0) + softmax.forward(in)(0, 1), 1.0, 1e-12);
}
