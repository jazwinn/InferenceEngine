#include <gtest/gtest.h>

#include <NeuralNet/Linear.h>
#include <NeuralNet/Module.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using math::Tensor;
using nn::Linear;
using nn::Module;

namespace {

// Fills a rank-2 tensor with 1, 2, 3, ... in row-major order.
template <typename T>
void IotaFill(Tensor<T>& t) {
    T next = T(1);
    for (size_t i = 0; i < t.Shape()[0]; ++i) {
        for (size_t j = 0; j < t.Shape()[1]; ++j) {
            t(i, j) = next++;
        }
    }
}

// A module with no parameters of its own, used to test composition.
template <typename T>
class Passthrough : public Module<T> {
public:
    Tensor<T> Forward(const Tensor<T>& input) { return input.Clone(); }
};

// A module whose only parameters come from registered submodules.
template <typename T>
class TwoLayer : public Module<T> {
public:
    TwoLayer(size_t in, size_t hidden, size_t out)
        : first(std::make_shared<Linear<T>>(in, hidden)),
          second(std::make_shared<Linear<T>>(hidden, out)) {
        this->RegisterSubmodule("first", first);
        this->RegisterSubmodule("second", second);
    }

    Tensor<T> Forward(const Tensor<T>& input) {
        return second->Forward(first->Forward(input));
    }

    std::shared_ptr<Linear<T>> first;
    std::shared_ptr<Linear<T>> second;
};

}  // namespace

// --- Module: parameter registration -----------------------------------------

TEST(Module, ReportsNoParametersWhenNoneRegistered) {
    Passthrough<float> m;
    EXPECT_TRUE(m.Parameters().empty());
}

TEST(Module, CollectsRegisteredParameters) {
    Passthrough<float> m;
    auto a = std::make_shared<Tensor<float>>(std::vector<size_t>{2, 2});
    auto b = std::make_shared<Tensor<float>>(std::vector<size_t>{3});

    m.RegisterParameter("a", a);
    m.RegisterParameter("b", b);

    const auto params = m.Parameters();
    EXPECT_EQ(params.size(), 2u);
    EXPECT_NE(std::find(params.begin(), params.end(), a), params.end());
    EXPECT_NE(std::find(params.begin(), params.end(), b), params.end());
}

TEST(Module, ParametersAreSharedNotCopied) {
    Passthrough<float> m;
    auto p = std::make_shared<Tensor<float>>(std::vector<size_t>{2, 2}, 0.0f);
    m.RegisterParameter("p", p);

    (*m.Parameters()[0])(1, 1) = 7.0f;

    EXPECT_FLOAT_EQ((*p)(1, 1), 7.0f) << "Parameters() must hand back the same tensor";
}

TEST(Module, ReregisteringTheSameNameReplaces) {
    Passthrough<float> m;
    auto first = std::make_shared<Tensor<float>>(std::vector<size_t>{2});
    auto second = std::make_shared<Tensor<float>>(std::vector<size_t>{2});

    m.RegisterParameter("w", first);
    m.RegisterParameter("w", second);

    const auto params = m.Parameters();
    ASSERT_EQ(params.size(), 1u);
    EXPECT_EQ(params[0], second);
}

TEST(Module, GathersParametersFromSubmodules) {
    Passthrough<float> parent;
    auto child = std::make_shared<Linear<float>>(3, 2);

    parent.RegisterSubmodule("child", child);

    const auto params = parent.Parameters();
    EXPECT_EQ(params.size(), 2u) << "child contributes weight and bias";
}

TEST(Module, GathersParametersRecursively) {
    TwoLayer<float> net(4, 3, 2);

    const auto params = net.Parameters();
    EXPECT_EQ(params.size(), 4u) << "two Linear layers, weight + bias each";
}

TEST(Module, CombinesOwnAndSubmoduleParameters) {
    TwoLayer<float> net(4, 3, 2);
    auto extra = std::make_shared<Tensor<float>>(std::vector<size_t>{5});
    net.RegisterParameter("extra", extra);

    const auto params = net.Parameters();
    EXPECT_EQ(params.size(), 5u);
    EXPECT_NE(std::find(params.begin(), params.end(), extra), params.end());
}

TEST(Module, ReportsParametersThroughABasePointer) {
    std::shared_ptr<Module<float>> m = std::make_shared<Linear<float>>(3, 2);

    EXPECT_EQ(m->Parameters().size(), 2u);
}

// --- Linear: construction ---------------------------------------------------

TEST(Linear, WeightIsOutputByInput) {
    Linear<float> layer(3, 2);

    ASSERT_TRUE(layer.weight);
    EXPECT_EQ(layer.weight->Shape(), (std::vector<size_t>{2, 3}));
}

TEST(Linear, BiasIsOneEntryPerOutputFeature) {
    Linear<float> layer(3, 2);

    ASSERT_TRUE(layer.bias);
    EXPECT_EQ(layer.bias->Shape(), (std::vector<size_t>{2}));
}

TEST(Linear, ParametersAreZeroInitialised) {
    Linear<float> layer(3, 2);

    for (size_t i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ((*layer.bias)(i), 0.0f);
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((*layer.weight)(i, j), 0.0f);
        }
    }
}

TEST(Linear, RegistersBothParametersByDefault) {
    Linear<float> layer(3, 2);
    EXPECT_EQ(layer.Parameters().size(), 2u);
}

TEST(Linear, WithoutBiasRegistersOnlyTheWeight) {
    Linear<float> layer(3, 2, false);

    EXPECT_EQ(layer.Parameters().size(), 1u);
    EXPECT_FALSE(layer.bias) << "use_bias=false must not allocate a bias";
}

// --- Linear: forward --------------------------------------------------------

TEST(LinearForward, ShapeIsBatchByOutputFeatures) {
    Linear<float> layer(3, 2);
    Tensor<float> input({4, 3}, 1.0f);

    const Tensor<float> out = layer.Forward(input);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{4, 2}));
}

TEST(LinearForward, ZeroWeightsGiveTheBias) {
    Linear<float> layer(3, 2);
    (*layer.bias)(0) = 5.0f;
    (*layer.bias)(1) = -5.0f;

    Tensor<float> input({2, 3}, 1.0f);
    const Tensor<float> out = layer.Forward(input);

    for (size_t row = 0; row < 2; ++row) {
        EXPECT_FLOAT_EQ(out(row, 0), 5.0f);
        EXPECT_FLOAT_EQ(out(row, 1), -5.0f);
    }
}

TEST(LinearForward, KnownProduct) {
    // weight = [[1 2 3],    input = [[1 2 3],     out = input * weightT
    //           [4 5 6]]             [1 1 1]]
    Linear<float> layer(3, 2, false);
    IotaFill(*layer.weight);

    Tensor<float> input({2, 3});
    input(0, 0) = 1.0f; input(0, 1) = 2.0f; input(0, 2) = 3.0f;
    input(1, 0) = 1.0f; input(1, 1) = 1.0f; input(1, 2) = 1.0f;

    const Tensor<float> out = layer.Forward(input);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(out(0, 0), 14.0f);   // 1*1 + 2*2 + 3*3
    EXPECT_FLOAT_EQ(out(0, 1), 32.0f);   // 1*4 + 2*5 + 3*6
    EXPECT_FLOAT_EQ(out(1, 0), 6.0f);    // 1+2+3
    EXPECT_FLOAT_EQ(out(1, 1), 15.0f);   // 4+5+6
}

TEST(LinearForward, BiasIsAddedPerOutputFeatureNotPerElement) {
    Linear<float> layer(2, 3);
    layer.weight->Fill(0.0f);
    (*layer.bias)(0) = 1.0f;
    (*layer.bias)(1) = 2.0f;
    (*layer.bias)(2) = 3.0f;

    Tensor<float> input({4, 2}, 1.0f);
    const Tensor<float> out = layer.Forward(input);

    ASSERT_EQ(out.Shape(), (std::vector<size_t>{4, 3}));
    for (size_t row = 0; row < 4; ++row) {
        EXPECT_FLOAT_EQ(out(row, 0), 1.0f) << "row " << row;
        EXPECT_FLOAT_EQ(out(row, 1), 2.0f) << "row " << row;
        EXPECT_FLOAT_EQ(out(row, 2), 3.0f) << "row " << row;
    }
}

TEST(LinearForward, WithoutBiasOmitsTheBiasTerm) {
    Linear<float> with(2, 2, true);
    Linear<float> without(2, 2, false);
    with.weight->Fill(1.0f);
    without.weight->Fill(1.0f);
    with.bias->Fill(10.0f);

    Tensor<float> input({1, 2}, 1.0f);

    EXPECT_FLOAT_EQ(with.Forward(input)(0, 0), 12.0f);
    EXPECT_FLOAT_EQ(without.Forward(input)(0, 0), 2.0f);
}

TEST(LinearForward, DoesNotMutateTheInput) {
    Linear<float> layer(3, 2);
    layer.weight->Fill(1.0f);
    layer.bias->Fill(1.0f);

    Tensor<float> input({2, 3}, 2.0f);
    const Tensor<float> before = input.Clone();

    const Tensor<float> out = layer.Forward(input);
    (void)out;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(input(i, j), before(i, j));
        }
    }
}

TEST(LinearForward, DoesNotMutateTheParameters) {
    Linear<float> layer(3, 2);
    IotaFill(*layer.weight);
    (*layer.bias)(0) = 1.0f;
    (*layer.bias)(1) = 2.0f;

    Tensor<float> input({2, 3}, 1.0f);
    layer.Forward(input);
    layer.Forward(input);

    EXPECT_FLOAT_EQ((*layer.weight)(0, 0), 1.0f);
    EXPECT_FLOAT_EQ((*layer.weight)(1, 2), 6.0f);
    EXPECT_FLOAT_EQ((*layer.bias)(0), 1.0f);
    EXPECT_FLOAT_EQ((*layer.bias)(1), 2.0f);
}

TEST(LinearForward, OutputDoesNotAliasTheParameters) {
    Linear<float> layer(2, 2);
    Tensor<float> input({1, 2}, 1.0f);

    const Tensor<float> out = layer.Forward(input);

    EXPECT_FALSE(out.SharesStorage(*layer.weight));
    EXPECT_FALSE(out.SharesStorage(*layer.bias));
    EXPECT_FALSE(out.SharesStorage(input));
}

TEST(LinearForward, IsRepeatable) {
    Linear<float> layer(3, 2);
    IotaFill(*layer.weight);
    layer.bias->Fill(0.5f);

    Tensor<float> input({2, 3}, 1.5f);

    const Tensor<float> first = layer.Forward(input);
    const Tensor<float> second = layer.Forward(input);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(first(i, j), second(i, j));
        }
    }
}

TEST(LinearForward, SingleRowBatch) {
    Linear<float> layer(3, 1, false);
    layer.weight->Fill(2.0f);

    Tensor<float> input({1, 3}, 1.0f);
    const Tensor<float> out = layer.Forward(input);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{1, 1}));
    EXPECT_FLOAT_EQ(out(0, 0), 6.0f);
}

TEST(LinearForward, RejectsMismatchedInputWidth) {
    Linear<float> layer(3, 2);
    Tensor<float> wrong({2, 4}, 1.0f);

    EXPECT_THROW(layer.Forward(wrong), std::invalid_argument);
}

TEST(LinearForward, RejectsRankOneInput) {
    Linear<float> layer(3, 2);
    Tensor<float> vector_input({3}, 1.0f);

    EXPECT_THROW(layer.Forward(vector_input), std::invalid_argument);
}

// --- composition ------------------------------------------------------------

TEST(LinearComposition, StackedLayersProduceTheFinalShape) {
    TwoLayer<float> net(4, 3, 2);
    Tensor<float> input({5, 4}, 1.0f);

    const Tensor<float> out = net.Forward(input);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{5, 2}));
}

TEST(LinearComposition, MutatingParametersThroughParametersListAffectsForward) {
    TwoLayer<float> net(2, 2, 1);
    net.first->weight->Fill(1.0f);
    net.second->weight->Fill(1.0f);

    Tensor<float> input({1, 2}, 1.0f);
    EXPECT_FLOAT_EQ(net.Forward(input)(0, 0), 4.0f);

    for (auto& p : net.Parameters()) {
        p->Fill(0.0f);
    }
    EXPECT_FLOAT_EQ(net.Forward(input)(0, 0), 0.0f);
}

// --- element type -----------------------------------------------------------

TEST(LinearElementType, WorksWithDouble) {
    Linear<double> layer(2, 2, false);
    layer.weight->Fill(0.5);

    Tensor<double> input({1, 2}, 2.0);
    const Tensor<double> out = layer.Forward(input);

    EXPECT_DOUBLE_EQ(out(0, 0), 2.0);
}
