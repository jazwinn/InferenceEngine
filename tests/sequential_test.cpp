#include <gtest/gtest.h>

#include <NeuralNet/Activation.h>
#include <NeuralNet/Linear.h>
#include <NeuralNet/Module.h>
#include <NeuralNet/Sequential.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

using activation::GELU;
using activation::ReLU;
using math::Tensor;
using nn::Linear;
using nn::Module;
using nn::Sequential;

namespace {

constexpr float kTol = 1e-5f;

// Linear zero-initialises; give it varied weights and biases so a wrong layer
// order or a skipped layer shows up in the output.
template <typename T>
std::shared_ptr<Linear<T>> SeededLinear(size_t in, size_t out, double phase) {
    auto linear = std::make_shared<Linear<T>>(in, out);
    int k = 0;
    for (const auto& p : {linear->weight, linear->bias}) {
        T* data = p->Data();
        for (size_t i = 0; i < p->Size(); ++i)
            data[i] = static_cast<T>(0.5 * std::sin(0.7 * static_cast<double>(++k) + phase));
    }
    return linear;
}

template <typename T>
Tensor<T> Wave(const std::vector<size_t>& shape) {
    Tensor<T> t(shape);
    T* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i)
        data[i] = static_cast<T>(std::sin(0.9 * static_cast<double>(i + 1)));
    return t;
}

template <typename T>
void ExpectAllNear(const Tensor<T>& a, const Tensor<T>& b, T tol) {
    ASSERT_EQ(a.Shape(), b.Shape());
    const Tensor<T> ac = a.Contiguous();
    const Tensor<T> bc = b.Contiguous();
    for (size_t i = 0; i < ac.Size(); ++i)
        EXPECT_NEAR(ac.Data()[i], bc.Data()[i], tol) << "at flat index " << i;
}

}  // namespace

// --- forward ----------------------------------------------------------------

TEST(SequentialForward, NoLayersIsTheIdentity) {
    Sequential<float> empty;
    const Tensor<float> x = Wave<float>({2, 3});

    ExpectAllNear(empty.Forward(x), x, 0.0f);
}

TEST(SequentialForward, OneLayerMatchesThatLayer) {
    auto linear = SeededLinear<float>(4, 3, 0.0);
    Sequential<float, Linear<float>> seq(linear);
    const Tensor<float> x = Wave<float>({2, 4});

    ExpectAllNear(seq.Forward(x), linear->Forward(x), 0.0f);
}

TEST(SequentialForward, AppliesLayersInOrder) {
    auto first = SeededLinear<float>(4, 5, 0.0);
    auto relu = std::make_shared<ReLU<float>>();
    auto second = SeededLinear<float>(5, 2, 1.0);
    Sequential<float, Linear<float>, ReLU<float>, Linear<float>> seq(first, relu, second);
    const Tensor<float> x = Wave<float>({3, 4});

    const Tensor<float> expected = second->Forward(relu->Forward(first->Forward(x)));

    ExpectAllNear(seq.Forward(x), expected, kTol);
}

TEST(SequentialForward, OrderChangesTheResult) {
    // ReLU before vs after a Linear with negative weights must differ.
    auto linear = SeededLinear<float>(4, 4, 0.0);
    auto relu = std::make_shared<ReLU<float>>();
    Sequential<float, Linear<float>, ReLU<float>> linear_then_relu(linear, relu);
    Sequential<float, ReLU<float>, Linear<float>> relu_then_linear(relu, linear);
    const Tensor<float> x = Wave<float>({2, 4});

    const Tensor<float> a = linear_then_relu.Forward(x);
    const Tensor<float> b = relu_then_linear.Forward(x);

    float largest = 0.0f;
    for (size_t i = 0; i < a.Size(); ++i)
        largest = std::max(largest, std::fabs(a.Data()[i] - b.Data()[i]));
    EXPECT_GT(largest, 1e-3f);
}

TEST(SequentialForward, BuildsTheGPT2FeedForwardShape) {
    // Linear(d -> 4d) -> GELU -> Linear(4d -> d) over [batch, seq, d].
    auto up = SeededLinear<float>(8, 32, 0.0);
    auto gelu = std::make_shared<GELU<float>>();
    auto down = SeededLinear<float>(32, 8, 1.0);
    Sequential<float, Linear<float>, GELU<float>, Linear<float>> mlp(up, gelu, down);
    const Tensor<float> x = Wave<float>({2, 3, 8});

    const Tensor<float> out = mlp.Forward(x);

    EXPECT_EQ(out.Shape(), (std::vector<size_t>{2, 3, 8}));
    ExpectAllNear(out, down->Forward(gelu->Forward(up->Forward(x))), kTol);
}

TEST(SequentialForward, TheSameLayerCanAppearTwice) {
    auto linear = SeededLinear<float>(3, 3, 0.0);
    Sequential<float, Linear<float>, Linear<float>> twice(linear, linear);
    const Tensor<float> x = Wave<float>({1, 3});

    ExpectAllNear(twice.Forward(x), linear->Forward(linear->Forward(x)), kTol);
}

TEST(SequentialForward, NestsInsideAnotherSequential) {
    using Inner = Sequential<float, Linear<float>, ReLU<float>>;
    auto a = SeededLinear<float>(4, 6, 0.0);
    auto relu = std::make_shared<ReLU<float>>();
    auto b = SeededLinear<float>(6, 2, 1.0);
    auto inner = std::make_shared<Inner>(a, relu);
    Sequential<float, Inner, Linear<float>> outer(inner, b);
    const Tensor<float> x = Wave<float>({2, 4});

    ExpectAllNear(outer.Forward(x), b->Forward(relu->Forward(a->Forward(x))), kTol);
}

TEST(SequentialForward, DoesNotMutateTheInput) {
    Sequential<float, Linear<float>, ReLU<float>> seq(SeededLinear<float>(4, 4, 0.0),
                                                      std::make_shared<ReLU<float>>());
    const Tensor<float> x = Wave<float>({2, 4});
    const Tensor<float> before = x.Clone();

    const Tensor<float> out = seq.Forward(x);

    EXPECT_FALSE(out.SharesStorage(x));
    ExpectAllNear(x, before, 0.0f);
}

TEST(SequentialForward, RejectsLayersWhoseSizesDoNotChain) {
    Sequential<float, Linear<float>, Linear<float>> broken(SeededLinear<float>(4, 5, 0.0),
                                                           SeededLinear<float>(6, 2, 1.0));

    EXPECT_THROW(broken.Forward(Wave<float>({1, 4})), std::invalid_argument);
}

TEST(SequentialForward, WorksWithDouble) {
    auto up = SeededLinear<double>(3, 5, 0.0);
    auto down = SeededLinear<double>(5, 2, 1.0);
    Sequential<double, Linear<double>, GELU<double>, Linear<double>> mlp(
        up, std::make_shared<GELU<double>>(), down);
    const Tensor<double> x = Wave<double>({2, 3});

    const Tensor<double> expected = down->Forward(GELU<double>().Forward(up->Forward(x)));

    ExpectAllNear(mlp.Forward(x), expected, 1e-12);
}

// --- accessing layers -------------------------------------------------------

TEST(SequentialGet, ReturnsTheLayersThatWerePassedIn) {
    auto first = SeededLinear<float>(4, 5, 0.0);
    auto gelu = std::make_shared<GELU<float>>();
    auto second = SeededLinear<float>(5, 2, 1.0);
    Sequential<float, Linear<float>, GELU<float>, Linear<float>> seq(first, gelu, second);

    EXPECT_EQ(seq.Get<0>(), first);
    EXPECT_EQ(seq.Get<1>(), gelu);
    EXPECT_EQ(seq.Get<2>(), second);
}

TEST(SequentialGet, KeepsEachLayersConcreteType) {
    Sequential<float, Linear<float>, GELU<float>> seq(SeededLinear<float>(2, 2, 0.0),
                                                      std::make_shared<GELU<float>>());

    static_assert(std::is_same_v<std::remove_reference_t<decltype(seq.Get<0>())>,
                                 std::shared_ptr<Linear<float>>>);
    static_assert(std::is_same_v<std::remove_reference_t<decltype(seq.Get<1>())>,
                                 std::shared_ptr<GELU<float>>>);
    EXPECT_EQ(seq.Get<0>()->weight->Shape(), (std::vector<size_t>{2, 2}));
}

TEST(SequentialGet, ChangingALayersWeightsChangesTheOutput) {
    // The Sequential shares the layers, it does not copy them -- which is what
    // lets a weight loader fill them in after construction.
    Sequential<float, Linear<float>> seq(SeededLinear<float>(3, 2, 0.0));
    const Tensor<float> x = Wave<float>({1, 3});

    seq.Get<0>()->weight->Fill(0.0f);
    seq.Get<0>()->bias->Fill(1.5f);

    const Tensor<float> out = seq.Forward(x);
    EXPECT_FLOAT_EQ(out(0, 0), 1.5f);
    EXPECT_FLOAT_EQ(out(0, 1), 1.5f);
}

// --- parameters -------------------------------------------------------------

TEST(SequentialParameters, CollectsEveryLayersParameters) {
    Sequential<float, Linear<float>, GELU<float>, Linear<float>> mlp(
        SeededLinear<float>(8, 32, 0.0), std::make_shared<GELU<float>>(), SeededLinear<float>(32, 8, 1.0));

    // two Linear layers, weight + bias each; GELU has none
    EXPECT_EQ(mlp.Parameters().size(), 4u);
}

TEST(SequentialParameters, ReachTheActualLayerTensors) {
    auto first = SeededLinear<float>(4, 5, 0.0);
    auto second = SeededLinear<float>(5, 2, 1.0);
    Sequential<float, Linear<float>, Linear<float>> seq(first, second);

    const auto params = seq.Parameters();
    for (const auto& expected : {first->weight, first->bias, second->weight, second->bias})
        EXPECT_NE(std::find(params.begin(), params.end(), expected), params.end());
}

TEST(SequentialParameters, IncludeNestedSequentials) {
    using Inner = Sequential<float, Linear<float>, ReLU<float>>;
    auto inner = std::make_shared<Inner>(SeededLinear<float>(4, 6, 0.0), std::make_shared<ReLU<float>>());
    Sequential<float, Inner, Linear<float>> outer(inner, SeededLinear<float>(6, 2, 1.0));

    EXPECT_EQ(outer.Parameters().size(), 4u);
}

TEST(SequentialParameters, AnEmptySequentialHasNone) {
    EXPECT_TRUE(Sequential<float>().Parameters().empty());
}

TEST(SequentialModule, ReportsParametersThroughABasePointer) {
    std::shared_ptr<Module<float>> m = std::make_shared<Sequential<float, Linear<float>, ReLU<float>>>(
        SeededLinear<float>(3, 2, 0.0), std::make_shared<ReLU<float>>());

    EXPECT_EQ(m->Parameters().size(), 2u);
}

// --- torch-style construction -----------------------------------------------

TEST(SequentialDeduction, TakesLayersByValueAndDeducesTheType) {
    Sequential model(
        Linear(10, 128),
        ReLU(),
        Linear(128, 64));

    static_assert(std::is_same_v<decltype(model),
                                 Sequential<float, Linear<float>, ReLU<float>, Linear<float>>>);
    EXPECT_EQ(model.Get<0>()->weight->Shape(), (std::vector<size_t>{128, 10}));
    EXPECT_EQ(model.Get<2>()->weight->Shape(), (std::vector<size_t>{64, 128}));
    EXPECT_EQ(model.Forward(Wave<float>({3, 10})).Shape(), (std::vector<size_t>{3, 64}));
}

TEST(SequentialDeduction, ByValueLayersStillReportTheirParameters) {
    Sequential model(Linear(4, 8), GELU(), Linear(8, 2));

    EXPECT_EQ(model.Parameters().size(), 4u);
}

TEST(SequentialDeduction, ByValueCopiesShareTheOriginalWeights) {
    // Linear keeps its weights behind shared_ptr, so passing it by value
    // copies the handles, not the numbers.
    Linear linear(3, 2);
    Sequential model(linear);

    linear.bias->Fill(2.0f);

    const Tensor<float> out = model.Forward(Wave<float>({1, 3}));
    EXPECT_FLOAT_EQ(out(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(out(0, 1), 2.0f);
}

TEST(SequentialDeduction, DeducesFromSharedPointersToo) {
    auto up = SeededLinear<float>(4, 8, 0.0);
    auto down = SeededLinear<float>(8, 2, 1.0);
    Sequential model(up, std::make_shared<GELU<float>>(), down);

    static_assert(std::is_same_v<decltype(model),
                                 Sequential<float, Linear<float>, GELU<float>, Linear<float>>>);
    EXPECT_EQ(model.Get<0>(), up);
}

TEST(SequentialDeduction, DeducesDoubleFromTheLayers) {
    Sequential model(Linear<double>(3, 2), ReLU<double>());

    static_assert(std::is_same_v<decltype(model), Sequential<double, Linear<double>, ReLU<double>>>);
    EXPECT_EQ(model.Forward(Wave<double>({1, 3})).Shape(), (std::vector<size_t>{1, 2}));
}

TEST(SequentialDeduction, LayersDefaultToFloat) {
    static_assert(std::is_same_v<Linear<>, Linear<float>>);
    static_assert(std::is_same_v<decltype(ReLU()), ReLU<float>>);
    static_assert(std::is_same_v<decltype(Linear(2, 3))::value_type, float>);
    SUCCEED();
}
