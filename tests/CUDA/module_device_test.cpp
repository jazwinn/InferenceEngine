#include <gtest/gtest.h>

#include <NeuralNet/Activation.h>
#include <NeuralNet/Embedding.h>
#include <NeuralNet/Linear.h>
#include <NeuralNet/Module.h>
#include <NeuralNet/Sequential.h>
#include <NeuralNet/Transformer/Normal/LayerNorm.h>
#include <NeuralNet/Transformer/RoPE.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using nn::Linear;
using nn::Module;
using nn::Sequential;
using tensor::Device;
using tensor::Tensor;

namespace {

template <typename T>
std::vector<T> Values(const Tensor<T>& t) {
    Tensor<T> cpu = t.To(Device::CPU).Contiguous();
    return std::vector<T>(cpu.Data(), cpu.Data() + cpu.Size());
}

template <typename T>
void ExpectAllOn(const Module<T>& m, Device device) {
    const auto params = m.NamedParameters();
    ASSERT_FALSE(params.empty());
    for (const auto& [name, param] : params) {
        EXPECT_EQ(param->GetDevice(), device) << name;
    }
}

template <typename T>
class TwoLayer : public Module<T> {
public:
    TwoLayer(size_t in, size_t hidden, size_t out)
        : first(std::make_shared<Linear<T>>(in, hidden)),
          second(std::make_shared<Linear<T>>(hidden, out)) {
        this->RegisterSubmodule("first", first);
        this->RegisterSubmodule("second", second);
    }

    std::shared_ptr<Linear<T>> first;
    std::shared_ptr<Linear<T>> second;
};

template <typename T>
class SharedLayer : public Module<T> {
public:
    explicit SharedLayer(std::shared_ptr<Linear<T>> layer) : shared(std::move(layer)) {
        this->RegisterSubmodule("a", shared);
        this->RegisterSubmodule("b", shared);
    }

    std::shared_ptr<Linear<T>> shared;
};

template <typename T>
class TiedParameter : public Module<T> {
public:
    explicit TiedParameter(std::shared_ptr<Tensor<T>> tensor) : tied(std::move(tensor)) {
        this->RegisterParameter("embed", tied);
        this->RegisterParameter("unembed", tied);
    }

    std::shared_ptr<Tensor<T>> tied;
};

template <typename T>
class NoParameters : public Module<T> {};

}

// --- moving to CUDA ---------------------------------------------------------

TEST(ModuleToCUDA, LinearWeightAndBiasMove) {
    Linear<float> linear(4, 3);
    linear.To(Device::CUDA);
    EXPECT_TRUE(linear.weight->IsCUDA());
    EXPECT_TRUE(linear.bias->IsCUDA());
    ExpectAllOn(linear, Device::CUDA);
}

TEST(ModuleToCUDA, LinearWithoutBiasMoves) {
    Linear<float> linear(4, 3, false);
    linear.To(Device::CUDA);
    EXPECT_TRUE(linear.weight->IsCUDA());
    EXPECT_EQ(linear.bias, nullptr);
}

TEST(ModuleToCUDA, ShapesArePreserved) {
    Linear<float> linear(5, 2);
    linear.To(Device::CUDA);
    EXPECT_EQ(linear.weight->Shape(), (std::vector<size_t>{2, 5}));
    EXPECT_EQ(linear.bias->Shape(), (std::vector<size_t>{2}));
}

TEST(ModuleToCUDA, MemberPointerSeesTheMove) {
    Linear<float> linear(4, 3);
    Tensor<float>* before = linear.weight.get();

    linear.To(Device::CUDA);

    EXPECT_EQ(linear.weight.get(), before);
    EXPECT_EQ(linear.NamedParameters()[0].tensor.get(), before);
    EXPECT_TRUE(linear.weight->IsCUDA());
}

TEST(ModuleToCUDA, ValuesAreUploaded) {
    Linear<float> linear(3, 2);
    linear.weight->RandomFill(-1.0f, 1.0f);
    linear.bias->RandomFill(-1.0f, 1.0f);
    const std::vector<float> weight = Values(*linear.weight);
    const std::vector<float> bias = Values(*linear.bias);

    linear.To(Device::CUDA);

    EXPECT_EQ(Values(*linear.weight), weight);
    EXPECT_EQ(Values(*linear.bias), bias);
}

TEST(ModuleToCUDA, NestedSubmodulesMove) {
    TwoLayer<float> model(4, 3, 2);
    model.To(Device::CUDA);
    EXPECT_TRUE(model.first->weight->IsCUDA());
    EXPECT_TRUE(model.second->bias->IsCUDA());
    ExpectAllOn(model, Device::CUDA);
}

TEST(ModuleToCUDA, SequentialMovesEveryLayer) {
    Sequential model(Linear<float>(4, 3), Linear<float>(3, 2));
    model.To(Device::CUDA);
    EXPECT_TRUE(model.Get<0>()->weight->IsCUDA());
    EXPECT_TRUE(model.Get<1>()->weight->IsCUDA());
    ExpectAllOn(model, Device::CUDA);
}

TEST(ModuleToCUDA, ModuleWithoutParametersIsANoOp) {
    NoParameters<float> m;
    EXPECT_NO_THROW(m.To(Device::CUDA));
    EXPECT_TRUE(m.NamedParameters().empty());
}

// --- repeated and shared moves ----------------------------------------------

TEST(ModuleToCUDA, MovingTwiceKeepsStorage) {
    Linear<float> linear(4, 3);
    linear.To(Device::CUDA);
    const Tensor<float> first = *linear.weight;

    linear.To(Device::CUDA);

    EXPECT_TRUE(linear.weight->SharesStorage(first));
}

TEST(ModuleToCUDA, SubmoduleRegisteredTwiceIsMovedOnce) {
    auto layer = std::make_shared<Linear<float>>(4, 3);
    SharedLayer<float> model(layer);

    model.To(Device::CUDA);

    EXPECT_TRUE(layer->weight->IsCUDA());
    const auto params = model.NamedParameters();
    ASSERT_EQ(params.size(), 4u);
    EXPECT_EQ(params[0].tensor, params[2].tensor);
    EXPECT_TRUE(params[0].tensor->SharesStorage(*params[2].tensor));
}

TEST(ModuleToCUDA, TiedParameterStaysTied) {
    auto tensor = std::make_shared<Tensor<float>>(std::vector<size_t>{3, 2}, 1.5f);
    TiedParameter<float> model(tensor);

    model.To(Device::CUDA);

    const auto params = model.NamedParameters();
    ASSERT_EQ(params.size(), 2u);
    EXPECT_TRUE(params[0].tensor->IsCUDA());
    EXPECT_EQ(params[0].tensor, params[1].tensor);
    EXPECT_EQ(Values(*tensor), std::vector<float>(6, 1.5f));
}

TEST(ModuleToCUDA, ViewTakenBeforeMoveStaysOnCPU) {
    Linear<float> linear(4, 3);
    const Tensor<float> old_weight = *linear.weight;

    linear.To(Device::CUDA);

    EXPECT_FALSE(old_weight.IsCUDA());
    EXPECT_FALSE(linear.weight->SharesStorage(old_weight));
}

// --- moving back to CPU -----------------------------------------------------

TEST(ModuleToCPU, RoundTripPreservesValues) {
    TwoLayer<float> model(4, 3, 2);
    for (const auto& [name, param] : model.NamedParameters()) param->RandomFill(-1.0f, 1.0f);
    std::vector<std::vector<float>> before;
    for (const auto& [name, param] : model.NamedParameters()) before.push_back(Values(*param));

    model.To(Device::CUDA);
    model.To(Device::CPU);

    ExpectAllOn(model, Device::CPU);
    const auto params = model.NamedParameters();
    for (size_t i = 0; i < params.size(); ++i) {
        EXPECT_EQ(Values(*params[i].tensor), before[i]) << params[i].name;
    }
}

TEST(ModuleToCPU, ElementAccessWorksAgainAfterReturn) {
    Linear<float> linear(2, 2);
    (*linear.weight)(0, 1) = 7.0f;

    linear.To(Device::CUDA);
    EXPECT_THROW((*linear.weight)(0, 1), std::logic_error);

    linear.To(Device::CPU);
    EXPECT_FLOAT_EQ((*linear.weight)(0, 1), 7.0f);
}

TEST(ModuleToCPU, ForwardWorksAgainAfterReturn) {
    Linear<float> linear(3, 2);
    linear.weight->RandomFill(-1.0f, 1.0f);
    linear.bias->RandomFill(-1.0f, 1.0f);
    Tensor<float> input({4, 3}, 0.5f);
    const std::vector<float> expected = Values(linear.Forward(input));

    linear.To(Device::CUDA);
    linear.To(Device::CPU);

    EXPECT_EQ(Values(linear.Forward(input)), expected);
}

// --- forward after moving ---------------------------------------------------

TEST(ModuleToCUDA, ForwardWithCPUInputThrows) {
    Linear<float> linear(3, 2);
    linear.To(Device::CUDA);
    Tensor<float> input({4, 3}, 1.0f);
    EXPECT_THROW(linear.Forward(input), std::invalid_argument);
}

// --- CPU-only modules throw instead of touching GPU memory ------------------

TEST(CPUOnlyModules, SoftmaxThrowsOnCUDAInput) {
    Tensor<float> gpu({2, 3}, Device::CUDA);
    EXPECT_THROW(nn::Softmax<float>().Forward(gpu), std::logic_error);
}

TEST(CPUOnlyModules, LayerNormThrowsOnCUDAInputOrParameters) {
    nn::transformer::norm::LayerNorm<float> norm(3);
    EXPECT_THROW(norm.Forward(Tensor<float>({2, 3}, Device::CUDA)), std::logic_error);
    norm.To(Device::CUDA);
    EXPECT_THROW(norm.Forward(Tensor<float>({2, 3}, 1.0f)), std::logic_error);
}

TEST(CPUOnlyModules, EmbeddingThrowsOnCUDAIndicesOrTable) {
    nn::Embedding<float> embedding(10, 4);
    EXPECT_THROW(embedding.Forward(Tensor<int64_t>({3}, Device::CUDA)), std::logic_error);
    embedding.To(Device::CUDA);
    EXPECT_THROW(embedding.Forward(Tensor<int64_t>({3})), std::logic_error);
}

TEST(CPUOnlyModules, RoPEThrowsOnCUDAInput) {
    nn::transformer::RotaryEmbedding<float> rope(4, 8);
    EXPECT_THROW(rope.Apply(Tensor<float>({1, 2, 3, 4}, Device::CUDA), 0), std::logic_error);
}

TEST(CPUOnlyModules, MaskedFillThrowsOnCUDATensor) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    EXPECT_THROW(gpu.MaskedFill(Tensor<bool>({2, 2}, true), 0.0f), std::logic_error);
}
