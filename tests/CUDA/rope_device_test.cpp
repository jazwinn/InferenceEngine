#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>
#include <NeuralNet/Transformer/GroupQueryAttention.h>
#include <NeuralNet/Transformer/RoPE.h>
#include <Tensor/Tensor.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using nn::transformer::GroupQueryAttention;
using nn::transformer::RotaryEmbedding;
using nn::transformer::config::PositionEncoding;
using nn::transformer::config::RopeScaling;
using nn::transformer::config::RopeScalingType;
using nn::transformer::config::TransformerConfig;
using tensor::Device;
using tensor::Tensor;

namespace {

template <typename T>
Tensor<T> Wave(const std::vector<size_t>& shape, double phase) {
    Tensor<T> t(shape);
    T* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i) {
        data[i] = static_cast<T>(std::sin(0.37 * static_cast<double>(i) + phase));
    }
    return t;
}

template <typename T>
std::vector<T> Values(const Tensor<T>& t) {
    Tensor<T> cpu = t.To(Device::CPU).Contiguous();
    return std::vector<T>(cpu.Data(), cpu.Data() + cpu.Size());
}

template <typename T>
void ExpectNear(const Tensor<T>& actual, const Tensor<T>& expected, T tol) {
    ASSERT_EQ(actual.Shape(), expected.Shape());
    const auto a = Values(actual);
    const auto e = Values(expected);
    for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], e[i], tol) << "at index " << i;
}

template <typename T>
void ExpectGPUMatchesCPU(const std::vector<size_t>& shape, size_t max_seq, size_t start_pos, T tol,
                         RopeScaling scaling = {}) {
    const size_t head_dim = shape[3];
    RotaryEmbedding<T> cpu_rope(head_dim, max_seq, 10000.0, scaling);
    RotaryEmbedding<T> gpu_rope(head_dim, max_seq, 10000.0, scaling);
    gpu_rope.To(Device::CUDA);

    auto x = Wave<T>(shape, 0.4);
    Tensor<T> expected = cpu_rope.Apply(x, start_pos);
    Tensor<T> actual = gpu_rope.Apply(x.To(Device::CUDA), start_pos);

    EXPECT_TRUE(actual.IsCUDA());
    ExpectNear(actual, expected, tol);
}

TransformerConfig RotaryConfig() {
    TransformerConfig cfg;
    cfg.d_model = 16;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.position = PositionEncoding::Rotary;
    cfg.max_seq_len = 8;
    cfg.rope_theta = 10000.0f;
    return cfg;
}

}

// --- RotaryEmbedding::Apply on CUDA ---------------------------------------------

TEST(RoPECUDA, PromptMatchesCPU) {
    ExpectGPUMatchesCPU<float>({1, 4, 10, 8}, 16, 0, 1e-6f);
}

TEST(RoPECUDA, SingleTokenDecodeMatchesCPU) {
    ExpectGPUMatchesCPU<float>({1, 4, 1, 8}, 16, 7, 1e-6f);
}

TEST(RoPECUDA, PrefillWithStartOffsetMatchesCPU) {
    ExpectGPUMatchesCPU<float>({1, 2, 5, 8}, 16, 3, 1e-6f);
}

TEST(RoPECUDA, LastPositionsOfTableMatchCPU) {
    ExpectGPUMatchesCPU<float>({1, 2, 4, 8}, 16, 12, 1e-6f);
}

TEST(RoPECUDA, BatchAndManyHeadsMatchCPU) {
    ExpectGPUMatchesCPU<float>({3, 6, 7, 16}, 32, 0, 1e-6f);
}

TEST(RoPECUDA, ModelSizedHeadMatchesCPU) {
    ExpectGPUMatchesCPU<float>({1, 32, 20, 64}, 128, 5, 1e-5f);
}

TEST(RoPECUDA, Llama3ScalingMatchesCPU) {
    RopeScaling scaling;
    scaling.type = RopeScalingType::Llama3;
    scaling.factor = 32.0;
    scaling.low_freq_factor = 1.0;
    scaling.high_freq_factor = 4.0;
    scaling.original_max_position = 64;
    ExpectGPUMatchesCPU<float>({1, 4, 9, 64}, 256, 100, 1e-5f, scaling);
}

TEST(RoPECUDA, DoubleMatchesCPU) {
    ExpectGPUMatchesCPU<double>({2, 3, 6, 8}, 16, 2, 1e-12);
}

TEST(RoPECUDA, PositionZeroIsIdentity) {
    RotaryEmbedding<float> rope(8, 4);
    rope.To(Device::CUDA);
    auto x = Wave<float>({1, 2, 1, 8}, 0.9);
    ExpectNear(rope.Apply(x.To(Device::CUDA), 0), x, 1e-7f);
}

TEST(RoPECUDA, PreservesPairNorms) {
    RotaryEmbedding<float> rope(8, 16);
    rope.To(Device::CUDA);
    auto x = Wave<float>({1, 1, 3, 8}, 0.2);
    const auto in = Values(x);
    const auto out = Values(rope.Apply(x.To(Device::CUDA), 5));
    for (size_t v = 0; v < 3; ++v) {
        for (size_t i = 0; i < 4; ++i) {
            const size_t a = v * 8 + i;
            const size_t b = a + 4;
            EXPECT_NEAR(out[a] * out[a] + out[b] * out[b], in[a] * in[a] + in[b] * in[b], 1e-5f);
        }
    }
}

TEST(RoPECUDA, SplitHeadsTransposedInputMatchesCPU) {
    RotaryEmbedding<float> cpu_rope(8, 16);
    RotaryEmbedding<float> gpu_rope(8, 16);
    gpu_rope.To(Device::CUDA);

    auto bshd = Wave<float>({2, 5, 3, 8}, 0.6);
    Tensor<float> gx = bshd.To(Device::CUDA).Transpose(1, 2);
    ASSERT_FALSE(gx.IsContiguous());
    ExpectNear(gpu_rope.Apply(gx, 2), cpu_rope.Apply(bshd.Transpose(1, 2), 2), 1e-6f);
}

TEST(RoPECUDA, DoesNotMutateInput) {
    RotaryEmbedding<float> rope(8, 16);
    rope.To(Device::CUDA);
    auto x = Wave<float>({1, 2, 4, 8}, 0.3);
    Tensor<float> gx = x.To(Device::CUDA);
    rope.Apply(gx, 1);
    EXPECT_EQ(Values(gx), Values(x));
}

TEST(RoPECUDA, PositionsPastMaxSeqThrow) {
    RotaryEmbedding<float> rope(8, 4);
    rope.To(Device::CUDA);
    EXPECT_THROW(rope.Apply(Tensor<float>({1, 1, 3, 8}, Device::CUDA), 2), std::out_of_range);
}

TEST(RoPECUDA, HeadDimMismatchThrows) {
    RotaryEmbedding<float> rope(8, 4);
    rope.To(Device::CUDA);
    EXPECT_THROW(rope.Apply(Tensor<float>({1, 1, 2, 4}, Device::CUDA), 0), std::invalid_argument);
}

TEST(RoPECUDA, CUDAInputWithCPUTablesThrows) {
    RotaryEmbedding<float> rope(4, 8);
    EXPECT_THROW(rope.Apply(Tensor<float>({1, 2, 3, 4}, Device::CUDA), 0), std::invalid_argument);
}

TEST(RoPECUDA, CPUInputWithCUDATablesThrows) {
    RotaryEmbedding<float> rope(4, 8);
    rope.To(Device::CUDA);
    EXPECT_THROW(rope.Apply(Tensor<float>({1, 2, 3, 4}, 1.0f), 0), std::invalid_argument);
}

// --- buffers move with Module::To ------------------------------------------------

TEST(RoPEBuffers, AreRegisteredAndNotParameters) {
    RotaryEmbedding<float> rope(8, 16);
    const auto buffers = rope.NamedBuffers();
    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_EQ(buffers[0].name, "cos");
    EXPECT_EQ(buffers[1].name, "sin");
    EXPECT_TRUE(rope.NamedParameters().empty());
}

TEST(RoPEBuffers, ToMovesBothTables) {
    RotaryEmbedding<float> rope(8, 16);
    rope.To(Device::CUDA);
    EXPECT_TRUE(rope.cos->IsCUDA());
    EXPECT_TRUE(rope.sin->IsCUDA());
    for (const auto& [name, buffer] : rope.NamedBuffers()) EXPECT_TRUE(buffer->IsCUDA()) << name;
}

TEST(RoPEBuffers, RoundTripPreservesTables) {
    RotaryEmbedding<float> rope(8, 16);
    const auto cos_before = Values(*rope.cos);
    const auto sin_before = Values(*rope.sin);
    rope.To(Device::CUDA);
    rope.To(Device::CPU);
    EXPECT_FALSE(rope.cos->IsCUDA());
    EXPECT_EQ(Values(*rope.cos), cos_before);
    EXPECT_EQ(Values(*rope.sin), sin_before);
}

TEST(RoPEBuffers, AttentionToMovesSharedRope) {
    auto rope = std::make_shared<RotaryEmbedding<float>>(4, 8);
    GroupQueryAttention<float> attn(RotaryConfig(), rope);
    attn.To(Device::CUDA);
    EXPECT_TRUE(rope->cos->IsCUDA());
    EXPECT_TRUE(rope->sin->IsCUDA());
    EXPECT_TRUE(attn.query->weight->IsCUDA());
}

TEST(RoPEBuffers, AttentionListsRopeBuffersButNoRopeParameters) {
    auto rope = std::make_shared<RotaryEmbedding<float>>(4, 8);
    GroupQueryAttention<float> attn(RotaryConfig(), rope);
    const auto buffers = attn.NamedBuffers();
    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_EQ(buffers[0].name, "rope.cos");
    EXPECT_EQ(buffers[1].name, "rope.sin");
    for (const auto& [name, param] : attn.NamedParameters()) EXPECT_EQ(name.find("rope"), std::string::npos) << name;
}

TEST(RoPEBuffers, TwoAttentionsSharingOneRopeBothWork) {
    auto rope = std::make_shared<RotaryEmbedding<float>>(4, 8);
    GroupQueryAttention<float> a(RotaryConfig(), rope);
    GroupQueryAttention<float> b(RotaryConfig(), rope);
    a.To(Device::CUDA);
    b.To(Device::CUDA);
    EXPECT_TRUE(rope->cos->IsCUDA());
    EXPECT_TRUE(a.RoPE->cos == b.RoPE->cos);
}

// --- launcher --------------------------------------------------------------------

TEST(RoPELauncher, ZeroPairsIsANoOp) {
    EXPECT_NO_THROW(cuda::LaunchRoPEDevice<float>(nullptr, nullptr, nullptr, nullptr, 0, 3, 4, 0));
    EXPECT_NO_THROW(cuda::LaunchRoPEDevice<float>(nullptr, nullptr, nullptr, nullptr, 2, 0, 4, 0));
}

TEST(RoPELauncher, WorksInPlace) {
    RotaryEmbedding<float> rope(8, 16);
    auto x = Wave<float>({1, 2, 3, 8}, 0.8);
    Tensor<float> expected = rope.Apply(x, 4);
    rope.To(Device::CUDA);
    Tensor<float> gx = x.To(Device::CUDA);
    cuda::LaunchRoPEDevice(gx.Data(), gx.Data(), rope.cos->Data(), rope.sin->Data(), 2, 3, 4, 4);
    ExpectNear(gx, expected, 1e-6f);
}
