#include <gtest/gtest.h>

#include <Models/CausalModel.h>
#include <NeuralNet/Transformer/AttentionOps.h>
#include <Tensor/Tensor.h>

#include <cmath>
#include <cstdint>
#include <vector>

using models::CausalModel;
using nn::transformer::KVCache;
using nn::transformer::SDPA;
using nn::transformer::config::Activation;
using nn::transformer::config::MLPType;
using nn::transformer::config::NormType;
using nn::transformer::config::PositionEncoding;
using nn::transformer::config::TransformerConfig;
using tensor::Device;
using tensor::Tensor;

namespace {

constexpr size_t kMaxSeq = 16;
constexpr size_t kVocab = 40;

Tensor<float> Wave(const std::vector<size_t>& shape, double phase = 0.0) {
    Tensor<float> t(shape);
    for (size_t i = 0; i < t.Size(); ++i) {
        t.Data()[i] = static_cast<float>(std::sin(phase + 0.37 * static_cast<double>(i)));
    }
    return t;
}

std::vector<float> Values(const Tensor<float>& t) {
    Tensor<float> cpu = t.To(Device::CPU).Contiguous();
    return std::vector<float>(cpu.Data(), cpu.Data() + cpu.Size());
}

void ExpectNear(const Tensor<float>& actual, const Tensor<float>& expected, float tol, const char* what) {
    ASSERT_EQ(actual.Shape(), expected.Shape()) << what;
    const auto a = Values(actual);
    const auto e = Values(expected);
    for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], e[i], tol) << what << " at index " << i;
}

TransformerConfig LlamaStyle() {
    TransformerConfig c;
    c.vocab_size = kVocab;
    c.d_model = 16;
    c.num_layers = 2;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.ffn_dim = 24;
    c.max_seq_len = kMaxSeq;
    c.norm_type = NormType::RMSNorm;
    c.mlp_type = MLPType::Gated;
    c.activation = Activation::SiLU;
    c.position = PositionEncoding::Rotary;
    c.attention_bias = false;
    c.output_bias = false;
    return c;
}

TransformerConfig GPT2Style() {
    TransformerConfig c = LlamaStyle();
    c.num_kv_heads = 0;
    c.norm_type = NormType::LayerNorm;
    c.mlp_type = MLPType::Standard;
    c.activation = Activation::GELUTanh;
    c.position = PositionEncoding::Learned;
    c.attention_bias = true;
    c.output_bias = true;
    return c;
}

void Seed(CausalModel<float>& model) {
    double phase = 0.1;
    for (const auto& p : model.NamedParameters()) {
        Tensor<float>& t = *p.tensor;
        for (size_t i = 0; i < t.Size(); ++i) {
            t.Data()[i] = static_cast<float>(0.3 * std::sin(phase + 0.41 * static_cast<double>(i)));
        }
        phase += 0.7;
    }
}

Tensor<int64_t> Tokens(const std::vector<int64_t>& ids) {
    Tensor<int64_t> t({1, ids.size()});
    for (size_t i = 0; i < ids.size(); ++i) t(0, i) = ids[i];
    return t;
}

void ExpectSDPAMatches(const Tensor<float>& q, const Tensor<float>& k, const Tensor<float>& v,
                       bool causal, const Tensor<bool>& mask = Tensor<bool>()) {
    const Tensor<bool> gmask = mask.Size() > 0 ? mask.To(Device::CUDA) : mask;
    const Tensor<float> gpu = SDPA(q.To(Device::CUDA), k.To(Device::CUDA), v.To(Device::CUDA), gmask, causal);
    EXPECT_TRUE(gpu.IsCUDA());
    ExpectNear(gpu, SDPA(q, k, v, mask, causal), 1e-5f, "SDPA");
}

void ExpectForwardMatches(const TransformerConfig& cfg) {
    CausalModel<float> cpu(cfg);
    CausalModel<float> gpu(cfg);
    Seed(cpu);
    Seed(gpu);
    gpu.To(Device::CUDA);

    KVCache<float> cpu_cache = cpu.MakeCache(1, kMaxSeq);
    KVCache<float> gpu_cache = gpu.MakeCache(1, kMaxSeq);

    const auto prompt = Tokens({3, 17, 0, 39, 8, 22});
    const Tensor<float> gpu_logits = gpu.Forward(prompt.To(Device::CUDA), &gpu_cache);
    EXPECT_TRUE(gpu_logits.IsCUDA());
    ExpectNear(gpu_logits, cpu.Forward(prompt, &cpu_cache), 1e-4f, "prompt logits");

    for (int64_t token : {11, 5, 30}) {
        const auto next = Tokens({token});
        ExpectNear(gpu.Forward(next.To(Device::CUDA), &gpu_cache), cpu.Forward(next, &cpu_cache),
                   1e-4f, "decode logits");
    }
}

}

// --- SDPA, GPU vs CPU ----------------------------------------------------------

TEST(AttentionCUDA, SDPAWithoutMaskMatchesCPU) {
    ExpectSDPAMatches(Wave({2, 4, 5, 8}), Wave({2, 4, 5, 8}, 1.0), Wave({2, 4, 5, 8}, 2.0), false);
}

TEST(AttentionCUDA, SDPACausalSquareMatchesCPU) {
    ExpectSDPAMatches(Wave({1, 4, 6, 8}), Wave({1, 4, 6, 8}, 1.0), Wave({1, 4, 6, 8}, 2.0), true);
}

TEST(AttentionCUDA, SDPACausalWithCachedPrefixMatchesCPU) {
    ExpectSDPAMatches(Wave({1, 4, 3, 8}), Wave({1, 4, 9, 8}, 1.0), Wave({1, 4, 9, 8}, 2.0), true);
}

TEST(AttentionCUDA, SDPASingleQueryDecodeMatchesCPU) {
    ExpectSDPAMatches(Wave({1, 4, 1, 8}), Wave({1, 4, 9, 8}, 1.0), Wave({1, 4, 9, 8}, 2.0), true);
}

TEST(AttentionCUDA, SDPAGroupedQueryBroadcastMatchesCPU) {
    ExpectSDPAMatches(Wave({1, 2, 3, 5, 8}), Wave({1, 2, 1, 5, 8}, 1.0), Wave({1, 2, 1, 5, 8}, 2.0), true);
}

TEST(AttentionCUDA, SDPASlicedCacheMatchesCPU) {
    const Tensor<float> cache_k = Wave({1, 2, kMaxSeq, 8}, 1.0);
    const Tensor<float> cache_v = Wave({1, 2, kMaxSeq, 8}, 2.0);
    const Tensor<float> q = Wave({1, 2, 3, 2, 8});

    const Tensor<float> gpu = SDPA(q.To(Device::CUDA),
                                   cache_k.To(Device::CUDA).Slice(2, 0, 7).Unsqueeze(2),
                                   cache_v.To(Device::CUDA).Slice(2, 0, 7).Unsqueeze(2),
                                   Tensor<bool>(), true);
    const Tensor<float> cpu = SDPA(q, cache_k.Slice(2, 0, 7).Unsqueeze(2),
                                   cache_v.Slice(2, 0, 7).Unsqueeze(2), Tensor<bool>(), true);
    ExpectNear(gpu, cpu, 1e-5f, "SDPA over sliced cache");
}

TEST(AttentionCUDA, SDPAExplicitMaskMatchesCPU) {
    Tensor<bool> mask({5, 5}, false);
    mask(0, 3) = true;
    mask(2, 1) = true;
    mask(4, 0) = true;
    ExpectSDPAMatches(Wave({1, 2, 5, 8}), Wave({1, 2, 5, 8}, 1.0), Wave({1, 2, 5, 8}, 2.0), false, mask);
}

// --- full model, GPU vs CPU ---------------------------------------------------

TEST(AttentionCUDA, LlamaStyleForwardPromptThenDecodeMatchesCPU) {
    ExpectForwardMatches(LlamaStyle());
}

TEST(AttentionCUDA, GPT2StyleForwardPromptThenDecodeMatchesCPU) {
    ExpectForwardMatches(GPT2Style());
}

TEST(AttentionCUDA, BatchedForwardWithoutCacheMatchesCPU) {
    CausalModel<float> cpu(LlamaStyle());
    CausalModel<float> gpu(LlamaStyle());
    Seed(cpu);
    Seed(gpu);
    gpu.To(Device::CUDA);

    Tensor<int64_t> ids({2, 4});
    const int64_t values[] = {1, 2, 3, 4, 39, 0, 20, 7};
    for (size_t i = 0; i < 8; ++i) ids.Data()[i] = values[i];
    ExpectNear(gpu.Forward(ids.To(Device::CUDA)), cpu.Forward(ids), 1e-4f, "batched logits");
}
