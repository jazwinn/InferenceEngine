#include <gtest/gtest.h>

#include <Models/CausalModel.h>
#include <NeuralNet/Transformer/AttentionOps.h>
#include <Tensor/Tensor.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using models::CausalModel;
using nn::transformer::KVCache;
using nn::transformer::SplitHeads;
using nn::transformer::config::Activation;
using nn::transformer::config::MLPType;
using nn::transformer::config::NormType;
using nn::transformer::config::PositionEncoding;
using nn::transformer::config::TransformerConfig;
using tensor::CopyInto;
using tensor::Device;
using tensor::Tensor;

namespace {

constexpr size_t kModel = 16;
constexpr size_t kLayers = 2;
constexpr size_t kMaxSeq = 16;
constexpr size_t kVocab = 40;

TransformerConfig LlamaStyle() {
    TransformerConfig c;
    c.vocab_size = kVocab;
    c.d_model = kModel;
    c.num_layers = kLayers;
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

TransformerConfig UntiedLlama() {
    TransformerConfig c = LlamaStyle();
    c.tie_word_embeddings = false;
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

struct Trace {
    Tensor<float> embeddings;
    std::vector<Tensor<float>> queries;
    std::vector<Tensor<float>> keys;
    std::vector<Tensor<float>> values;
    std::vector<Tensor<float>> block_outputs;
    Tensor<float> logits;
};

Trace RunWithoutAttention(CausalModel<float>& model, const Tensor<int64_t>& ids, size_t start_pos,
                          KVCache<float>* cache) {
    const Device device = model.wte->weight->GetDevice();
    const size_t seq = ids.Shape()[1];
    Trace trace;

    Tensor<float> x = model.wte->Forward(ids.To(device));
    if (model.wpe) {
        Tensor<int64_t> positions({seq});
        for (size_t i = 0; i < seq; ++i) positions(i) = static_cast<int64_t>(start_pos + i);
        x = x + model.wpe->Forward(positions.To(device));
    }
    trace.embeddings = x;

    for (size_t layer = 0; layer < model.transformer->blocks.size(); ++layer) {
        auto& block = *model.transformer->blocks[layer];
        auto& attn = *block.attention;

        const Tensor<float> normed = block.attenNorm->Forward(x);
        Tensor<float> Q = SplitHeads(attn.query->Forward(normed), attn.numHeadsQ, attn.headDim);
        Tensor<float> K = SplitHeads(attn.key->Forward(normed), attn.numHeadsKV, attn.headDim);
        Tensor<float> V = SplitHeads(attn.value->Forward(normed), attn.numHeadsKV, attn.headDim);
        if (attn.RoPE) {
            Q = attn.RoPE->Apply(Q, start_pos);
            K = attn.RoPE->Apply(K, start_pos);
        }
        if (cache) {
            CopyInto(K, cache->Layer(layer).k.Slice(2, start_pos, seq));
            CopyInto(V, cache->Layer(layer).v.Slice(2, start_pos, seq));
        }
        trace.queries.push_back(Q);
        trace.keys.push_back(K);
        trace.values.push_back(V);

        x = x + attn.output_proj->Forward(normed);
        x = x + block.mlp->Forward(block.mlpNorm->Forward(x));
        trace.block_outputs.push_back(x);
    }
    if (cache) cache->Advance(seq);

    const Tensor<float> h = model.transformer->finalNorm ? model.transformer->finalNorm->Forward(x) : x;
    trace.logits = model.lm_head ? model.lm_head->Forward(h) : tensor::Linear(h, *model.wte->weight);
    return trace;
}

void ExpectTracesMatch(const Trace& gpu, const Trace& cpu, float tol) {
    EXPECT_TRUE(gpu.logits.IsCUDA());
    ExpectNear(gpu.embeddings, cpu.embeddings, tol, "embeddings");
    for (size_t l = 0; l < cpu.queries.size(); ++l) {
        ExpectNear(gpu.queries[l], cpu.queries[l], tol, "queries");
        ExpectNear(gpu.keys[l], cpu.keys[l], tol, "keys");
        ExpectNear(gpu.values[l], cpu.values[l], tol, "values");
        ExpectNear(gpu.block_outputs[l], cpu.block_outputs[l], tol, "block output");
    }
    ExpectNear(gpu.logits, cpu.logits, tol, "logits");
}

struct ModelPair {
    explicit ModelPair(const TransformerConfig& cfg) : cpu(cfg), gpu(cfg) {
        Seed(cpu);
        Seed(gpu);
        gpu.To(Device::CUDA);
    }
    CausalModel<float> cpu;
    CausalModel<float> gpu;
};

void ExpectPromptThenDecodeMatches(const TransformerConfig& cfg) {
    ModelPair models(cfg);
    KVCache<float> cpu_cache = models.cpu.MakeCache(1, kMaxSeq);
    KVCache<float> gpu_cache = models.gpu.MakeCache(1, kMaxSeq);
    ASSERT_EQ(gpu_cache.GetDevice(), Device::CUDA);

    const auto prompt = Tokens({3, 17, 0, 39, 8, 22});
    ExpectTracesMatch(RunWithoutAttention(models.gpu, prompt, 0, &gpu_cache),
                      RunWithoutAttention(models.cpu, prompt, 0, &cpu_cache), 1e-4f);

    size_t pos = 6;
    for (int64_t token : {11, 5, 30}) {
        const auto next = Tokens({token});
        ExpectTracesMatch(RunWithoutAttention(models.gpu, next, pos, &gpu_cache),
                          RunWithoutAttention(models.cpu, next, pos, &cpu_cache), 1e-4f);
        ++pos;
    }

    for (size_t l = 0; l < kLayers; ++l) {
        ExpectNear(gpu_cache.Layer(l).k, cpu_cache.Layer(l).k, 1e-4f, "cached keys");
        ExpectNear(gpu_cache.Layer(l).v, cpu_cache.Layer(l).v, 1e-4f, "cached values");
    }
}

}

// --- everything except attention, GPU vs CPU --------------------------------------

TEST(Phase3CUDA, LlamaStylePromptThenDecode) {
    ExpectPromptThenDecodeMatches(LlamaStyle());
}

TEST(Phase3CUDA, GPT2StylePromptThenDecode) {
    ExpectPromptThenDecodeMatches(GPT2Style());
}

TEST(Phase3CUDA, UntiedHeadPromptThenDecode) {
    ExpectPromptThenDecodeMatches(UntiedLlama());
}

TEST(Phase3CUDA, BatchedPromptMatchesCPU) {
    ModelPair models(LlamaStyle());
    Tensor<int64_t> ids({2, 4});
    const int64_t values[] = {1, 2, 3, 4, 39, 0, 20, 7};
    for (size_t i = 0; i < 8; ++i) ids.Data()[i] = values[i];
    ExpectTracesMatch(RunWithoutAttention(models.gpu, ids, 0, nullptr),
                      RunWithoutAttention(models.cpu, ids, 0, nullptr), 1e-4f);
}

TEST(Phase3CUDA, TiedHeadLogitsHaveVocabularyWidth) {
    ModelPair models(LlamaStyle());
    const Trace trace = RunWithoutAttention(models.gpu, Tokens({1, 2, 3}), 0, nullptr);
    EXPECT_EQ(trace.logits.Shape(), (std::vector<size_t>{1, 3, kVocab}));
}

// --- model-level device plumbing ---------------------------------------------------

TEST(Phase3CUDA, ModelToMovesEveryParameterAndBuffer) {
    ModelPair models(LlamaStyle());
    for (const auto& p : models.gpu.NamedParameters()) EXPECT_TRUE(p.tensor->IsCUDA()) << p.name;
    const auto buffers = models.gpu.NamedBuffers();
    ASSERT_FALSE(buffers.empty());
    for (const auto& b : buffers) EXPECT_TRUE(b.tensor->IsCUDA()) << b.name;
}

TEST(Phase3CUDA, MakeCacheFollowsModelDevice) {
    ModelPair models(LlamaStyle());
    KVCache<float> cpu_cache = models.cpu.MakeCache(2, 8);
    KVCache<float> gpu_cache = models.gpu.MakeCache(2, 8);
    EXPECT_EQ(cpu_cache.GetDevice(), Device::CPU);
    EXPECT_EQ(gpu_cache.GetDevice(), Device::CUDA);
    EXPECT_EQ(gpu_cache.Layers(), kLayers);
    EXPECT_EQ(gpu_cache.Layer(0).k.Shape(), (std::vector<size_t>{2, 2, 8, 4}));
}

TEST(Phase3CUDA, CPUIdsIntoCUDAModelThrow) {
    ModelPair models(LlamaStyle());
    EXPECT_THROW(models.gpu.wte->Forward(Tokens({1, 2})), std::invalid_argument);
}

TEST(Phase3CUDA, CPUCacheWithCUDAModelThrows) {
    ModelPair models(LlamaStyle());
    KVCache<float> cpu_cache(kLayers, 1, 2, kMaxSeq, 4);
    EXPECT_THROW(models.gpu.Forward(Tokens({1, 2}).To(Device::CUDA), &cpu_cache), std::invalid_argument);
}
