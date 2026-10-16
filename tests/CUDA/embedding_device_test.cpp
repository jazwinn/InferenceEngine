#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>
#include <NeuralNet/Embedding.h>
#include <Tensor/Tensor.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using nn::Embedding;
using tensor::Device;
using tensor::Tensor;

namespace {

template <typename T>
Tensor<T> Wave(const std::vector<size_t>& shape, double phase) {
    Tensor<T> t(shape);
    for (size_t i = 0; i < t.Size(); ++i) t.Data()[i] = static_cast<T>(std::sin(0.37 * static_cast<double>(i) + phase));
    return t;
}

template <typename Index>
Tensor<Index> Ids(const std::vector<size_t>& shape, const std::vector<Index>& values) {
    Tensor<Index> t(shape);
    for (size_t i = 0; i < t.Size(); ++i) t.Data()[i] = values[i % values.size()];
    return t;
}

template <typename T>
std::vector<T> Values(const Tensor<T>& t) {
    Tensor<T> cpu = t.To(Device::CPU).Contiguous();
    return std::vector<T>(cpu.Data(), cpu.Data() + cpu.Size());
}

template <typename T, typename Index>
void ExpectGPUMatchesCPU(size_t vocab, size_t dim, const Tensor<Index>& ids) {
    Embedding<T> cpu_emb(vocab, dim);
    *cpu_emb.weight = Wave<T>({vocab, dim}, 0.3);
    Embedding<T> gpu_emb(vocab, dim);
    *gpu_emb.weight = *cpu_emb.weight;
    gpu_emb.To(Device::CUDA);

    Tensor<T> expected = cpu_emb.Forward(ids);
    Tensor<T> actual = gpu_emb.Forward(ids.To(Device::CUDA));

    EXPECT_TRUE(actual.IsCUDA());
    ASSERT_EQ(actual.Shape(), expected.Shape());
    EXPECT_EQ(Values(actual), Values(expected));
}

}

// --- Embedding::Forward on CUDA --------------------------------------------------

TEST(EmbeddingCUDA, PromptOfTokenIdsMatchesCPU) {
    ExpectGPUMatchesCPU<float, int64_t>(10, 4, Ids<int64_t>({5}, {3, 0, 9, 1, 7}));
}

TEST(EmbeddingCUDA, BatchedIdsKeepLeadingShape) {
    auto ids = Ids<int64_t>({2, 3}, {4, 2, 8, 1, 1, 0});
    ExpectGPUMatchesCPU<float, int64_t>(10, 6, ids);
}

TEST(EmbeddingCUDA, SingleTokenDecode) {
    ExpectGPUMatchesCPU<float, int64_t>(50, 16, Ids<int64_t>({1, 1}, {42}));
}

TEST(EmbeddingCUDA, IntIdsMatchCPU) {
    ExpectGPUMatchesCPU<float, int>(10, 4, Ids<int>({4}, {2, 9, 0, 5}));
}

TEST(EmbeddingCUDA, SizeTIdsMatchCPU) {
    ExpectGPUMatchesCPU<float, size_t>(10, 4, Ids<size_t>({4}, {2, 9, 0, 5}));
}

TEST(EmbeddingCUDA, DoubleTableMatchesCPU) {
    ExpectGPUMatchesCPU<double, int64_t>(12, 5, Ids<int64_t>({3}, {11, 0, 6}));
}

TEST(EmbeddingCUDA, LargeVocabularyAndDimMatchCPU) {
    ExpectGPUMatchesCPU<float, int64_t>(5000, 256, Ids<int64_t>({1, 37}, {4999, 0, 1234, 77, 2500, 3}));
}

TEST(EmbeddingCUDA, RepeatedIdsGetTheSameRow) {
    Embedding<float> emb(10, 4);
    *emb.weight = Wave<float>({10, 4}, 0.1);
    emb.To(Device::CUDA);
    const auto out = Values(emb.Forward(Ids<int64_t>({3}, {6, 6, 6}).To(Device::CUDA)));
    for (size_t j = 0; j < 4; ++j) {
        EXPECT_EQ(out[j], out[4 + j]);
        EXPECT_EQ(out[j], out[8 + j]);
    }
}

TEST(EmbeddingCUDA, CopiesExactTableRows) {
    Embedding<float> emb(5, 3);
    *emb.weight = Wave<float>({5, 3}, 0.9);
    const auto table = Values(*emb.weight);
    emb.To(Device::CUDA);
    const auto out = Values(emb.Forward(Ids<int64_t>({2}, {4, 1}).To(Device::CUDA)));
    for (size_t j = 0; j < 3; ++j) {
        EXPECT_EQ(out[j], table[4 * 3 + j]);
        EXPECT_EQ(out[3 + j], table[1 * 3 + j]);
    }
}

TEST(EmbeddingCUDA, OutOfRangeIdsGiveZeroRows) {
    Embedding<float> emb(4, 3);
    *emb.weight = Wave<float>({4, 3}, 0.5);
    emb.To(Device::CUDA);
    const auto out = Values(emb.Forward(Ids<int64_t>({3}, {1, 4, -1}).To(Device::CUDA)));
    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NE(out[j], 0.0f);
        EXPECT_EQ(out[3 + j], 0.0f);
        EXPECT_EQ(out[6 + j], 0.0f);
    }
}

TEST(EmbeddingCUDA, NonContiguousIdsMatchCPU) {
    auto ids = Ids<int64_t>({3, 2}, {0, 1, 2, 3, 4, 5});
    Tensor<int64_t> gpu_ids = ids.To(Device::CUDA).Transpose();
    ASSERT_FALSE(gpu_ids.IsContiguous());

    Embedding<float> cpu_emb(6, 4);
    *cpu_emb.weight = Wave<float>({6, 4}, 0.2);
    Embedding<float> gpu_emb(6, 4);
    *gpu_emb.weight = *cpu_emb.weight;
    gpu_emb.To(Device::CUDA);

    EXPECT_EQ(Values(gpu_emb.Forward(gpu_ids)), Values(cpu_emb.Forward(ids.Transpose())));
}

TEST(EmbeddingCUDA, PositionAndTokenEmbeddingsSumLikeGPT2) {
    Embedding<float> cpu_wte(20, 8), cpu_wpe(16, 8);
    *cpu_wte.weight = Wave<float>({20, 8}, 0.1);
    *cpu_wpe.weight = Wave<float>({16, 8}, 1.7);
    Embedding<float> gpu_wte(20, 8), gpu_wpe(16, 8);
    *gpu_wte.weight = *cpu_wte.weight;
    *gpu_wpe.weight = *cpu_wpe.weight;
    gpu_wte.To(Device::CUDA);
    gpu_wpe.To(Device::CUDA);

    auto tokens = Ids<int64_t>({1, 5}, {19, 3, 3, 0, 11});
    auto positions = Ids<int64_t>({1, 5}, {0, 1, 2, 3, 4});
    Tensor<float> expected = cpu_wte.Forward(tokens) + cpu_wpe.Forward(positions);
    Tensor<float> actual = gpu_wte.Forward(tokens.To(Device::CUDA)) + gpu_wpe.Forward(positions.To(Device::CUDA));
    const auto a = Values(actual);
    const auto e = Values(expected);
    ASSERT_EQ(a.size(), e.size());
    for (size_t i = 0; i < a.size(); ++i) EXPECT_FLOAT_EQ(a[i], e[i]) << "at index " << i;
}

TEST(EmbeddingCUDA, FromPretrainedThenToMatchesCPU) {
    auto table = Wave<float>({7, 3}, 0.6);
    auto cpu_emb = Embedding<float>::FromPretrained(table);
    auto gpu_emb = Embedding<float>::FromPretrained(table);
    gpu_emb.To(Device::CUDA);
    auto ids = Ids<int64_t>({4}, {6, 2, 0, 5});
    EXPECT_EQ(Values(gpu_emb.Forward(ids.To(Device::CUDA))), Values(cpu_emb.Forward(ids)));
}

TEST(EmbeddingCUDA, WeightMovesWithTo) {
    Embedding<float> emb(10, 4);
    emb.To(Device::CUDA);
    EXPECT_TRUE(emb.weight->IsCUDA());
    emb.To(Device::CPU);
    EXPECT_FALSE(emb.weight->IsCUDA());
}

TEST(EmbeddingCUDA, CUDAIdsWithCPUTableThrow) {
    Embedding<float> emb(10, 4);
    EXPECT_THROW(emb.Forward(Tensor<int64_t>({3}, Device::CUDA)), std::invalid_argument);
}

TEST(EmbeddingCUDA, CPUIdsWithCUDATableThrow) {
    Embedding<float> emb(10, 4);
    emb.To(Device::CUDA);
    EXPECT_THROW(emb.Forward(Tensor<int64_t>({3})), std::invalid_argument);
}

TEST(EmbeddingCUDA, CPUPathStillChecksRange) {
    Embedding<float> emb(4, 3);
    EXPECT_THROW(emb.Forward(Ids<int64_t>({2}, {1, 4})), std::out_of_range);
}

// --- launcher --------------------------------------------------------------------

TEST(EmbeddingLauncher, ZeroIdsIsANoOp) {
    EXPECT_NO_THROW((cuda::LaunchEmbeddingLookUpDevice<float, int64_t>(nullptr, nullptr, nullptr, 0, 10, 4)));
}

TEST(EmbeddingLauncher, ZeroDimIsANoOp) {
    EXPECT_NO_THROW((cuda::LaunchEmbeddingLookUpDevice<float, int64_t>(nullptr, nullptr, nullptr, 3, 10, 0)));
}
