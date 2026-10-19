#include <gtest/gtest.h>

#include <NeuralNet/Transformer/KVCache.h>
#include <Tensor/Tensor.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using nn::transformer::KVCache;
using tensor::CopyInto;
using tensor::Device;
using tensor::Tensor;

namespace {

Tensor<float> Wave(const std::vector<size_t>& shape, double phase) {
    Tensor<float> t(shape);
    for (size_t i = 0; i < t.Size(); ++i) t.Data()[i] = static_cast<float>(std::sin(0.37 * static_cast<double>(i) + phase));
    return t;
}

std::vector<float> Values(const Tensor<float>& t) {
    Tensor<float> cpu = t.To(Device::CPU).Contiguous();
    return std::vector<float>(cpu.Data(), cpu.Data() + cpu.Size());
}

void Append(KVCache<float>& cache, size_t layer, const Tensor<float>& k, const Tensor<float>& v) {
    const size_t start = cache.Room(k.Shape()[2]);
    CopyInto(k, cache.Layer(layer).k.Slice(2, start, k.Shape()[2]));
    CopyInto(v, cache.Layer(layer).v.Slice(2, start, v.Shape()[2]));
}

}

TEST(KVCacheCUDA, DefaultDeviceIsCPU) {
    KVCache<float> cache(2, 1, 2, 8, 4);
    EXPECT_EQ(cache.GetDevice(), Device::CPU);
    EXPECT_FALSE(cache.Layer(0).k.IsCUDA());
}

TEST(KVCacheCUDA, AllocatesEveryLayerOnCUDA) {
    KVCache<float> cache(3, 2, 2, 8, 4, Device::CUDA);
    EXPECT_EQ(cache.GetDevice(), Device::CUDA);
    for (size_t i = 0; i < cache.Layers(); ++i) {
        EXPECT_TRUE(cache.Layer(i).k.IsCUDA()) << "layer " << i;
        EXPECT_TRUE(cache.Layer(i).v.IsCUDA()) << "layer " << i;
        EXPECT_EQ(cache.Layer(i).k.Shape(), (std::vector<size_t>{2, 2, 8, 4}));
    }
}

TEST(KVCacheCUDA, StartsZeroed) {
    KVCache<float> cache(1, 1, 2, 4, 3, Device::CUDA);
    for (float x : Values(cache.Layer(0).k)) EXPECT_EQ(x, 0.0f);
    for (float x : Values(cache.Layer(0).v)) EXPECT_EQ(x, 0.0f);
}

TEST(KVCacheCUDA, PrefillThenDecodeMatchesCPU) {
    KVCache<float> cpu(2, 1, 2, 8, 4);
    KVCache<float> gpu(2, 1, 2, 8, 4, Device::CUDA);

    for (size_t layer = 0; layer < 2; ++layer) {
        auto k = Wave({1, 2, 5, 4}, 0.1 * layer);
        auto v = Wave({1, 2, 5, 4}, 0.7 + layer);
        Append(cpu, layer, k, v);
        Append(gpu, layer, k.To(Device::CUDA), v.To(Device::CUDA));
    }
    cpu.Advance(5);
    gpu.Advance(5);

    for (size_t step = 0; step < 3; ++step) {
        for (size_t layer = 0; layer < 2; ++layer) {
            auto k = Wave({1, 2, 1, 4}, 2.0 + step + layer);
            auto v = Wave({1, 2, 1, 4}, 3.0 + step + layer);
            Append(cpu, layer, k, v);
            Append(gpu, layer, k.To(Device::CUDA), v.To(Device::CUDA));
        }
        cpu.Advance(1);
        gpu.Advance(1);
    }

    ASSERT_EQ(gpu.Length(), 8u);
    for (size_t layer = 0; layer < 2; ++layer) {
        EXPECT_EQ(Values(gpu.Layer(layer).k), Values(cpu.Layer(layer).k)) << "layer " << layer;
        EXPECT_EQ(Values(gpu.Layer(layer).v), Values(cpu.Layer(layer).v)) << "layer " << layer;
    }
}

TEST(KVCacheCUDA, ReadBackViewMatchesWhatWasWritten) {
    KVCache<float> cache(1, 1, 2, 6, 4, Device::CUDA);
    auto first = Wave({1, 2, 3, 4}, 0.2);
    auto second = Wave({1, 2, 2, 4}, 1.4);
    Append(cache, 0, first.To(Device::CUDA), first.To(Device::CUDA));
    cache.Advance(3);
    Append(cache, 0, second.To(Device::CUDA), second.To(Device::CUDA));
    cache.Advance(2);

    Tensor<float> view = cache.Layer(0).k.Slice(2, 0, cache.Length());
    EXPECT_TRUE(view.IsCUDA());
    EXPECT_EQ(view.Shape(), (std::vector<size_t>{1, 2, 5, 4}));

    const auto got = Values(view);
    const auto a = Values(first);
    const auto b = Values(second);
    for (size_t h = 0; h < 2; ++h) {
        for (size_t s = 0; s < 5; ++s) {
            for (size_t d = 0; d < 4; ++d) {
                const float expected = s < 3 ? a[(h * 3 + s) * 4 + d] : b[(h * 2 + (s - 3)) * 4 + d];
                EXPECT_EQ(got[(h * 5 + s) * 4 + d], expected) << "h " << h << " s " << s << " d " << d;
            }
        }
    }
}

TEST(KVCacheCUDA, AppendLeavesTailUntouched) {
    KVCache<float> cache(1, 1, 1, 6, 2, Device::CUDA);
    Append(cache, 0, Wave({1, 1, 2, 2}, 0.5).To(Device::CUDA), Wave({1, 1, 2, 2}, 0.9).To(Device::CUDA));
    cache.Advance(2);
    const auto k = Values(cache.Layer(0).k);
    for (size_t i = 4; i < k.size(); ++i) EXPECT_EQ(k[i], 0.0f) << "at index " << i;
}

TEST(KVCacheCUDA, ResetMakesPositionZeroWritableAgain) {
    KVCache<float> cache(1, 1, 1, 4, 2, Device::CUDA);
    Append(cache, 0, Wave({1, 1, 2, 2}, 0.1).To(Device::CUDA), Wave({1, 1, 2, 2}, 0.2).To(Device::CUDA));
    cache.Advance(2);
    cache.Reset();
    auto fresh = Wave({1, 1, 1, 2}, 4.0);
    Append(cache, 0, fresh.To(Device::CUDA), fresh.To(Device::CUDA));
    const auto k = Values(cache.Layer(0).k);
    const auto f = Values(fresh);
    EXPECT_EQ(k[0], f[0]);
    EXPECT_EQ(k[1], f[1]);
}

TEST(KVCacheCUDA, CPUKeysIntoCUDACacheThrow) {
    KVCache<float> cache(1, 1, 1, 4, 2, Device::CUDA);
    EXPECT_THROW(CopyInto(Wave({1, 1, 1, 2}, 0.0), cache.Layer(0).k.Slice(2, 0, 1)), std::invalid_argument);
}
