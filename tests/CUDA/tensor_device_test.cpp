#include <gtest/gtest.h>

#include <Tensor/Tensor.h>

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using tensor::Device;
using tensor::Tensor;

namespace {

template <typename T>
Tensor<T> Iota(const std::vector<size_t>& shape) {
    Tensor<T> t(shape);
    T* data = t.Data();
    for (size_t i = 0; i < t.Size(); ++i) data[i] = T(i + 1);
    return t;
}

template <typename T>
std::vector<T> Values(const Tensor<T>& t) {
    Tensor<T> cpu = t.To(Device::CPU).Contiguous();
    return std::vector<T>(cpu.Data(), cpu.Data() + cpu.Size());
}

}

// --- construction -----------------------------------------------------------

TEST(TensorCUDAConstruction, DeviceConstructorAllocatesOnCUDA) {
    Tensor<float> t({2, 3}, Device::CUDA);
    EXPECT_TRUE(t.IsCUDA());
    EXPECT_EQ(t.Shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(t.Size(), 6u);
    EXPECT_NE(t.Data(), nullptr);
}

TEST(TensorCUDAConstruction, DeviceConstructorZeroFills) {
    Tensor<float> t({3, 4}, Device::CUDA);
    EXPECT_EQ(Values(t), std::vector<float>(12, 0.0f));
}

TEST(TensorCUDAConstruction, InitialValueIsUploaded) {
    Tensor<float> t({2, 3}, 4.5f, Device::CUDA);
    EXPECT_TRUE(t.IsCUDA());
    EXPECT_EQ(Values(t), std::vector<float>(6, 4.5f));
}

TEST(TensorCUDAConstruction, ZeroInitialValueOnCUDA) {
    Tensor<double> t({5}, 0.0, Device::CUDA);
    EXPECT_TRUE(t.IsCUDA());
    EXPECT_EQ(Values(t), std::vector<double>(5, 0.0));
}

// --- To ---------------------------------------------------------------------

TEST(TensorCUDATo, RoundTripPreservesValues) {
    Tensor<float> cpu = Iota<float>({2, 3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    EXPECT_TRUE(gpu.IsCUDA());
    EXPECT_EQ(gpu.Shape(), cpu.Shape());

    Tensor<float> back = gpu.To(Device::CPU);
    EXPECT_FALSE(back.IsCUDA());
    EXPECT_FALSE(back.SharesStorage(cpu));
    EXPECT_EQ(Values(back), Values(cpu));
}

TEST(TensorCUDATo, RoundTripDouble) {
    Tensor<double> cpu = Iota<double>({7});
    EXPECT_EQ(Values(cpu.To(Device::CUDA)), Values(cpu));
}

TEST(TensorCUDATo, ToSameDeviceSharesStorage) {
    Tensor<float> gpu = Iota<float>({4}).To(Device::CUDA);
    EXPECT_TRUE(gpu.To(Device::CUDA).SharesStorage(gpu));
}

TEST(TensorCUDATo, UploadCompactsNonContiguousView) {
    Tensor<float> cpu = Iota<float>({2, 3});
    Tensor<float> transposed = cpu.Transpose();
    Tensor<float> gpu = transposed.To(Device::CUDA);
    EXPECT_EQ(gpu.Shape(), (std::vector<size_t>{3, 2}));
    EXPECT_TRUE(gpu.IsContiguous());
    EXPECT_EQ(Values(gpu), Values(transposed));
}

TEST(TensorCUDATo, UploadOfSliceCopiesOnlyTheSlice) {
    Tensor<float> cpu = Iota<float>({4, 3});
    Tensor<float> gpu = cpu.Slice(0, 1, 2).To(Device::CUDA);
    EXPECT_EQ(gpu.Size(), 6u);
    EXPECT_EQ(Values(gpu), (std::vector<float>{4, 5, 6, 7, 8, 9}));
}

TEST(TensorCUDATo, UploadIsIndependentOfSource) {
    Tensor<float> cpu = Iota<float>({3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    cpu(0) = 100.0f;
    EXPECT_EQ(Values(gpu), (std::vector<float>{1, 2, 3}));
}

TEST(TensorCUDATo, DownloadOfNonContiguousViewThrows) {
    Tensor<float> gpu = Iota<float>({2, 3}).To(Device::CUDA);
    EXPECT_THROW(gpu.Transpose().To(Device::CPU), std::logic_error);
}

// --- views ------------------------------------------------------------------

TEST(TensorCUDAViews, EveryViewKeepsDeviceAndStorage) {
    Tensor<float> gpu = Iota<float>({2, 3, 4}).To(Device::CUDA);
    const std::vector<Tensor<float>> views = {
        gpu.Reshape(6, 4),
        gpu.View(4, 6),
        gpu.Row(1),
        gpu.Permute(2, 0, 1),
        gpu.Transpose(0, 2),
        gpu.Transpose(),
        gpu.Slice(1, 1, 2),
        gpu.Unsqueeze(0),
        gpu.Unsqueeze(0).Squeeze(0),
        gpu.Contiguous(),
    };
    for (size_t i = 0; i < views.size(); ++i) {
        EXPECT_TRUE(views[i].IsCUDA()) << "view " << i;
        EXPECT_TRUE(views[i].SharesStorage(gpu)) << "view " << i;
    }
}

TEST(TensorCUDAViews, ReshapeMatchesCPU) {
    Tensor<float> cpu = Iota<float>({2, 3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    EXPECT_EQ(Values(gpu.Reshape(4, 6)), Values(cpu.Reshape(4, 6)));
    EXPECT_EQ(Values(gpu.View(24)), Values(cpu.View(24)));
}

TEST(TensorCUDAViews, RowMatchesCPU) {
    Tensor<float> cpu = Iota<float>({3, 4});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    EXPECT_EQ(Values(gpu.Row(2)), Values(cpu.Row(2)));
}

TEST(TensorCUDAViews, OuterSliceMatchesCPU) {
    Tensor<float> cpu = Iota<float>({5, 2, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    EXPECT_EQ(Values(gpu.Slice(0, 2, 3)), Values(cpu.Slice(0, 2, 3)));
}

TEST(TensorCUDAViews, UnsqueezeSqueezeMatchesCPU) {
    Tensor<float> cpu = Iota<float>({2, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> u = gpu.Unsqueeze(1);
    EXPECT_EQ(u.Shape(), (std::vector<size_t>{2, 1, 3}));
    EXPECT_EQ(Values(u), Values(cpu));
    EXPECT_EQ(Values(u.Squeeze(1)), Values(cpu));
}

TEST(TensorCUDAViews, DoubleTransposeIsContiguousAndDownloads) {
    Tensor<float> cpu = Iota<float>({2, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    EXPECT_EQ(Values(gpu.Transpose().Transpose()), Values(cpu));
}

TEST(TensorCUDAViews, ViewsKeepStorageAliveAfterOriginalDies) {
    Tensor<float> view;
    {
        Tensor<float> gpu = Iota<float>({4, 2}).To(Device::CUDA);
        view = gpu.Slice(0, 2, 2);
        EXPECT_EQ(gpu.UseCount(), 2);
    }
    EXPECT_EQ(view.UseCount(), 1);
    EXPECT_EQ(Values(view), (std::vector<float>{5, 6, 7, 8}));
}

TEST(TensorCUDAViews, ReshapeOfNonContiguousThrows) {
    Tensor<float> gpu = Iota<float>({2, 3}).To(Device::CUDA);
    EXPECT_THROW(gpu.Transpose().Reshape(6), std::logic_error);
}

// --- CPU-only operations ----------------------------------------------------

TEST(TensorCUDACPUOnly, ElementAccessThrows) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    const Tensor<float>& cgpu = gpu;
    const std::vector<size_t> idx = {0, 0};
    EXPECT_THROW(gpu(0, 0), std::logic_error);
    EXPECT_THROW(cgpu(0, 0), std::logic_error);
    EXPECT_THROW(gpu(std::span<const size_t>(idx)), std::logic_error);
    EXPECT_THROW(cgpu(std::span<const size_t>(idx)), std::logic_error);
}

TEST(TensorCUDACPUOnly, ForEachBasedOperationsThrow) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    EXPECT_THROW(gpu.Fill(1.0f), std::logic_error);
    EXPECT_THROW(gpu.RandomFill(), std::logic_error);
    EXPECT_THROW(gpu += 1.0f, std::logic_error);
    EXPECT_THROW(gpu *= 2.0f, std::logic_error);
    EXPECT_THROW(gpu.Mean(), std::logic_error);
}

TEST(TensorCUDACPUOnly, CopyingOperationsThrow) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    EXPECT_THROW(gpu.Clone(), std::logic_error);
    EXPECT_THROW(gpu.Sqrt(), std::logic_error);
    EXPECT_THROW(gpu.Mean(0), std::logic_error);
    EXPECT_THROW(gpu * 2.0f, std::logic_error);
}

TEST(TensorCUDACPUOnly, BinaryOperationsThrowForEitherSide) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    Tensor<float> cpu({2, 2}, 1.0f);
    EXPECT_THROW(gpu + cpu, std::logic_error);
    EXPECT_THROW(cpu + gpu, std::logic_error);
    EXPECT_THROW(cpu += gpu, std::logic_error);
    EXPECT_THROW(gpu += cpu, std::logic_error);
}

TEST(TensorCUDAMatMul, MixedDevicesThrow) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    Tensor<float> cpu({2, 2}, 1.0f);
    EXPECT_THROW(MatMul(gpu, cpu), std::invalid_argument);
    EXPECT_THROW(MatMul(cpu, gpu), std::invalid_argument);
}

TEST(TensorCUDAMatMul, TwoDimensionalMatchesCPU) {
    Tensor<float> a = Iota<float>({3, 4});
    Tensor<float> b = Iota<float>({4, 5});
    Tensor<float> result = MatMul(a.To(Device::CUDA), b.To(Device::CUDA));
    EXPECT_TRUE(result.IsCUDA());
    EXPECT_EQ(result.Shape(), (std::vector<size_t>{3, 5}));
    EXPECT_EQ(Values(result), Values(MatMul(a, b)));
}

TEST(TensorCUDAMatMul, DoubleMatchesCPU) {
    Tensor<double> a = Iota<double>({2, 3});
    Tensor<double> b = Iota<double>({3, 2});
    EXPECT_EQ(Values(MatMul(a.To(Device::CUDA), b.To(Device::CUDA))), Values(MatMul(a, b)));
}

TEST(TensorCUDAMatMul, InnerDimensionMismatchThrows) {
    Tensor<float> a({2, 3}, Device::CUDA);
    Tensor<float> b({4, 2}, Device::CUDA);
    EXPECT_THROW(MatMul(a, b), std::invalid_argument);
}

TEST(TensorCUDAMatMul, NonContiguousOperandThrows) {
    Tensor<float> a({2, 3}, Device::CUDA);
    Tensor<float> b({2, 3}, Device::CUDA);
    EXPECT_THROW(MatMul(a, b.Transpose()), std::logic_error);
}

TEST(TensorCUDAMatMul, BatchedOperandThrows) {
    Tensor<float> a({2, 2, 3}, Device::CUDA);
    Tensor<float> b({3, 2}, Device::CUDA);
    EXPECT_THROW(MatMul(a, b), std::logic_error);
}

TEST(TensorCUDACPUOnly, CopyIntoThrows) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    Tensor<float> cpu({2, 2}, 1.0f);
    EXPECT_THROW(CopyInto(gpu, cpu), std::logic_error);
    EXPECT_THROW(CopyInto(cpu, gpu), std::logic_error);
}

TEST(TensorCUDACPUOnly, MaskedFillThrowsForCUDAMaskOrTensor) {
    Tensor<float> cpu({2, 2}, 1.0f);
    Tensor<bool> cpu_mask({2, 2}, true);
    Tensor<float> gpu({2, 2}, Device::CUDA);
    Tensor<bool> gpu_mask({2, 2}, Device::CUDA);
    EXPECT_THROW(gpu.MaskedFill(cpu_mask, 0.0f), std::logic_error);
    EXPECT_THROW(cpu.MaskedFill(gpu_mask, 0.0f), std::logic_error);
}

TEST(TensorCUDACPUOnly, RequireCPUMessageNamesTheOperation) {
    Tensor<float> gpu({1}, Device::CUDA);
    try {
        gpu.RequireCPU("Softmax");
        FAIL() << "expected logic_error";
    } catch (const std::logic_error& e) {
        EXPECT_NE(std::string(e.what()).find("Softmax"), std::string::npos);
    }
}
