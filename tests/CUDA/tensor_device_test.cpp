#include <gtest/gtest.h>

#include <CUDA/Kernel.cuh>
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

TEST(TensorCUDATo, DownloadOfTransposedViewMatchesCPU) {
    Tensor<float> cpu = Iota<float>({2, 3});
    Tensor<float> back = cpu.To(Device::CUDA).Transpose().To(Device::CPU);
    EXPECT_FALSE(back.IsCUDA());
    EXPECT_TRUE(back.IsContiguous());
    EXPECT_EQ(back.Shape(), (std::vector<size_t>{3, 2}));
    EXPECT_EQ(Values(back), Values(cpu.Transpose()));
}

TEST(TensorCUDATo, DownloadOfPermutedViewMatchesCPU) {
    Tensor<float> cpu = Iota<float>({2, 3, 4});
    Tensor<float> back = cpu.To(Device::CUDA).Permute(2, 0, 1).To(Device::CPU);
    EXPECT_EQ(Values(back), Values(cpu.Permute(2, 0, 1)));
}

TEST(TensorCUDATo, DownloadOfInnerSliceMatchesCPU) {
    Tensor<float> cpu = Iota<float>({3, 4});
    Tensor<float> back = cpu.To(Device::CUDA).Slice(1, 1, 2).To(Device::CPU);
    EXPECT_EQ(Values(back), (std::vector<float>{2, 3, 6, 7, 10, 11}));
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

TEST(TensorCUDAViews, ReshapeOfNonContiguousCopiesAndMatchesCPU) {
    Tensor<float> cpu = Iota<float>({2, 3});
    Tensor<float> gpu = cpu.To(Device::CUDA);
    Tensor<float> reshaped = gpu.Transpose().Reshape(6);
    EXPECT_TRUE(reshaped.IsCUDA());
    EXPECT_FALSE(reshaped.SharesStorage(gpu));
    EXPECT_EQ(Values(reshaped), Values(cpu.Transpose().Reshape(6)));
}

TEST(TensorCUDAViews, ContiguousOfTransposeMatchesCPU) {
    Tensor<float> cpu = Iota<float>({2, 3});
    Tensor<float> c = cpu.To(Device::CUDA).Transpose().Contiguous();
    EXPECT_TRUE(c.IsCUDA());
    EXPECT_TRUE(c.IsContiguous());
    EXPECT_EQ(c.Offset(), 0u);
    EXPECT_EQ(Values(c), (std::vector<float>{1, 4, 2, 5, 3, 6}));
}

TEST(TensorCUDAViews, SplitHeadsPatternMatchesCPU) {
    Tensor<float> cpu = Iota<float>({2, 5, 12});
    auto split = [](const Tensor<float>& x) { return x.Reshape(2, 5, 3, 4).Transpose(1, 2).Contiguous(); };
    Tensor<float> gpu = split(cpu.To(Device::CUDA));
    EXPECT_TRUE(gpu.IsCUDA());
    EXPECT_EQ(gpu.Shape(), (std::vector<size_t>{2, 3, 5, 4}));
    EXPECT_EQ(Values(gpu), Values(split(cpu)));
}

TEST(TensorCUDAViews, MergeHeadsPatternMatchesCPU) {
    Tensor<float> cpu = Iota<float>({2, 3, 5, 4});
    auto merge = [](const Tensor<float>& x) { return x.Transpose(1, 2).Reshape(2, 5, 12); };
    EXPECT_EQ(Values(merge(cpu.To(Device::CUDA))), Values(merge(cpu)));
}

// --- Clone ------------------------------------------------------------------

TEST(TensorCUDAClone, ContiguousCloneIsIndependentCopy) {
    Tensor<float> gpu = Iota<float>({2, 3}).To(Device::CUDA);
    Tensor<float> clone = gpu.Clone();
    EXPECT_TRUE(clone.IsCUDA());
    EXPECT_FALSE(clone.SharesStorage(gpu));
    EXPECT_EQ(Values(clone), Values(gpu));
}

TEST(TensorCUDAClone, CloneOfSliceStartsAtOffsetZero) {
    Tensor<float> gpu = Iota<float>({4, 3}).To(Device::CUDA);
    Tensor<float> clone = gpu.Slice(0, 1, 2).Clone();
    EXPECT_EQ(clone.Offset(), 0u);
    EXPECT_EQ(Values(clone), (std::vector<float>{4, 5, 6, 7, 8, 9}));
}

TEST(TensorCUDAClone, CloneOfPermuteMatchesCPU) {
    Tensor<double> cpu = Iota<double>({3, 2, 4});
    Tensor<double> clone = cpu.To(Device::CUDA).Permute(1, 2, 0).Clone();
    EXPECT_TRUE(clone.IsContiguous());
    EXPECT_EQ(Values(clone), Values(cpu.Permute(1, 2, 0)));
}

TEST(TensorCUDAClone, CloneOfBoolMask) {
    Tensor<bool> cpu({2, 3}, true);
    Tensor<bool> clone = cpu.To(Device::CUDA).Transpose().Clone();
    EXPECT_TRUE(clone.IsCUDA());
    EXPECT_EQ(Values(clone), std::vector<bool>(6, true));
}

TEST(TensorCUDAClone, EmptyCloneStaysEmpty) {
    Tensor<float> empty;
    EXPECT_EQ(empty.Clone().Size(), 0u);
}

// --- CopyInto ---------------------------------------------------------------

TEST(TensorCUDACopyInto, ContiguousCopyMatches) {
    Tensor<float> src = Iota<float>({2, 3}).To(Device::CUDA);
    Tensor<float> dst({2, 3}, Device::CUDA);
    CopyInto(src, dst);
    EXPECT_EQ(Values(dst), Values(src));
}

TEST(TensorCUDACopyInto, WritesIntoKVCacheSliceAndKeepsOldRows) {
    Tensor<float> cache_cpu({2, 4, 2}, -1.0f);
    Tensor<float> src_cpu = Iota<float>({2, 2, 2});
    Tensor<float> cache = cache_cpu.To(Device::CUDA);

    CopyInto(src_cpu.To(Device::CUDA), cache.Slice(1, 2, 2));
    CopyInto(src_cpu, cache_cpu.Slice(1, 2, 2));

    EXPECT_EQ(Values(cache), (std::vector<float>{-1, -1, -1, -1, 1, 2, 3, 4, -1, -1, -1, -1, 5, 6, 7, 8}));
    EXPECT_EQ(Values(cache), Values(cache_cpu));
}

TEST(TensorCUDACopyInto, NonContiguousSourceIntoContiguousDestination) {
    Tensor<float> cpu = Iota<float>({2, 3});
    Tensor<float> dst({3, 2}, Device::CUDA);
    CopyInto(cpu.To(Device::CUDA).Transpose(), dst);
    EXPECT_EQ(Values(dst), Values(cpu.Transpose()));
}

TEST(TensorCUDACopyInto, IntoTransposedDestination) {
    Tensor<float> src = Iota<float>({3, 2}).To(Device::CUDA);
    Tensor<float> dst({2, 3}, Device::CUDA);
    CopyInto(src, dst.Transpose());
    EXPECT_EQ(Values(dst), (std::vector<float>{1, 3, 5, 2, 4, 6}));
}

TEST(TensorCUDACopyInto, ShapeMismatchThrows) {
    Tensor<float> src({2, 3}, Device::CUDA);
    Tensor<float> dst({3, 2}, Device::CUDA);
    EXPECT_THROW(CopyInto(src, dst), std::invalid_argument);
}

TEST(TensorCUDACopyInto, MixedDevicesThrow) {
    Tensor<float> gpu({2, 2}, Device::CUDA);
    Tensor<float> cpu({2, 2}, 1.0f);
    EXPECT_THROW(CopyInto(cpu, gpu), std::invalid_argument);
    EXPECT_THROW(CopyInto(gpu, cpu), std::invalid_argument);
}

// --- strided copy launcher --------------------------------------------------

TEST(StridedCopyLauncher, RankAboveMaxThrows) {
    const std::vector<size_t> shape(9, 1), strides(9, 1);
    EXPECT_THROW(cuda::LaunchStridedCopyDevice<float>(nullptr, nullptr, shape, strides, strides),
                 std::invalid_argument);
}

TEST(StridedCopyLauncher, StrideRankMismatchThrows) {
    const std::vector<size_t> shape{2, 3}, good{3, 1}, bad{1};
    EXPECT_THROW(cuda::LaunchStridedCopyDevice<float>(nullptr, nullptr, shape, bad, good), std::invalid_argument);
    EXPECT_THROW(cuda::LaunchStridedCopyDevice<float>(nullptr, nullptr, shape, good, bad), std::invalid_argument);
}

TEST(StridedCopyLauncher, ZeroElementsIsANoOp) {
    const std::vector<size_t> shape{0, 3}, strides{3, 1};
    EXPECT_NO_THROW(cuda::LaunchStridedCopyDevice<float>(nullptr, nullptr, shape, strides, strides));
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
    EXPECT_THROW(gpu.RandomFill(), std::logic_error);
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

TEST(TensorCUDAMatMul, TransposedRightOperandMatchesCPU) {
    Tensor<float> a = Iota<float>({2, 3});
    Tensor<float> b = Iota<float>({4, 3});
    Tensor<float> gb = b.To(Device::CUDA).Transpose();
    ASSERT_FALSE(gb.IsContiguous());
    Tensor<float> result = MatMul(a.To(Device::CUDA), gb);
    EXPECT_EQ(result.Shape(), (std::vector<size_t>{2, 4}));
    EXPECT_EQ(Values(result), Values(MatMul(a, b.Transpose())));
}

TEST(TensorCUDAMatMul, TransposedLeftOperandMatchesCPU) {
    Tensor<float> a = Iota<float>({3, 2});
    Tensor<float> b = Iota<float>({3, 4});
    EXPECT_EQ(Values(MatMul(a.To(Device::CUDA).Transpose(), b.To(Device::CUDA))),
              Values(MatMul(a.Transpose(), b)));
}

TEST(TensorCUDAMatMul, BothOperandsTransposedMatchesCPU) {
    Tensor<float> a = Iota<float>({3, 2});
    Tensor<float> b = Iota<float>({4, 3});
    EXPECT_EQ(Values(MatMul(a.To(Device::CUDA).Transpose(), b.To(Device::CUDA).Transpose())),
              Values(MatMul(a.Transpose(), b.Transpose())));
}

TEST(TensorCUDAMatMul, ColumnSliceOperandMatchesCPU) {
    Tensor<float> a = Iota<float>({2, 5});
    Tensor<float> b = Iota<float>({3, 4});
    Tensor<float> ga = a.To(Device::CUDA).Slice(1, 1, 3);
    ASSERT_FALSE(ga.IsContiguous());
    EXPECT_EQ(Values(MatMul(ga, b.To(Device::CUDA))), Values(MatMul(a.Slice(1, 1, 3), b)));
}

TEST(TensorCUDAMatMul, NonContiguousOperandIsNotModified) {
    Tensor<float> a = Iota<float>({2, 3});
    Tensor<float> b = Iota<float>({4, 3});
    Tensor<float> gb = b.To(Device::CUDA);
    MatMul(a.To(Device::CUDA), gb.Transpose());
    EXPECT_EQ(Values(gb), Values(b));
}

TEST(TensorCUDAMatMul, BatchedOperandMatchesCPU) {
    Tensor<float> a = Iota<float>({2, 2, 3});
    Tensor<float> b = Iota<float>({3, 2});
    Tensor<float> result = MatMul(a.To(Device::CUDA), b.To(Device::CUDA));
    EXPECT_EQ(result.Shape(), (std::vector<size_t>{2, 2, 2}));
    EXPECT_EQ(Values(result), Values(MatMul(a, b)));
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
