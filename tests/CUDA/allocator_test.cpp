#include <gtest/gtest.h>

#include <CUDA/Memory.h>
#include <Tensor/Tensor.h>

#include <stdexcept>
#include <thread>
#include <vector>

using tensor::Device;
using tensor::Tensor;

namespace {

class CachingAllocator : public ::testing::Test {
protected:
    void SetUp() override { cuda::EmptyCache(); }
    void TearDown() override { cuda::EmptyCache(); }
};

}

TEST_F(CachingAllocator, FreedBlockIsReusedForSameSize) {
    void* a = cuda::DeviceAlloc(4096);
    cuda::DeviceFree(a);
    void* b = cuda::DeviceAlloc(4096);
    EXPECT_EQ(a, b);
    cuda::DeviceFree(b);
}

TEST_F(CachingAllocator, SizesInTheSameBucketShareBlocks) {
    void* a = cuda::DeviceAlloc(4000);
    cuda::DeviceFree(a);
    void* b = cuda::DeviceAlloc(4096);
    EXPECT_EQ(a, b);
    cuda::DeviceFree(b);
}

TEST_F(CachingAllocator, SizesInDifferentBucketsDoNotShare) {
    void* a = cuda::DeviceAlloc(4096);
    cuda::DeviceFree(a);
    void* b = cuda::DeviceAlloc(4097);
    EXPECT_NE(a, b);
    cuda::DeviceFree(b);
}

TEST_F(CachingAllocator, SmallRequestsRoundTo512Bytes) {
    const size_t before = cuda::AllocatedBytes();
    void* a = cuda::DeviceAlloc(4);
    EXPECT_EQ(cuda::AllocatedBytes() - before, 512u);
    cuda::DeviceFree(a);
}

TEST_F(CachingAllocator, LargeRequestsRoundTo2MiB) {
    const size_t before = cuda::AllocatedBytes();
    void* a = cuda::DeviceAlloc((1u << 20) + 1);
    EXPECT_EQ(cuda::AllocatedBytes() - before, 2u << 20);
    cuda::DeviceFree(a);
}

TEST_F(CachingAllocator, CachedBytesTrackFreeAndReuse) {
    EXPECT_EQ(cuda::CachedBytes(), 0u);
    void* a = cuda::DeviceAlloc(8192);
    EXPECT_EQ(cuda::CachedBytes(), 0u);
    cuda::DeviceFree(a);
    EXPECT_EQ(cuda::CachedBytes(), 8192u);
    void* b = cuda::DeviceAlloc(8192);
    EXPECT_EQ(cuda::CachedBytes(), 0u);
    cuda::DeviceFree(b);
}

TEST_F(CachingAllocator, CachedBlocksStillCountAsAllocated) {
    const size_t before = cuda::AllocatedBytes();
    void* a = cuda::DeviceAlloc(8192);
    cuda::DeviceFree(a);
    EXPECT_EQ(cuda::AllocatedBytes() - before, 8192u);
}

TEST_F(CachingAllocator, EmptyCacheReleasesOnlyCachedBlocks) {
    void* live = cuda::DeviceAlloc(4096);
    void* idle = cuda::DeviceAlloc(8192);
    cuda::DeviceFree(idle);
    const size_t allocated = cuda::AllocatedBytes();

    cuda::EmptyCache();

    EXPECT_EQ(cuda::CachedBytes(), 0u);
    EXPECT_EQ(cuda::AllocatedBytes(), allocated - 8192u);
    cuda::DeviceFree(live);
}

TEST_F(CachingAllocator, AllocationWorksAfterEmptyCache) {
    void* a = cuda::DeviceAlloc(4096);
    cuda::DeviceFree(a);
    cuda::EmptyCache();
    void* b = cuda::DeviceAlloc(4096);
    EXPECT_NE(b, nullptr);
    cuda::DeviceFree(b);
}

TEST_F(CachingAllocator, BlockLargerThanCapIsReleasedOnFree) {
    const size_t before = cuda::AllocatedBytes();
    void* big = cuda::DeviceAlloc(300ull << 20);
    cuda::DeviceFree(big);
    EXPECT_EQ(cuda::AllocatedBytes(), before);
    EXPECT_EQ(cuda::CachedBytes(), 0u);
}

TEST_F(CachingAllocator, ZeroBytesReturnsNull) {
    EXPECT_EQ(cuda::DeviceAlloc(0), nullptr);
}

TEST_F(CachingAllocator, FreeNullIsANoOp) {
    EXPECT_NO_THROW(cuda::DeviceFree(nullptr));
}

TEST_F(CachingAllocator, DoubleFreeThrows) {
    void* a = cuda::DeviceAlloc(4096);
    cuda::DeviceFree(a);
    EXPECT_THROW(cuda::DeviceFree(a), std::invalid_argument);
}

TEST_F(CachingAllocator, StrayPointerThrows) {
    int host_value = 0;
    EXPECT_THROW(cuda::DeviceFree(&host_value), std::invalid_argument);
}

TEST_F(CachingAllocator, LiveBlocksAreNeverHandedOutTwice) {
    void* a = cuda::DeviceAlloc(4096);
    void* b = cuda::DeviceAlloc(4096);
    EXPECT_NE(a, b);
    cuda::DeviceFree(a);
    cuda::DeviceFree(b);
}

TEST_F(CachingAllocator, TensorStorageIsReused) {
    const float* first = nullptr;
    {
        Tensor<float> t({64, 64}, Device::CUDA);
        first = t.Data();
    }
    Tensor<float> again({64, 64}, Device::CUDA);
    EXPECT_EQ(again.Data(), first);
}

TEST_F(CachingAllocator, ReusedTensorStorageIsStillZeroed) {
    {
        Tensor<float> t = Tensor<float>({32, 32}, 7.0f).To(Device::CUDA);
    }
    Tensor<float> again({32, 32}, Device::CUDA);
    Tensor<float> back = again.To(Device::CPU);
    for (size_t i = 0; i < back.Size(); ++i) EXPECT_EQ(back.Data()[i], 0.0f) << "at index " << i;
}

TEST_F(CachingAllocator, ConcurrentAllocateAndFreeBalance) {
    const size_t before = cuda::AllocatedBytes();
    auto worker = [] {
        for (int i = 0; i < 500; ++i) {
            void* p = cuda::DeviceAlloc(4096 + static_cast<size_t>(i % 4) * 512);
            cuda::DeviceFree(p);
        }
    };
    std::thread t1(worker), t2(worker), t3(worker);
    t1.join();
    t2.join();
    t3.join();
    EXPECT_EQ(cuda::AllocatedBytes() - before, cuda::CachedBytes());
}
