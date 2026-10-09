#include <gtest/gtest.h>

#include <CUDA/Memory.h>

#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using cuda::DeviceBuffer;

static_assert(!std::is_copy_constructible_v<DeviceBuffer<float>>);
static_assert(!std::is_copy_assignable_v<DeviceBuffer<float>>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer<float>>);
static_assert(std::is_nothrow_move_assignable_v<DeviceBuffer<float>>);

namespace {

std::vector<float> Iota(size_t n, float start = 1.0f) {
    std::vector<float> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
}

std::vector<float> Download(const DeviceBuffer<float>& buf) {
    std::vector<float> out(buf.Size());
    buf.CopyToHost(out.data(), out.size());
    return out;
}

}

TEST(DeviceBufferConstruction, DefaultIsEmpty) {
    DeviceBuffer<float> buf;
    EXPECT_EQ(buf.Size(), 0u);
    EXPECT_EQ(buf.Bytes(), 0u);
    EXPECT_EQ(buf.Data(), nullptr);
}

TEST(DeviceBufferConstruction, CountAllocates) {
    DeviceBuffer<float> buf(10);
    EXPECT_EQ(buf.Size(), 10u);
    EXPECT_EQ(buf.Bytes(), 10 * sizeof(float));
    EXPECT_NE(buf.Data(), nullptr);
}

TEST(DeviceBufferConstruction, ZeroCountDoesNotAllocate) {
    DeviceBuffer<float> buf(0);
    EXPECT_EQ(buf.Size(), 0u);
    EXPECT_EQ(buf.Data(), nullptr);
}

TEST(DeviceBufferConstruction, FromHostCopiesData) {
    auto host = Iota(8);
    DeviceBuffer<float> buf(host.data(), host.size());
    EXPECT_EQ(buf.Size(), 8u);
    EXPECT_EQ(Download(buf), host);
}

TEST(DeviceBufferConstruction, BytesUsesElementSize) {
    DeviceBuffer<double> buf(5);
    EXPECT_EQ(buf.Bytes(), 5 * sizeof(double));
}

TEST(DeviceBufferCopy, RoundTripPreservesData) {
    auto host = Iota(256, -100.5f);
    DeviceBuffer<float> buf(host.size());
    buf.CopyFromHost(host.data(), host.size());
    EXPECT_EQ(Download(buf), host);
}

TEST(DeviceBufferCopy, PartialCopyFromHostLeavesTailUntouched) {
    auto initial = Iota(6, 10.0f);
    DeviceBuffer<float> buf(initial.data(), initial.size());

    std::vector<float> head = {-1.0f, -2.0f};
    buf.CopyFromHost(head.data(), head.size());

    std::vector<float> expected = {-1.0f, -2.0f, 12.0f, 13.0f, 14.0f, 15.0f};
    EXPECT_EQ(Download(buf), expected);
}

TEST(DeviceBufferCopy, PartialCopyToHostReadsPrefix) {
    auto host = Iota(6);
    DeviceBuffer<float> buf(host.data(), host.size());

    std::vector<float> out(3, 0.0f);
    buf.CopyToHost(out.data(), out.size());

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f}));
}

TEST(DeviceBufferMove, ConstructorTransfersOwnership) {
    auto host = Iota(4);
    DeviceBuffer<float> src(host.data(), host.size());
    float* ptr = src.Data();

    DeviceBuffer<float> dst(std::move(src));

    EXPECT_EQ(dst.Data(), ptr);
    EXPECT_EQ(dst.Size(), 4u);
    EXPECT_EQ(Download(dst), host);
    EXPECT_EQ(src.Data(), nullptr);
    EXPECT_EQ(src.Size(), 0u);
}

TEST(DeviceBufferMove, AssignmentTransfersOwnership) {
    auto host = Iota(4);
    DeviceBuffer<float> src(host.data(), host.size());
    DeviceBuffer<float> dst(100);
    float* ptr = src.Data();

    dst = std::move(src);

    EXPECT_EQ(dst.Data(), ptr);
    EXPECT_EQ(dst.Size(), 4u);
    EXPECT_EQ(Download(dst), host);
    EXPECT_EQ(src.Data(), nullptr);
    EXPECT_EQ(src.Size(), 0u);
}

TEST(DeviceBufferMove, CanBeStoredInVector) {
    std::vector<DeviceBuffer<float>> buffers;
    for (size_t i = 1; i <= 10; ++i) buffers.emplace_back(i);

    for (size_t i = 0; i < buffers.size(); ++i) {
        EXPECT_EQ(buffers[i].Size(), i + 1);
        EXPECT_NE(buffers[i].Data(), nullptr);
    }
}

TEST(DeviceBufferResize, ChangesSize) {
    DeviceBuffer<float> buf(4);
    buf.Resize(16);
    EXPECT_EQ(buf.Size(), 16u);
    EXPECT_EQ(buf.Bytes(), 16 * sizeof(float));
    EXPECT_NE(buf.Data(), nullptr);
}

TEST(DeviceBufferResize, SameSizeKeepsAllocation) {
    DeviceBuffer<float> buf(8);
    float* ptr = buf.Data();
    buf.Resize(8);
    EXPECT_EQ(buf.Data(), ptr);
    EXPECT_EQ(buf.Size(), 8u);
}

TEST(DeviceBufferResize, ToZeroReleasesMemory) {
    DeviceBuffer<float> buf(8);
    buf.Resize(0);
    EXPECT_EQ(buf.Size(), 0u);
    EXPECT_EQ(buf.Data(), nullptr);
}

TEST(DeviceBufferResize, FromEmptyAllocates) {
    DeviceBuffer<float> buf;
    buf.Resize(3);
    EXPECT_EQ(buf.Size(), 3u);
    EXPECT_NE(buf.Data(), nullptr);
}

TEST(DeviceBufferResize, ResizedBufferIsUsable) {
    DeviceBuffer<float> buf(2);
    buf.Resize(5);
    auto host = Iota(5);
    buf.CopyFromHost(host.data(), host.size());
    EXPECT_EQ(Download(buf), host);
}

TEST(DeviceBufferCopy, CopyToDeviceCopiesBetweenBuffers) {
    auto data = Iota(6);
    DeviceBuffer<float> src(data.data(), data.size());
    DeviceBuffer<float> dst(6);
    src.CopyToDevice(dst.Data(), 6);
    EXPECT_EQ(Download(dst), data);
}

TEST(DeviceBufferCopy, CopyToDeviceThrowsWhenCountTooLarge) {
    DeviceBuffer<float> src(4);
    DeviceBuffer<float> dst(8);
    EXPECT_THROW(src.CopyToDevice(dst.Data(), 5), std::invalid_argument);
}

TEST(DeviceBufferMemset, ZeroFillsWholeBuffer) {
    auto data = Iota(5);
    DeviceBuffer<float> buf(data.data(), data.size());
    buf.Memset(0, buf.Size());
    EXPECT_EQ(Download(buf), std::vector<float>(5, 0.0f));
}

TEST(DeviceBufferMemset, PartialCountLeavesTailUntouched) {
    auto data = Iota(5);
    DeviceBuffer<double> buf(5);
    std::vector<double> init(data.begin(), data.end());
    buf.CopyFromHost(init.data(), init.size());
    buf.Memset(0, 2);
    std::vector<double> out(5);
    buf.CopyToHost(out.data(), out.size());
    EXPECT_EQ(out, (std::vector<double>{0, 0, 3, 4, 5}));
}

TEST(DeviceBufferMemset, ThrowsWhenCountTooLarge) {
    DeviceBuffer<float> buf(3);
    EXPECT_THROW(buf.Memset(0, 4), std::invalid_argument);
}

TEST(DeviceBufferMemset, ZeroCountOnEmptyBufferIsNoOp) {
    DeviceBuffer<float> buf;
    EXPECT_NO_THROW(buf.Memset(0, 0));
}
