#include <gtest/gtest.h>

#include <Serialize/Safetensors.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using math::Tensor;
using serialize::DType;
using serialize::ReadSafetensors;
using serialize::ReadSafetensorsHeader;

namespace {

// Builds a .safetensors file in memory: an 8-byte little-endian header length,
// the JSON header, then the raw tensor bytes.
class SafetensorsBuilder {
public:
    template <typename Raw>
    SafetensorsBuilder& Add(const std::string& name, const std::string& dtype,
                            const std::vector<size_t>& shape, const std::vector<Raw>& values) {
        const size_t begin = m_data.size();
        const auto* bytes = reinterpret_cast<const char*>(values.data());
        m_data.insert(m_data.end(), bytes, bytes + values.size() * sizeof(Raw));
        m_header[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {begin, m_data.size()}}};
        return *this;
    }

    SafetensorsBuilder& Metadata(const std::string& key, const std::string& value) {
        m_header["__metadata__"][key] = value;
        return *this;
    }

    nlohmann::json& Header() { return m_header; }

    std::string Bytes() const { return Assemble(m_header.dump()); }

    std::string Assemble(const std::string& header_text) const {
        const uint64_t size = header_text.size();
        std::string out(reinterpret_cast<const char*>(&size), sizeof(size));
        out += header_text;
        out.append(m_data.begin(), m_data.end());
        return out;
    }

private:
    nlohmann::json m_header = nlohmann::json::object();
    std::vector<char> m_data;
};

// A uniquely named file in the temp directory, removed on destruction.
class TempFile {
public:
    explicit TempFile(const std::string& contents)
        : m_path(std::filesystem::temp_directory_path() /
                 ("safetensors_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".safetensors")) {
        std::ofstream(m_path, std::ios::binary) << contents;
    }
    ~TempFile() { std::filesystem::remove(m_path); }

    std::string Path() const { return m_path.string(); }

private:
    std::filesystem::path m_path;
};

}  // namespace

// --- reading values ---------------------------------------------------------

TEST(SafetensorsRead, ReadsAnF32Tensor) {
    const TempFile file(SafetensorsBuilder()
                            .Add<float>("w", "F32", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
                            .Bytes());

    const auto tensors = ReadSafetensors(file.Path());

    ASSERT_EQ(tensors.size(), 1u);
    const Tensor<float>& w = tensors.at("w");
    EXPECT_EQ(w.Shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(w(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(w(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(w(1, 1), 5.0f);
}

TEST(SafetensorsRead, ReadsEveryTensorByName) {
    const TempFile file(SafetensorsBuilder()
                            .Add<float>("a.weight", "F32", {2}, {1.0f, 2.0f})
                            .Add<float>("a.bias", "F32", {1}, {3.0f})
                            .Add<float>("b.weight", "F32", {2, 2}, {4.0f, 5.0f, 6.0f, 7.0f})
                            .Bytes());

    const auto tensors = ReadSafetensors(file.Path());

    ASSERT_EQ(tensors.size(), 3u);
    EXPECT_FLOAT_EQ(tensors.at("a.weight")(1), 2.0f);
    EXPECT_FLOAT_EQ(tensors.at("a.bias")(0), 3.0f);
    EXPECT_FLOAT_EQ(tensors.at("b.weight")(1, 1), 7.0f);
}

TEST(SafetensorsRead, FollowsDataOffsetsNotHeaderOrder) {
    // Put the second tensor's bytes first by swapping the offsets.
    SafetensorsBuilder builder;
    builder.Add<float>("first", "F32", {2}, {1.0f, 2.0f}).Add<float>("second", "F32", {2}, {3.0f, 4.0f});
    std::swap(builder.Header()["first"]["data_offsets"], builder.Header()["second"]["data_offsets"]);
    const TempFile file(builder.Bytes());

    const auto tensors = ReadSafetensors(file.Path());

    EXPECT_FLOAT_EQ(tensors.at("first")(0), 3.0f);
    EXPECT_FLOAT_EQ(tensors.at("second")(0), 1.0f);
}

TEST(SafetensorsRead, ReadsAScalar) {
    const TempFile file(SafetensorsBuilder().Add<float>("s", "F32", {}, {42.0f}).Bytes());

    const auto tensors = ReadSafetensors(file.Path());
    const Tensor<float>& s = tensors.at("s");

    EXPECT_TRUE(s.Shape().empty());
    EXPECT_EQ(s.Size(), 1u);
    EXPECT_FLOAT_EQ(s.Data()[0], 42.0f);
}

TEST(SafetensorsRead, ConvertsF16) {
    // 1.0, -2.0, 0.5, 65504 (largest half), smallest subnormal, +infinity
    const std::vector<uint16_t> halves = {0x3C00, 0xC000, 0x3800, 0x7BFF, 0x0001, 0x7C00};
    const TempFile file(SafetensorsBuilder().Add<uint16_t>("h", "F16", {6}, halves).Bytes());

    const auto tensors = ReadSafetensors(file.Path());
    const Tensor<float>& h = tensors.at("h");

    EXPECT_FLOAT_EQ(h(0), 1.0f);
    EXPECT_FLOAT_EQ(h(1), -2.0f);
    EXPECT_FLOAT_EQ(h(2), 0.5f);
    EXPECT_FLOAT_EQ(h(3), 65504.0f);
    EXPECT_FLOAT_EQ(h(4), 5.9604645e-8f);
    EXPECT_TRUE(std::isinf(h(5)));
}

TEST(SafetensorsRead, ConvertsBF16) {
    // bf16 is the top 16 bits of an f32: 1.0, -2.0, 3.140625
    const std::vector<uint16_t> values = {0x3F80, 0xC000, 0x4049};
    const TempFile file(SafetensorsBuilder().Add<uint16_t>("b", "BF16", {3}, values).Bytes());

    const auto tensors = ReadSafetensors(file.Path());
    const Tensor<float>& b = tensors.at("b");

    EXPECT_FLOAT_EQ(b(0), 1.0f);
    EXPECT_FLOAT_EQ(b(1), -2.0f);
    EXPECT_FLOAT_EQ(b(2), 3.140625f);
}

TEST(SafetensorsRead, ConvertsF64) {
    const TempFile file(SafetensorsBuilder().Add<double>("d", "F64", {2}, {0.25, -8.5}).Bytes());

    const auto tensors = ReadSafetensors(file.Path());
    const Tensor<float>& d = tensors.at("d");

    EXPECT_FLOAT_EQ(d(0), 0.25f);
    EXPECT_FLOAT_EQ(d(1), -8.5f);
}

TEST(SafetensorsRead, ConvertsIntegerAndBoolTypes) {
    const TempFile file(SafetensorsBuilder()
                            .Add<int64_t>("i64", "I64", {2}, {-3, 7})
                            .Add<int32_t>("i32", "I32", {1}, {-40})
                            .Add<int16_t>("i16", "I16", {1}, {300})
                            .Add<int8_t>("i8", "I8", {1}, {-5})
                            .Add<uint8_t>("u8", "U8", {1}, {200})
                            .Add<uint8_t>("mask", "BOOL", {2}, {0, 1})
                            .Bytes());

    const auto t = ReadSafetensors(file.Path());

    EXPECT_FLOAT_EQ(t.at("i64")(0), -3.0f);
    EXPECT_FLOAT_EQ(t.at("i64")(1), 7.0f);
    EXPECT_FLOAT_EQ(t.at("i32")(0), -40.0f);
    EXPECT_FLOAT_EQ(t.at("i16")(0), 300.0f);
    EXPECT_FLOAT_EQ(t.at("i8")(0), -5.0f);
    EXPECT_FLOAT_EQ(t.at("u8")(0), 200.0f);
    EXPECT_FLOAT_EQ(t.at("mask")(0), 0.0f);
    EXPECT_FLOAT_EQ(t.at("mask")(1), 1.0f);
}

TEST(SafetensorsRead, ConvertsIntoOtherElementTypes) {
    const TempFile file(SafetensorsBuilder()
                            .Add<float>("f", "F32", {2}, {1.5f, -2.25f})
                            .Add<int64_t>("ids", "I64", {3}, {464, 3290, 318})
                            .Bytes());

    const auto as_double = ReadSafetensors<double>(file.Path());
    EXPECT_DOUBLE_EQ(as_double.at("f")(1), -2.25);

    const auto as_int = ReadSafetensors<int64_t>(file.Path());
    EXPECT_EQ(as_int.at("ids")(2), 318);
}

TEST(SafetensorsRead, EmptyFileHasNoTensors) {
    const TempFile file(SafetensorsBuilder().Bytes());
    EXPECT_TRUE(ReadSafetensors(file.Path()).empty());
}

TEST(SafetensorsRead, IgnoresMetadata) {
    const TempFile file(SafetensorsBuilder()
                            .Metadata("format", "pt")
                            .Add<float>("w", "F32", {1}, {1.0f})
                            .Bytes());

    const auto tensors = ReadSafetensors(file.Path());

    ASSERT_EQ(tensors.size(), 1u);
    EXPECT_TRUE(tensors.contains("w"));
}

// --- header only ------------------------------------------------------------

TEST(SafetensorsHeader, DescribesEachTensorWithoutLoadingIt) {
    const TempFile file(SafetensorsBuilder()
                            .Add<float>("w", "F32", {2, 3}, std::vector<float>(6, 0.0f))
                            .Add<uint16_t>("b", "BF16", {4}, std::vector<uint16_t>(4, 0))
                            .Bytes());

    const auto header = ReadSafetensorsHeader(file.Path());

    ASSERT_EQ(header.tensors.size(), 2u);
    const auto& w = header.tensors.at("w");
    EXPECT_EQ(w.dtype, DType::F32);
    EXPECT_EQ(w.shape, (std::vector<size_t>{2, 3}));
    EXPECT_EQ(w.begin, 0u);
    EXPECT_EQ(w.end, 24u);
    const auto& b = header.tensors.at("b");
    EXPECT_EQ(b.dtype, DType::BF16);
    EXPECT_EQ(b.begin, 24u);
    EXPECT_EQ(b.end, 32u);
}

TEST(SafetensorsHeader, ReportsMetadataAndDataStart) {
    SafetensorsBuilder builder;
    builder.Metadata("format", "pt").Add<float>("w", "F32", {1}, {1.0f});
    const std::string bytes = builder.Bytes();
    const TempFile file(bytes);

    const auto header = ReadSafetensorsHeader(file.Path());

    EXPECT_EQ(header.metadata.at("format"), "pt");
    EXPECT_EQ(header.data_start, bytes.size() - sizeof(float));
}

// --- malformed files --------------------------------------------------------

TEST(SafetensorsErrors, MissingFile) {
    EXPECT_THROW(ReadSafetensors("does/not/exist.safetensors"), std::runtime_error);
    EXPECT_THROW(ReadSafetensorsHeader("does/not/exist.safetensors"), std::runtime_error);
}

TEST(SafetensorsErrors, FileTooSmallForTheLengthPrefix) {
    const TempFile file("abc");
    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, HeaderLengthPastTheEndOfTheFile) {
    const uint64_t huge = 1'000'000;
    const TempFile file(std::string(reinterpret_cast<const char*>(&huge), sizeof(huge)) + "{}");
    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, HeaderIsNotJson) {
    const TempFile file(SafetensorsBuilder().Assemble("{not json"));
    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, HeaderIsNotAnObject) {
    const TempFile file(SafetensorsBuilder().Assemble("[1, 2]"));
    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, UnknownDType) {
    SafetensorsBuilder builder;
    builder.Add<float>("w", "F32", {1}, {1.0f});
    builder.Header()["w"]["dtype"] = "F8_E4M3";
    const TempFile file(builder.Bytes());

    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, MissingField) {
    SafetensorsBuilder builder;
    builder.Add<float>("w", "F32", {1}, {1.0f});
    builder.Header()["w"].erase("shape");
    const TempFile file(builder.Bytes());

    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, OffsetsPastTheData) {
    SafetensorsBuilder builder;
    builder.Add<float>("w", "F32", {2}, {1.0f, 2.0f});
    builder.Header()["w"]["data_offsets"] = {0, 64};
    const TempFile file(builder.Bytes());

    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, OffsetsInTheWrongOrder) {
    SafetensorsBuilder builder;
    builder.Add<float>("w", "F32", {1}, {1.0f});
    builder.Header()["w"]["data_offsets"] = {4, 0};
    const TempFile file(builder.Bytes());

    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, ByteRangeDisagreesWithShape) {
    SafetensorsBuilder builder;
    builder.Add<float>("w", "F32", {2}, {1.0f, 2.0f});
    builder.Header()["w"]["shape"] = {3};
    const TempFile file(builder.Bytes());

    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, ZeroSizedTensorIsRejected) {
    const TempFile file(SafetensorsBuilder().Add<float>("empty", "F32", {0, 4}, std::vector<float>{}).Bytes());

    EXPECT_NO_THROW(ReadSafetensorsHeader(file.Path())) << "the header itself is valid";
    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, DTypeIsNotAString) {
    SafetensorsBuilder builder;
    builder.Add<float>("w", "F32", {1}, {1.0f});
    builder.Header()["w"]["dtype"] = 32;
    const TempFile file(builder.Bytes());

    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, ShapeIsNotAnArray) {
    // A bare number must not be read as a one-element shape.
    SafetensorsBuilder builder;
    builder.Add<float>("w", "F32", {1}, {1.0f});
    builder.Header()["w"]["shape"] = 1;
    const TempFile file(builder.Bytes());

    EXPECT_THROW(ReadSafetensors(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, ShapeThatOverflowsIsRejected) {
    // 2^32 * 2^32 * 4 bytes wraps a 64-bit count; it must not be mistaken for a small tensor.
    SafetensorsBuilder builder;
    builder.Add<float>("w", "F32", {1}, {1.0f});
    builder.Header()["w"]["shape"] = {uint64_t{1} << 32, uint64_t{1} << 32};
    const TempFile file(builder.Bytes());

    EXPECT_THROW(ReadSafetensorsHeader(file.Path()), std::runtime_error);
}

TEST(SafetensorsErrors, MetadataMustBeAnObjectOfStrings) {
    SafetensorsBuilder not_object;
    not_object.Header()["__metadata__"] = "pt";
    const TempFile a(not_object.Bytes());
    EXPECT_THROW(ReadSafetensorsHeader(a.Path()), std::runtime_error);

    SafetensorsBuilder not_string;
    not_string.Header()["__metadata__"] = {{"format", 1}};
    const TempFile b(not_string.Bytes());
    EXPECT_THROW(ReadSafetensorsHeader(b.Path()), std::runtime_error);
}
