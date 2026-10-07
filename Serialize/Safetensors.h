#pragma once

#include <nlohmann/json.hpp>

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "Tensor/Tensor.h"

namespace serialize
{
    static_assert(std::endian::native == std::endian::little,
                  "Safetensors data is little-endian; big-endian hosts are not supported.");

    enum class DType
    {
        F64,
        F32,
        F16,
        BF16,
        I64,
        I32,
        I16,
        I8,
        U8,
        BOOL
    };

    struct SafetensorsEntry
    {
        DType dtype;
        std::vector<size_t> shape;
        uint64_t begin = 0;
        uint64_t end = 0;
    };

    struct SafetensorsHeader
    {
        std::map<std::string, SafetensorsEntry> tensors;
        std::map<std::string, std::string> metadata;
        uint64_t data_start = 0;
    };

    namespace detail
    {
        inline std::runtime_error Error(const std::string &message)
        {
            return std::runtime_error("Safetensors: " + message);
        }

        inline DType ParseDType(const std::string &name)
        {
            static const std::unordered_map<std::string, DType> dtypes = {
                {"F64", DType::F64}, {"F32", DType::F32}, {"F16", DType::F16}, {"BF16", DType::BF16},
                {"I64", DType::I64}, {"I32", DType::I32}, {"I16", DType::I16}, {"I8", DType::I8},
                {"U8", DType::U8}, {"BOOL", DType::BOOL}};

            const auto it = dtypes.find(name);
            if (it == dtypes.end())
            {
                throw Error("unsupported dtype '" + name + "'.");
            }
            return it->second;
        }

        inline size_t DTypeSize(DType dtype)
        {
            switch (dtype)
            {
            case DType::F64:
            case DType::I64:
                return 8;
            case DType::F32:
            case DType::I32:
                return 4;
            case DType::F16:
            case DType::BF16:
            case DType::I16:
                return 2;
            case DType::I8:
            case DType::U8:
            case DType::BOOL:
                return 1;
            }
            throw std::logic_error("Safetensors: unknown dtype.");
        }

        inline bool MultiplyChecked(uint64_t a, uint64_t b, uint64_t &result)
        {
            if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
            {
                return false;
            }
            result = a * b;
            return true;
        }

        inline float HalfToFloat(uint16_t half)
        {
            const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
            uint32_t exponent = (half >> 10) & 0x1Fu;
            uint32_t mantissa = half & 0x3FFu;

            if (exponent == 0x1Fu)
            {
                return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));
            }
            if (exponent != 0)
            {
                return std::bit_cast<float>(sign | ((exponent + 112u) << 23) | (mantissa << 13));
            }
            if (mantissa == 0)
            {
                return std::bit_cast<float>(sign);
            }

            exponent = 113u;
            while ((mantissa & 0x400u) == 0)
            {
                mantissa <<= 1;
                --exponent;
            }
            return std::bit_cast<float>(sign | (exponent << 23) | ((mantissa & 0x3FFu) << 13));
        }

        inline float BFloat16ToFloat(uint16_t value)
        {
            return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
        }

        template <typename Source, typename T, typename Decode>
        void ConvertAll(const char *bytes, T *out, size_t count, Decode decode)
        {
            for (size_t i = 0; i < count; ++i)
            {
                Source value;
                std::memcpy(&value, bytes + i * sizeof(Source), sizeof(Source));
                out[i] = static_cast<T>(decode(value));
            }
        }

        template <typename T>
        void Convert(DType dtype, const char *bytes, T *out, size_t count)
        {
            const auto same = [](auto value) { return value; };

            switch (dtype)
            {
            case DType::F64:
                return ConvertAll<double>(bytes, out, count, same);
            case DType::F32:
                return ConvertAll<float>(bytes, out, count, same);
            case DType::F16:
                return ConvertAll<uint16_t>(bytes, out, count, HalfToFloat);
            case DType::BF16:
                return ConvertAll<uint16_t>(bytes, out, count, BFloat16ToFloat);
            case DType::I64:
                return ConvertAll<int64_t>(bytes, out, count, same);
            case DType::I32:
                return ConvertAll<int32_t>(bytes, out, count, same);
            case DType::I16:
                return ConvertAll<int16_t>(bytes, out, count, same);
            case DType::I8:
                return ConvertAll<int8_t>(bytes, out, count, same);
            case DType::U8:
                return ConvertAll<uint8_t>(bytes, out, count, same);
            case DType::BOOL:
                return ConvertAll<uint8_t>(bytes, out, count, [](uint8_t value) { return value != 0; });
            }
            throw std::logic_error("Safetensors: unknown dtype.");
        }

        inline std::ifstream Open(const std::string &path)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file)
            {
                throw Error("failed to open file: " + path);
            }
            return file;
        }

        inline uint64_t FileSize(std::ifstream &file)
        {
            file.seekg(0, std::ios::end);
            const std::streamoff size = file.tellg();
            file.seekg(0, std::ios::beg);
            if (size < 0 || !file)
            {
                throw Error("failed to determine the file size.");
            }
            return static_cast<uint64_t>(size);
        }

        inline nlohmann::json ReadHeaderJson(std::ifstream &file, uint64_t file_size, uint64_t &header_size)
        {
            if (file_size < sizeof(uint64_t))
            {
                throw Error("file is too small to hold a header.");
            }

            file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
            if (!file || header_size > file_size - sizeof(uint64_t))
            {
                throw Error("header size runs past the end of the file.");
            }

            std::string text(header_size, '\0');
            file.read(text.data(), static_cast<std::streamsize>(header_size));
            if (!file)
            {
                throw Error("failed to read the header.");
            }

            nlohmann::json json = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
            if (json.is_discarded())
            {
                throw Error("header is not valid JSON.");
            }
            if (!json.is_object())
            {
                throw Error("header must be a JSON object.");
            }
            return json;
        }

        inline std::map<std::string, std::string> ParseMetadata(const nlohmann::json &value)
        {
            if (!value.is_object())
            {
                throw Error("__metadata__ must be an object.");
            }

            std::map<std::string, std::string> metadata;
            for (const auto &[key, text] : value.items())
            {
                if (!text.is_string())
                {
                    throw Error("__metadata__ value for '" + key + "' must be a string.");
                }
                metadata[key] = text.get<std::string>();
            }
            return metadata;
        }

        inline SafetensorsEntry ParseEntry(const std::string &name, const nlohmann::json &value, uint64_t data_size)
        {
            const auto fail = [&name](const std::string &reason) {
                return Error("tensor '" + name + "' " + reason);
            };

            if (!value.is_object())
            {
                throw fail("must be an object.");
            }

            const auto dtype = value.find("dtype");
            const auto shape = value.find("shape");
            const auto offsets = value.find("data_offsets");
            if (dtype == value.end() || shape == value.end() || offsets == value.end())
            {
                throw fail("needs dtype, shape and data_offsets.");
            }
            if (!dtype->is_string())
            {
                throw fail("has a dtype that is not a string.");
            }
            if (!shape->is_array())
            {
                throw fail("has a shape that is not an array.");
            }
            if (!offsets->is_array() || offsets->size() != 2 ||
                !(*offsets)[0].is_number_unsigned() || !(*offsets)[1].is_number_unsigned())
            {
                throw fail("has invalid data_offsets.");
            }

            SafetensorsEntry entry;
            entry.dtype = ParseDType(dtype->get<std::string>());

            uint64_t bytes = DTypeSize(entry.dtype);
            for (const auto &dim : *shape)
            {
                if (!dim.is_number_unsigned())
                {
                    throw fail("has a shape with a non-integer dimension.");
                }
                entry.shape.push_back(dim.get<size_t>());
                if (!MultiplyChecked(bytes, entry.shape.back(), bytes))
                {
                    throw fail("has a shape too large to address.");
                }
            }

            entry.begin = (*offsets)[0].get<uint64_t>();
            entry.end = (*offsets)[1].get<uint64_t>();
            if (entry.begin > entry.end || entry.end > data_size)
            {
                throw fail("has data_offsets outside the data section.");
            }
            if (entry.end - entry.begin != bytes)
            {
                throw fail("has a byte range that does not match its shape and dtype.");
            }

            return entry;
        }

        inline SafetensorsHeader ParseHeader(std::ifstream &file)
        {
            const uint64_t file_size = FileSize(file);
            uint64_t header_size = 0;
            const nlohmann::json json = ReadHeaderJson(file, file_size, header_size);

            SafetensorsHeader header;
            header.data_start = sizeof(uint64_t) + header_size;
            const uint64_t data_size = file_size - header.data_start;

            for (const auto &[name, value] : json.items())
            {
                if (name == "__metadata__")
                {
                    header.metadata = ParseMetadata(value);
                }
                else
                {
                    header.tensors.emplace(name, ParseEntry(name, value, data_size));
                }
            }
            return header;
        }

        template <typename T>
        void ReadTensorData(std::ifstream &file, const SafetensorsEntry &entry, T *out, size_t count, std::vector<char> &scratch)
        {
            const size_t bytes = static_cast<size_t>(entry.end - entry.begin);

            if constexpr (std::is_same_v<T, float>)
            {
                if (entry.dtype == DType::F32)
                {
                    file.read(reinterpret_cast<char *>(out), static_cast<std::streamsize>(bytes));
                    return;
                }
            }

            scratch.resize(bytes);
            file.read(scratch.data(), static_cast<std::streamsize>(bytes));
            if (file)
            {
                Convert(entry.dtype, scratch.data(), out, count);
            }
        }
    }

    inline SafetensorsHeader ReadSafetensorsHeader(const std::string &path)
    {
        std::ifstream file = detail::Open(path);
        return detail::ParseHeader(file);
    }

    template <typename T = float>
    std::map<std::string, tensor::Tensor<T>> ReadSafetensors(const std::string &path)
    {
        std::ifstream file = detail::Open(path);
        const SafetensorsHeader header = detail::ParseHeader(file);

        std::map<std::string, tensor::Tensor<T>> tensors;
        std::vector<char> scratch;

        for (const auto &[name, entry] : header.tensors)
        {
            for (size_t dim : entry.shape)
            {
                if (dim == 0)
                {
                    throw detail::Error("tensor '" + name + "' has a zero-sized dimension, which Tensor cannot hold.");
                }
            }

            tensor::Tensor<T> tensor(entry.shape);
            file.seekg(static_cast<std::streamoff>(header.data_start + entry.begin));
            detail::ReadTensorData(file, entry, tensor.Data(), tensor.Size(), scratch);
            if (!file)
            {
                throw detail::Error("failed to read data for tensor '" + name + "'.");
            }

            tensors.emplace(name, std::move(tensor));
        }

        return tensors;
    }
}
