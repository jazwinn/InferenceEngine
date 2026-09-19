#pragma once

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace serialize
{
    template <typename T>
    concept Reflectable = requires(T &t) {
        T::Names();
        t.ApplyFunction([](auto &) {});
    };

    template <typename T>
    struct IsVector : std::false_type
    {
    };

    template <typename U, typename A>
    struct IsVector<std::vector<U, A>> : std::true_type
    {
    };

    template <typename T>
    void Read(T &out, const nlohmann::json &value)
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            if (value.is_boolean())
            {
                out = value.get<bool>();
            }
        }
        else if constexpr (std::is_enum_v<T>)
        {
            if (value.is_number_integer())
            {
                out = static_cast<T>(value.get<std::underlying_type_t<T>>());
            }
        }
        else if constexpr (std::is_unsigned_v<T>)
        {
            if (value.is_number_unsigned())
            {
                out = value.get<T>();
            }
        }
        else if constexpr (std::is_integral_v<T>)
        {
            if (value.is_number_integer())
            {
                out = value.get<T>();
            }
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            if (value.is_number())
            {
                out = value.get<T>();
            }
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            if (value.is_string())
            {
                out = value.get<std::string>();
            }
        }
        else if constexpr (IsVector<T>::value)
        {
            if (value.is_array())
            {
                for (const nlohmann::json &element : value)
                {
                    typename T::value_type item{};
                    Read(item, element);
                    out.push_back(std::move(item));
                }
            }
        }
        else if constexpr (Reflectable<T>)
        {
            if (value.is_object())
            {
                const auto names = T::Names();
                size_t index = 0;

                out.ApplyFunction([&](auto &member)
                {
                    const auto field = value.find(names[index++]);
                    if (field != value.end())
                    {
                        Read(member, *field);
                    }
                });
            }
        }
        else
        {
            static_assert(sizeof(T) == 0, "serialize::Read: unsupported member type.");
        }
    }

    template <typename T>
    T LoadJson(const nlohmann::json &doc)
    {
        T data{};

        if (doc.is_object())
        {
            Read(data, doc);
        }
        else if (doc.is_array())
        {
            for (const nlohmann::json &entry : doc)
            {
                if (entry.is_object() && entry.contains(T::classname()))
                {
                    Read(data, entry[T::classname()]);
                }
            }
        }

        return data;
    }

    template <typename T>
    T LoadJson(const std::string &filepath)
    {
        std::ifstream file(filepath);

        if (!file)
        {
            throw std::runtime_error("Failed to open JSON file: " + filepath);
        }

        return LoadJson<T>(nlohmann::json::parse(file));
    }

    template <typename T>
    T LoadJson(const char *filepath)
    {
        return LoadJson<T>(std::string(filepath));
    }
}
