#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <type_traits>

#include "Module.h"
#include "Math/Tensor.h"

namespace nn
{

    template <typename T, typename... Layers>
    class Sequential : public Module<T>
    {
    static_assert((std::is_same_v<typename Layers::value_type, T> && ...), "Sequential: every layer must use the same element type.");
    public:
        explicit Sequential(Layers... layers)
            requires(sizeof...(Layers) > 0)
            : Sequential(std::make_shared<Layers>(std::move(layers))...)
        {
        }

        explicit Sequential(std::shared_ptr<Layers>... layers)
            : m_layers(std::move(layers)...)
        {
            size_t index = 0;
            std::apply([&](auto &...layer)
                       { (this->RegisterSubmodule(std::to_string(index++), layer), ...); }, m_layers);
        }

        Tensor<T> Forward(const Tensor<T> &input)
        {
            Tensor<T> x = input;
            std::apply([&](auto &...layer)
                       { ((x = layer->Forward(x)), ...); }, m_layers);
            return x;
        }

        template <size_t I>
        auto &Get() { return std::get<I>(m_layers); }

    private:
        std::tuple<std::shared_ptr<Layers>...> m_layers;
    };

    template <typename First, typename... Rest>
    Sequential(First, Rest...) -> Sequential<typename First::value_type, First, Rest...>;

    template <typename First, typename... Rest>
    Sequential(std::shared_ptr<First>, std::shared_ptr<Rest>...) -> Sequential<typename First::value_type, First, Rest...>;
}
