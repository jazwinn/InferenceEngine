#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Tensor/TensorCore.h"
#include "Tensor/detail/Shape.h"

namespace tensor {

    enum class Activation { GELU, GELUTanh, SiLU, ReLU, Sigmoid, Tanh };

    namespace detail {

        template <typename T, typename F>
        Tensor<T> Map(const Tensor<T>& input, F f) {
            if (input.Size() == 0) {
                return Tensor<T>();
            }
            const Tensor<T> src = input.Contiguous();
            Tensor<T> output(src.Shape());
            const T* in = src.Data();
            T* out = output.Data();
            for (size_t i = 0; i < src.Size(); ++i) {
                out[i] = f(in[i]);
            }
            return output;
        }

    }

    template <typename T>
    Tensor<T> Activate(const Tensor<T>& input, Activation kind) {
        input.RequireCPU("Activate");

        switch (kind) {
        case Activation::ReLU:
            return detail::Map(input, [](T x) { return x < T(0) ? T(0) : x; });
        case Activation::GELU:
            return detail::Map(input, [](T x) {
                constexpr T ONE_OVER_SQRT_2 = T(0.7071067811865476);
                return T(0.5) * x * (T(1) + std::erf(x * ONE_OVER_SQRT_2));
            });
        case Activation::GELUTanh:
            return detail::Map(input, [](T x) {
                constexpr T SQRT_2_OVER_PI = T(0.7978845608028654);
                constexpr T COEFFICIENT = T(0.044715);
                const T inner = SQRT_2_OVER_PI * (x + COEFFICIENT * x * x * x);
                return T(0.5) * x * (T(1) + std::tanh(inner));
            });
        case Activation::Sigmoid:
            return detail::Map(input, [](T x) { return T(1) / (T(1) + std::exp(-x)); });
        case Activation::SiLU:
            return detail::Map(input, [](T x) { return x / (T(1) + std::exp(-x)); });
        case Activation::Tanh:
            return detail::Map(input, [](T x) { return std::tanh(x); });
        }
        throw std::invalid_argument("Activate: unsupported activation.");
    }

    template <typename T>
    Tensor<T> Softmax(const Tensor<T>& input, long long axis = -1) {
        const size_t ax = detail::NormalizeAxis(axis, input.Rank());
        input.RequireCPU("Softmax");

        const Tensor<T> src = input.Contiguous();
        const std::vector<size_t>& shape = src.Shape();
        Tensor<T> output(shape);

        size_t outer = 1;
        for (size_t i = 0; i < ax; ++i) {
            outer *= shape[i];
        }
        const size_t count = shape[ax];
        size_t inner = 1;
        for (size_t i = ax + 1; i < shape.size(); ++i) {
            inner *= shape[i];
        }

        const T* in = src.Data();
        T* out = output.Data();

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                const size_t base = o * count * inner + i;

                T max_value = in[base];
                for (size_t a = 1; a < count; ++a) {
                    max_value = std::max(max_value, in[base + a * inner]);
                }

                T sum = T(0);
                for (size_t a = 0; a < count; ++a) {
                    out[base + a * inner] = std::exp(in[base + a * inner] - max_value);
                    sum += out[base + a * inner];
                }

                for (size_t a = 0; a < count; ++a) {
                    out[base + a * inner] /= sum;
                }
            }
        }

        return output;
    }

}
