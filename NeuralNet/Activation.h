#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "Math/Tensor.h"
#include "Module.h"

namespace activation {

    using math::Tensor;
    using nn::Module;

    template <typename T>
    class ReLU : public Module<T> {
    public:
        Tensor<T> Forward(const Tensor<T>& input) {
            Tensor<T> output = input.Clone();

            std::for_each(output.Data(), output.Data() + output.Size(), [](T& value) {
                if (value < T(0)) {
                    value = T(0);
                }
            });

            return output;
        }
    };

    template <typename T>
    class Sigmoid : public Module<T> {
    public:
        Tensor<T> Forward(const Tensor<T>& input) {
            Tensor<T> output = input.Clone();

            std::for_each(output.Data(), output.Data() + output.Size(), [](T& value) {
                value = T(1) / (T(1) + std::exp(-value));
            });

            return output;
        }
    };

    template <typename T>
    class Tanh : public Module<T> {
    public:
        Tensor<T> Forward(const Tensor<T>& input) {
            Tensor<T> output = input.Clone();

            std::for_each(output.Data(), output.Data() + output.Size(), [](T& value) {
                value = std::tanh(value);
            });

            return output;
        }
    };

    template <typename T>
    class Softmax : public Module<T> {
    public:
        explicit Softmax(long long axis = -1) : m_axis(axis) {}

        Tensor<T> Forward(const Tensor<T>& input) {
            const size_t axis = math::detail::NormalizeAxis(m_axis, input.Rank());

            // Clone() compacts, so the flat decomposition below holds for views.
            Tensor<T> output = input.Clone();
            const std::vector<size_t>& shape = output.Shape();

            size_t outer = 1;
            for (size_t i = 0; i < axis; ++i) {
                outer *= shape[i];
            }
            const size_t count = shape[axis];
            size_t inner = 1;
            for (size_t i = axis + 1; i < shape.size(); ++i) {
                inner *= shape[i];
            }

            T* data = output.Data();

            for (size_t o = 0; o < outer; ++o) {
                for (size_t i = 0; i < inner; ++i) {
                    T* base = data + o * count * inner + i;

                    T max_value = base[0];
                    for (size_t a = 1; a < count; ++a) {
                        max_value = std::max(max_value, base[a * inner]);
                    }

                    T sum = T(0);
                    for (size_t a = 0; a < count; ++a) {
                        base[a * inner] = std::exp(base[a * inner] - max_value);
                        sum += base[a * inner];
                    }

                    for (size_t a = 0; a < count; ++a) {
                        base[a * inner] /= sum;
                    }
                }
            }

            return output;
        }

        long long Axis() const noexcept { return m_axis; }

    private:
        long long m_axis;
    };
}