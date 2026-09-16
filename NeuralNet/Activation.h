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
        Tensor<T> forward(const Tensor<T>& input) override {
            Tensor<T> output = input.clone();

            std::for_each(output.data(), output.data() + output.size(), [](T& value) {
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
        Tensor<T> forward(const Tensor<T>& input) override {
            Tensor<T> output = input.clone();

            std::for_each(output.data(), output.data() + output.size(), [](T& value) {
                value = T(1) / (T(1) + std::exp(-value));
            });

            return output;
        }
    };

    template <typename T>
    class Tanh : public Module<T> {
    public:
        Tensor<T> forward(const Tensor<T>& input) override {
            Tensor<T> output = input.clone();

            std::for_each(output.data(), output.data() + output.size(), [](T& value) {
                value = std::tanh(value);
            });

            return output;
        }
    };

    template <typename T>
    class Softmax : public Module<T> {
    public:
        Tensor<T> forward(const Tensor<T>& input) override {
            if (input.rank() != 2) {
                throw std::invalid_argument("Softmax expects a rank-2 tensor (batch_size, num_classes).");
            }

            Tensor<T> output = input.clone();
            const size_t batch_size = input.shape()[0];
            const size_t num_classes = input.shape()[1];

            for (size_t i = 0; i < batch_size; ++i) {
                T max_val = *std::max_element(output.data() + i * num_classes, output.data() + (i + 1) * num_classes);
                T sum_exp = T(0);

                for (size_t j = 0; j < num_classes; ++j) {
                    output(i, j) = std::exp(output(i, j) - max_val);
                    sum_exp += output(i, j);
                }

                for (size_t j = 0; j < num_classes; ++j) {
                    output(i, j) /= sum_exp;
                }
            }

            return output;
        }
    };
}