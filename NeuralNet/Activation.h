#pragma once

#include <algorithm>
#include <cmath>

#include "Math/Tensor.h"
#include "Module.h"

namespace activation {

  using math::Tensor;
  using nn::Module;

  template <typename T = float>
  class ReLU : public Module<T> {
  public:
    Tensor<T> Forward(const Tensor<T> &input) {
      Tensor<T> output = input.Clone();

      std::for_each(output.Data(), output.Data() + output.Size(), [](T &value) {
        if (value < T(0)) {
          value = T(0);
        }
      });

      return output;
    }
  };

  template <typename T = float>
  class GELU : public Module<T> {
  public:
    Tensor<T> Forward(const Tensor<T> &input) {
      constexpr T SQRT_2_OVER_PI = T(0.7978845608028654);
      constexpr T COEFFICIENT = T(0.044715);

      Tensor<T> output = input.Clone();

      std::for_each(output.Data(), output.Data() + output.Size(), [](T &x) {
        const T inner = SQRT_2_OVER_PI * (x + COEFFICIENT * x * x * x);
        x = T(0.5) * x * (T(1) + std::tanh(inner));
      });

      return output;
    }
  };

  template <typename T = float>
  class Sigmoid : public Module<T> {
  public:
    Tensor<T> Forward(const Tensor<T> &input) {
      Tensor<T> output = input.Clone();

      std::for_each(output.Data(), output.Data() + output.Size(),
                    [](T &value) { value = T(1) / (T(1) + std::exp(-value)); });

      return output;
    }
  };

  template <typename T = float>
  class Tanh : public Module<T> {
  public:
    Tensor<T> Forward(const Tensor<T> &input) {
      Tensor<T> output = input.Clone();

      std::for_each(output.Data(), output.Data() + output.Size(),
                    [](T &value) { value = std::tanh(value); });

      return output;
    }
  };

  template <typename T = float>
  class Softmax : public Module<T> {
  public:
    explicit Softmax(long long axis = -1) : m_axis(axis) {}

    Tensor<T> Forward(const Tensor<T> &input) {
      const size_t axis = math::detail::NormalizeAxis(m_axis, input.Rank());

      // Clone() compacts, so the flat decomposition below holds for views.
      Tensor<T> output = input.Clone();
      const std::vector<size_t> &shape = output.Shape();

      size_t outer = 1;
      for (size_t i = 0; i < axis; ++i) {
        outer *= shape[i];
      }
      const size_t count = shape[axis];
      size_t inner = 1;
      for (size_t i = axis + 1; i < shape.size(); ++i) {
        inner *= shape[i];
      }

      T *data = output.Data();

      for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
          T *base = data + o * count * inner + i;

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
} // namespace activation