#pragma once

#include <algorithm>
#include <cmath>
#include <memory>

#include "Tensor/Tensor.h"
#include "Module.h"
#include "Linear.h"

namespace nn{

  using tensor::Tensor;
  using nn::Module;
  using nn::Linear;

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
      constexpr T ONE_OVER_SQRT_2 = T(0.7071067811865476);

      Tensor<T> output = input.Clone();

      std::for_each(output.Data(), output.Data() + output.Size(), [](T &x) {
        x = T(0.5) * x * (T(1) + std::erf(x * ONE_OVER_SQRT_2));
      });

      return output;
    }
  };

  template <typename T = float>
  class GELUTanh : public Module<T> {
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
  class SiLU : public Module<T> {
  public:
    Tensor<T> Forward(const Tensor<T> &input) {
      Tensor<T> output = input.Clone();

      std::for_each(output.Data(), output.Data() + output.Size(),
                    [](T &value) { value = value / (T(1) + std::exp(-value)); });

      return output;
    }
  };

  template <typename T = float>
  class SwiGLU : public Module<T> {
  public:
    SwiGLU(size_t d_model, size_t d_ff)
        : m_gate_proj(std::make_shared<Linear<T>>(d_model, d_ff, false)),
          m_up_proj(std::make_shared<Linear<T>>(d_model, d_ff, false)),
          m_down_proj(std::make_shared<Linear<T>>(d_ff, d_model, false)) {
      this->RegisterSubmodule("gate_proj", m_gate_proj);
      this->RegisterSubmodule("up_proj", m_up_proj);
      this->RegisterSubmodule("down_proj", m_down_proj);
    }

    Tensor<T> Forward(const Tensor<T> &input) {
      Tensor<T> gate = SiLU<T>().Forward(m_gate_proj->Forward(input));
      Tensor<T> up = m_up_proj->Forward(input);

      return m_down_proj->Forward(gate * up);
    }

  private:
    std::shared_ptr<Linear<T>> m_gate_proj;
    std::shared_ptr<Linear<T>> m_up_proj;
    std::shared_ptr<Linear<T>> m_down_proj;
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
      const size_t axis = tensor::detail::NormalizeAxis(m_axis, input.Rank());

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



} 