#pragma once

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
      return tensor::Activate(input, tensor::Activation::ReLU);
    }
  };

  template <typename T = float>
  class GELU : public Module<T> {
  public:
    Tensor<T> Forward(const Tensor<T> &input) {
      return tensor::Activate(input, tensor::Activation::GELU);
    }
  };

  template <typename T = float>
  class GELUTanh : public Module<T> {
  public:
    Tensor<T> Forward(const Tensor<T> &input) {
      return tensor::Activate(input, tensor::Activation::GELUTanh);
    }
  };

  template <typename T = float>
  class Sigmoid : public Module<T> {
  public:
    Tensor<T> Forward(const Tensor<T> &input) {
      return tensor::Activate(input, tensor::Activation::Sigmoid);
    }
  };

  template <typename T = float>
  class SiLU : public Module<T> {
  public:
    Tensor<T> Forward(const Tensor<T> &input) {
      return tensor::Activate(input, tensor::Activation::SiLU);
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
      Tensor<T> gate = tensor::Activate(m_gate_proj->Forward(input), tensor::Activation::SiLU);
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
      return tensor::Activate(input, tensor::Activation::Tanh);
    }
  };

  template <typename T = float>
  class Softmax : public Module<T> {
  public:
    explicit Softmax(long long axis = -1) : m_axis(axis) {}

    Tensor<T> Forward(const Tensor<T> &input) {
      return tensor::Softmax(input, m_axis);
    }

    long long Axis() const noexcept { return m_axis; }

  private:
    long long m_axis;
  };

}
