#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Math/Tensor.h"
#include "NeuralNet/Module.h"
#include "NeuralNet/Transformer/Normal/Norm.h"

namespace nn::transformer::norm{

    template <typename T = float>
    class LayerNorm : public Norm<T>{
    public:
        explicit LayerNorm(std::vector<size_t> normalized_shape, T epsilon = static_cast<T>(1e-5)) :
            m_normalized_shape(std::move(normalized_shape)),
            m_epsilon(epsilon) {
            if (m_normalized_shape.empty()) {
                throw std::invalid_argument("LayerNorm: normalized_shape needs at least one dimension.");
            }
            for (size_t d : m_normalized_shape) {
                if (d == 0) {
                    throw std::invalid_argument("LayerNorm: normalized_shape dimensions must be greater than zero.");
                }
                m_normalized_size *= d;
            }

            gamma = std::make_shared<Tensor<T>>(m_normalized_shape, T(1));
            beta = std::make_shared<Tensor<T>>(m_normalized_shape, T(0));
            this->RegisterParameter("gamma", gamma);
            this->RegisterParameter("beta", beta);
        }

        explicit LayerNorm(size_t feature_size, T epsilon = static_cast<T>(1e-5)) :
            LayerNorm(std::vector<size_t>{feature_size}, epsilon) {}

        LayerNorm(Tensor<T> initial_gamma, Tensor<T> initial_beta, T epsilon = static_cast<T>(1e-5)) :
            m_normalized_shape(initial_gamma.Shape()),
            m_epsilon(epsilon) {
            if (initial_gamma.Shape() != initial_beta.Shape()) {
                throw std::invalid_argument("LayerNorm: gamma and beta must have the same shape.");
            }
            if (m_normalized_shape.empty()) {
                throw std::invalid_argument("LayerNorm: gamma needs at least one dimension.");
            }
            for (size_t d : m_normalized_shape) {
                m_normalized_size *= d;
            }

            gamma = std::make_shared<Tensor<T>>(std::move(initial_gamma));
            beta = std::make_shared<Tensor<T>>(std::move(initial_beta));
            this->RegisterParameter("gamma", gamma);
            this->RegisterParameter("beta", beta);
        }

        Tensor<T> Forward(const Tensor<T>& input) override{
            const std::vector<size_t>& shape = input.Shape();
            const size_t axes = m_normalized_shape.size();

            if (shape.size() < axes) {
                throw std::invalid_argument("LayerNorm: input rank is smaller than normalized_shape.");
            }
            const size_t first = shape.size() - axes;
            for (size_t i = 0; i < axes; ++i) {
                if (shape[first + i] != m_normalized_shape[i]) {
                    throw std::invalid_argument("LayerNorm: trailing input dimensions do not match normalized_shape.");
                }
            }

            // Clone() compacts, so the flat walk below is valid even when the
            // caller hands us a strided view.
            Tensor<T> output = input.Clone();
            const Tensor<T> gamma_c = gamma->Contiguous();
            const Tensor<T> beta_c = beta->Contiguous();

            T* out = output.Data();
            const T* g = gamma_c.Data();
            const T* b = beta_c.Data();

            const size_t inner = m_normalized_size;
            const size_t outer = output.Size() / inner;

            for (size_t o = 0; o < outer; ++o) {
                T* block = out + o * inner;

                T sum = T(0);
                for (size_t i = 0; i < inner; ++i) {
                    sum += block[i];
                }
                const T mean = sum / static_cast<T>(inner);

                // Biased variance (divide by N), matching torch.nn.LayerNorm.
                T squared = T(0);
                for (size_t i = 0; i < inner; ++i) {
                    const T centred = block[i] - mean;
                    squared += centred * centred;
                }
                const T inv_stddev = T(1) / std::sqrt(squared / static_cast<T>(inner) + m_epsilon);

                for (size_t i = 0; i < inner; ++i) {
                    block[i] = (block[i] - mean) * inv_stddev * g[i] + b[i];
                }
            }

            return output;
        }

        const std::vector<size_t>& NormalizedShape() const noexcept { return m_normalized_shape; }
        size_t NormalizedSize() const noexcept { return m_normalized_size; }
        T Epsilon() const noexcept { return m_epsilon; }

        std::shared_ptr<Tensor<T>> gamma;
        std::shared_ptr<Tensor<T>> beta;

    private:
        std::vector<size_t> m_normalized_shape;
        size_t m_normalized_size = 1;
        T m_epsilon;
    };
}
