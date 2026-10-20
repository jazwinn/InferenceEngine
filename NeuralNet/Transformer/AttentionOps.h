#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>

#include "Tensor/Tensor.h"
#include "NeuralNet/Activation.h"

namespace nn::transformer
{

    using tensor::Tensor;
    using nn::Softmax;
    
    inline size_t ValidatedDivide(size_t numerator, size_t denominator)
    {
        if (denominator == 0)
        {
            throw std::invalid_argument("denominator must be greater than zero.");
        }
        if (numerator % denominator != 0)
        {
            throw std::invalid_argument("numerator must be divisible by denominator.");
        }
        return numerator / denominator;
    }

    inline Tensor<bool> NormalizeMask(const Tensor<bool>& mask, size_t batch, size_t heads, size_t seq_q, size_t seq_k)
    {
        if (mask.Size() == 0)
        {
            return mask;
        }

        Tensor<bool> result;
        if (mask.Rank() == 2)
        {
            result = mask.Unsqueeze(0).Unsqueeze(0);
        }
        else if (mask.Rank() == 3)
        {
            result = mask.Unsqueeze(1);
        }
        else if (mask.Rank() == 4)
        {
            result = mask;
        }
        else
        {
            throw std::invalid_argument("mask must have rank 2, 3 or 4.");
        }

        const std::vector<size_t>& shape = result.Shape();
        const std::vector<size_t> target = {batch, heads, seq_q, seq_k};
        for (size_t i = 0; i < target.size(); ++i)
        {
            if (shape[i] != 1 && shape[i] != target[i])
            {
                throw std::invalid_argument("mask is not broadcastable to (batch, heads, seq_q, seq_k).");
            }
        }
        return result;
    }

    template <typename T>
    Tensor<T> SplitHeads(const Tensor<T> &x, size_t num_heads, size_t head_dim)
    {
        std::vector<size_t> new_shape = {x.Shape()[0], x.Shape()[1], num_heads, head_dim};
        return x.Reshape(new_shape).Transpose(1, 2).Contiguous(); // (batch_size, num_heads, seq_length, head_dim)
    }

    template <typename T>
    Tensor<T> MergeHeads(const Tensor<T> &x, size_t d_model)
    {
        Tensor<T> transposed = x.Transpose(1, 2);                                         // (batch_size, seq_length, num_heads, head_dim)
        return transposed.Reshape(transposed.Shape()[0], transposed.Shape()[1], d_model); // (batch_size, seq_length, d_model)
    }

    template <typename T>
    Tensor<T> SDPA(const Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V, const Tensor<bool>& mask = Tensor<bool>(), bool causal = false, T scale = 0)
    {
        if(scale == 0) scale = 1/T(std::sqrt(Q.Shape().back()));

        Tensor<T> scores = MatMul(Q, K.Transpose(-2, -1)) * scale;

        const size_t seq_q = scores.Shape()[scores.Rank() - 2];
        const size_t seq_k = scores.Shape()[scores.Rank() - 1];


        if (causal && seq_q > 1)
        {
            Tensor<bool> causal_mask({seq_q, seq_k}, false);

            for (size_t i = 0; i < seq_q; ++i)
            {
                for (size_t j = i + 1 + (seq_k - seq_q); j < seq_k; ++j)
                {
                    causal_mask(i, j) = true;
                }
            }
            scores = scores.MaskedFill(causal_mask.To(scores.GetDevice()), std::numeric_limits<T>::lowest());
        }

        if (mask.Size() > 0)
        {
            scores = scores.MaskedFill(mask, std::numeric_limits<T>::lowest());
        }

        Tensor<T> attention_weights = Softmax<T>(-1).Forward(scores);
        Tensor<T> context = MatMul(attention_weights, V);
        return context;
    }
}