#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "Math/Tensor.h"
#include "NeuralNet/Transformer/TransformerConfig.h"

namespace nn::transformer{

    using math::Tensor;
    using nn::transformer::config::RopeScaling;
    using nn::transformer::config::RopeScalingType;

    template <typename T = float>
    class RotaryEmbedding{
    public:
        RotaryEmbedding(size_t head_dim, size_t max_seq, double theta = 10000.0, RopeScaling scaling = {}):
            headDim(ValidatedHeadDim(head_dim)),
            seq(max_seq > 0 ? max_seq : throw std::invalid_argument("RotaryEmbedding: max_seq must be greater than zero.")),
            cos({seq, headDim / 2}),
            sin({seq, headDim / 2})
        {
            if (!(theta > 0.0))
            {
                throw std::invalid_argument("RotaryEmbedding: theta must be greater than zero.");
            }
            ValidateScaling(scaling);

            const size_t half = headDim / 2;
            T* cosData = cos.Data();
            T* sinData = sin.Data();
            for (size_t i = 0; i < half; ++i)
            {
                const double freq = Scaled(std::pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(headDim)), scaling);
                for (size_t p = 0; p < seq; ++p)
                {
                    const double angle = static_cast<double>(p) * freq;
                    cosData[p * half + i] = static_cast<T>(std::cos(angle));
                    sinData[p * half + i] = static_cast<T>(std::sin(angle));
                }
            }
        }

        Tensor<T> Apply(const Tensor<T>& x, size_t start_pos) const
        {
            if (x.Rank() != 4)
            {
                throw std::invalid_argument("RotaryEmbedding: input must have shape (batch, heads, seq, head_dim).");
            }
            const std::vector<size_t>& shape = x.Shape();
            if (shape[3] != headDim)
            {
                throw std::invalid_argument("RotaryEmbedding: input head_dim does not match.");
            }
            const size_t length = shape[2];
            if (start_pos > seq || length > seq - start_pos)
            {
                throw std::out_of_range("RotaryEmbedding: positions exceed max_seq.");
            }

            const Tensor<T> src = x.Contiguous();
            Tensor<T> out(shape);

            const size_t half = headDim / 2;
            const size_t rows = shape[0] * shape[1];
            const T* in = src.Data();
            T* dst = out.Data();
            const T* cosData = cos.Data();
            const T* sinData = sin.Data();

            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t s = 0; s < length; ++s)
                {
                    const size_t base = (r * length + s) * headDim;
                    const T* c = cosData + (start_pos + s) * half;
                    const T* sn = sinData + (start_pos + s) * half;
                    for (size_t i = 0; i < half; ++i)
                    {
                        const T x1 = in[base + i];
                        const T x2 = in[base + i + half];
                        dst[base + i] = x1 * c[i] - x2 * sn[i];
                        dst[base + i + half] = x2 * c[i] + x1 * sn[i];
                    }
                }
            }
            return out;
        }

        size_t headDim;
        size_t seq;
        Tensor<T> cos;
        Tensor<T> sin;

    private:
        static size_t ValidatedHeadDim(size_t head_dim)
        {
            if (head_dim == 0 || head_dim % 2 != 0)
            {
                throw std::invalid_argument("RotaryEmbedding: head_dim must be a positive even number.");
            }
            return head_dim;
        }

        static void ValidateScaling(const RopeScaling& s)
        {
            switch (s.type)
            {
            case RopeScalingType::Default:
                return;
            case RopeScalingType::Linear:
                if (!(s.factor > 0.0))
                {
                    throw std::invalid_argument("RotaryEmbedding: linear scaling needs factor > 0.");
                }
                return;
            case RopeScalingType::Llama3:
                if (!(s.factor > 0.0) || !(s.low_freq_factor > 0.0) || !(s.high_freq_factor > s.low_freq_factor) ||
                    s.original_max_position == 0)
                {
                    throw std::invalid_argument("RotaryEmbedding: llama3 scaling needs factor > 0, 0 < low_freq_factor < high_freq_factor and original_max_position > 0.");
                }
                return;
            case RopeScalingType::Dynamic:
            case RopeScalingType::Yarn:
            case RopeScalingType::LongRope:
            case RopeScalingType::MRope:
                break;
            }
            throw std::invalid_argument("RotaryEmbedding: this rope scaling type is not implemented.");
        }

        static double Scaled(double freq, const RopeScaling& s)
        {
            if (s.type == RopeScalingType::Default)
            {
                return freq;
            }
            if (s.type == RopeScalingType::Linear)
            {
                return freq / s.factor;
            }

            const double pi = 3.14159265358979323846;
            const double wavelen = 2.0 * pi / freq;
            const double oldLen = static_cast<double>(s.original_max_position);
            const double lowFreqWavelen = oldLen / s.low_freq_factor;
            const double highFreqWavelen = oldLen / s.high_freq_factor;

            if (wavelen < highFreqWavelen)
            {
                return freq;
            }
            if (wavelen > lowFreqWavelen)
            {
                return freq / s.factor;
            }

            const double smooth = (oldLen / wavelen - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor);
            return (1.0 - smooth) * freq / s.factor + smooth * freq;
        }

    };
}
