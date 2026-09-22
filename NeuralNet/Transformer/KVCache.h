#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

#include "Math/Tensor.h"

namespace transformer{

    using math::Tensor;

    template <typename T>
    struct KVLayer{
        Tensor<T> k;
        Tensor<T> v;
    };


    template <typename T>
    class KVCache{
    public:

        KVCache(size_t layers, size_t batch, size_t kv_heads, size_t max_seq, size_t head_dim)
        : m_maxSeq(max_seq)
        {
            if (layers == 0 || batch == 0 || kv_heads == 0 || max_seq == 0 || head_dim == 0) {
                throw std::invalid_argument("KVCache: every dimension must be greater than zero.");
            }
            m_KVLayers.reserve(layers);
            for (size_t i = 0; i < layers; ++i) {
                m_KVLayers.push_back({ Tensor<T>({batch, kv_heads, max_seq, head_dim}),
                                     Tensor<T>({batch, kv_heads, max_seq, head_dim}) });
            }
        }

        KVLayer<T>& Layer(size_t idx){
            if(idx >= m_KVLayers.size()) throw std::out_of_range("KVCache: layer index out of range");
            return m_KVLayers[idx];
        }

        const KVLayer<T>& Layer(size_t idx) const{
            if(idx >= m_KVLayers.size()) throw std::out_of_range("KVCache: layer index out of range");
            return m_KVLayers[idx];
        }

        size_t Room(size_t seq) const{
            if(seq > m_maxSeq - m_length)
                throw std::out_of_range("KVCache: sequence would exceed max seq_len");

            return m_length;
        }

        void Advance(size_t seq){
            m_length = Room(seq) + seq;
        }

        void Reset() noexcept { m_length = 0; }

        size_t Length()    const noexcept { return m_length; }
        size_t MaxLength() const noexcept { return m_maxSeq; }
        size_t Layers()    const noexcept { return m_KVLayers.size(); }
    private:
        std::vector<KVLayer<T>> m_KVLayers;
        size_t m_maxSeq;
        size_t m_length = 0;
    };
}
