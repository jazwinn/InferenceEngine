#pragma once

#include <algorithm>
#include <concepts>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "Math/Tensor.h"
#include "NeuralNet/Module.h"

namespace nn
{

    template <typename T = float>
    class Embedding : public Module<T>
    {
    public:
        Embedding(size_t num_embeddings, size_t embedding_dim, std::optional<long long> padding_idx = std::nullopt)
            : weight(std::make_shared<Tensor<T>>(std::vector<size_t>{num_embeddings, embedding_dim})),
              m_padding_idx(NormalizePaddingIdx(padding_idx, num_embeddings))
        {
            this->RegisterParameter("weight", weight);
        }

        static Embedding FromPretrained(Tensor<T> embeddings, std::optional<long long> padding_idx = std::nullopt)
        {
            if (embeddings.Rank() != 2)
            {
                throw std::invalid_argument("Embedding: pretrained embeddings must be a 2-D [num_embeddings, embedding_dim] tensor.");
            }

            Embedding embedding(embeddings.Shape()[0], embeddings.Shape()[1], padding_idx);
            *embedding.weight = embeddings.Contiguous();
            return embedding;
        }

        template <std::integral Index>
        requires(!std::is_same_v<Index, bool>)
        Tensor<T> Forward(const Tensor<Index> &indices) const
        {
            if (indices.Size() == 0)
            {
                throw std::invalid_argument("Embedding: indices must not be empty.");
            }

            const size_t rows = NumEmbeddings();
            const size_t dim = EmbeddingDim();

            std::vector<size_t> shape = indices.Shape();
            shape.push_back(dim);
            Tensor<T> output(shape);

            const Tensor<Index> ids = indices.Contiguous();
            const Tensor<T> table = weight->Contiguous();
            const T *source = table.Data();
            T *destination = output.Data();

            for (size_t i = 0; i < ids.Size(); ++i)
            {
                const Index id = ids.Data()[i];
                if (id < 0 || static_cast<size_t>(id) >= rows)
                {
                    throw std::out_of_range("Embedding: index is out of range for the embedding table.");
                }
                std::copy_n(source + static_cast<size_t>(id) * dim, dim, destination + i * dim);
            }

            return output;
        }

        size_t NumEmbeddings() const noexcept { return weight->Shape()[0]; }
        size_t EmbeddingDim() const noexcept { return weight->Shape()[1]; }
        std::optional<size_t> PaddingIdx() const noexcept { return m_padding_idx; }

        std::shared_ptr<Tensor<T>> weight;

    private:
        static std::optional<size_t> NormalizePaddingIdx(std::optional<long long> padding_idx, size_t num_embeddings)
        {
            if (!padding_idx)
            {
                return std::nullopt;
            }

            const long long rows = static_cast<long long>(num_embeddings);
            const long long index = *padding_idx < 0 ? *padding_idx + rows : *padding_idx;
            if (index < 0 || index >= rows)
            {
                throw std::invalid_argument("Embedding: padding_idx must be within num_embeddings.");
            }
            return static_cast<size_t>(index);
        }

        std::optional<size_t> m_padding_idx;
    };

}
