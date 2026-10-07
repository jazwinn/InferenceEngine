#pragma once

#include <cstddef>
#include <stdexcept>

#include "Tensor/Tensor.h"

namespace generation{
    using tensor::Tensor;

    template <typename T>
    class Sampler
    {
    public:
        virtual ~Sampler() = default;

        virtual size_t Pick(const Tensor<T>& scores) = 0;
    };

    template <typename T>
    class Greedy: public Sampler<T>
    {
    public:
        size_t Pick(const Tensor<T>& scores)override{
            if(scores.Rank() != 1){
                throw std::invalid_argument("Greedy: scores must be a [vocab] tensor.");
            }

            const size_t vocab = scores.Shape()[0];

            size_t best = 0;
            for(size_t n = 1; n < vocab; ++n){
                best = (scores(n) > scores(best))? n : best;
            }

            return best;
        }
    };
}
