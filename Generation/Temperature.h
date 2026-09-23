#pragma once

#include <cstddef>
#include <stdexcept>

#include "LogitsProcessor.h"

namespace generation{
    template <typename T>
    class Temperature: public LogitsProcessor<T>
    {
    public:
        explicit Temperature(T temp):m_temp(temp){
            if(!(temp > T(0))){
                throw std::invalid_argument("Temperature: temperature must be greater than zero.");
            }
        }

        void Process(std::span<const size_t>, Tensor<T> &scores)override {
            const size_t vocab = this->ValidatedVocab(scores);

            for(size_t v = 0; v < vocab; ++v){
                scores(v) /= m_temp;
            }
        }

    private:
        T m_temp;
    };
}
