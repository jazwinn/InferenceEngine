#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Math/Tensor.h"

namespace generation
{
    using math::Tensor;

    template <typename T>
    class LogitsProcessor
    {
    public:
        virtual ~LogitsProcessor() = default;

        virtual void Process(std::span<const size_t> history, Tensor<T> &scores) = 0;

        void operator()(std::span<const size_t> history, Tensor<T> &scores)
        {
            Process(history, scores);
        }

    protected:
        static size_t ValidatedVocab(const Tensor<T> &scores)
        {
            if (scores.Rank() != 1)
            {
                throw std::invalid_argument("LogitsProcessor: scores must be a [vocab] tensor.");
            }
            return scores.Shape()[0];
        }
    };

    template <typename T>
    class ProcessorList final : public LogitsProcessor<T>
    {
    public:
        ProcessorList() = default;

        ProcessorList(std::initializer_list<std::shared_ptr<LogitsProcessor<T>>> processors)
        {
            for (const auto &processor : processors)
            {
                Add(processor);
            }
        }

        void Add(std::shared_ptr<LogitsProcessor<T>> processor)
        {
            if (!processor)
            {
                throw std::invalid_argument("ProcessorList: processor must not be null.");
            }
            m_processors.push_back(std::move(processor));
        }

        void Process(std::span<const size_t> history, Tensor<T> &scores) override
        {
            for (const auto &processor : m_processors)
            {
                processor->Process(history, scores);
            }
        }

        size_t Size() const noexcept { return m_processors.size(); }
        bool Empty() const noexcept { return m_processors.empty(); }

    private:
        std::vector<std::shared_ptr<LogitsProcessor<T>>> m_processors;
    };
}
