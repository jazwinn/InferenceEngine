#include <gtest/gtest.h>

#include <Generation/LogitsProcessor.h>
#include <Generation/Sampler.h>

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

using generation::Greedy;
using generation::LogitsProcessor;
using generation::ProcessorList;
using generation::Sampler;
using tensor::Tensor;

namespace {
Tensor<float> Scores(std::initializer_list<float> values) {
    Tensor<float> t({values.size()});
    size_t i = 0;
    for (float v : values) t(i++) = v;
    return t;
}

template <typename T>
class AddConstant : public LogitsProcessor<T> {
public:
    explicit AddConstant(T amount) : m_amount(amount) {}
    void Process(std::span<const size_t>, Tensor<T>& scores) override {
        for (size_t v = 0; v < this->ValidatedVocab(scores); ++v) scores(v) += m_amount;
    }
private:
    T m_amount;
};

template <typename T>
class Scale : public LogitsProcessor<T> {
public:
    explicit Scale(T factor) : m_factor(factor) {}
    void Process(std::span<const size_t>, Tensor<T>& scores) override {
        for (size_t v = 0; v < this->ValidatedVocab(scores); ++v) scores(v) *= m_factor;
    }
private:
    T m_factor;
};

template <typename T>
class RecordHistory : public LogitsProcessor<T> {
public:
    void Process(std::span<const size_t> history, Tensor<T>&) override {
        seen.assign(history.begin(), history.end());
        ++calls;
    }
    std::vector<size_t> seen;
    int calls = 0;
};

}

TEST(GreedySampler, PicksTheHighestScore) {
    Greedy<float> greedy;
    EXPECT_EQ(greedy.Pick(Scores({0.1f, 0.9f, 0.4f})), 1u);
}

TEST(GreedySampler, PicksIndexZeroWhenItWins) {
    Greedy<float> greedy;
    EXPECT_EQ(greedy.Pick(Scores({5.0f, 0.9f, 0.4f})), 0u);
}

TEST(GreedySampler, PicksTheLastIndexWhenItWins) {
    Greedy<float> greedy;
    EXPECT_EQ(greedy.Pick(Scores({0.1f, 0.9f, 4.0f})), 2u);
}

TEST(GreedySampler, HandlesNegativeScores) {
    Greedy<float> greedy;
    EXPECT_EQ(greedy.Pick(Scores({-8.0f, -2.0f, -5.0f})), 1u)
        << "logits are routinely negative; the running best must not start at zero";
}

TEST(GreedySampler, KeepsTheLowestIndexOnATie) {
    Greedy<float> greedy;
    EXPECT_EQ(greedy.Pick(Scores({1.0f, 3.0f, 3.0f, 2.0f})), 1u)
        << "ties must resolve the same way every run";
}

TEST(GreedySampler, HandlesASingleTokenVocabulary) {
    Greedy<float> greedy;
    EXPECT_EQ(greedy.Pick(Scores({0.0f})), 0u);
}

TEST(GreedySampler, RejectsScoresThatAreNotARow) {
    Greedy<float> greedy;
    EXPECT_THROW(greedy.Pick(Tensor<float>({1, 4})), std::invalid_argument);
    EXPECT_THROW(greedy.Pick(Tensor<float>({1, 2, 4})), std::invalid_argument)
        << "a sampler must not be handed [batch, seq, vocab]";
    EXPECT_THROW(greedy.Pick(Tensor<float>()), std::invalid_argument);
}

TEST(GreedySampler, WorksThroughTheBaseInterface) {
    std::unique_ptr<Sampler<float>> sampler = std::make_unique<Greedy<float>>();
    EXPECT_EQ(sampler->Pick(Scores({0.1f, 0.9f, 0.4f})), 1u);
}

TEST(GreedySampler, WorksForDouble) {
    Greedy<double> greedy;

    EXPECT_EQ(greedy.Pick([] {
        Tensor<double> t({2});
        t(0) = 1.000000001;
        t(1) = 1.000000002;
        return t;
    }()), 1u);
}

TEST(ProcessorList, EmptyListIsTheIdentity) {
    ProcessorList<float> list;
    Tensor<float> scores = Scores({1.0f, 2.0f, 3.0f});

    list.Process({}, scores);

    EXPECT_TRUE(list.Empty());
    EXPECT_FLOAT_EQ(scores(0), 1.0f);
    EXPECT_FLOAT_EQ(scores(2), 3.0f);
}

TEST(ProcessorList, AppliesEveryProcessor) {
    ProcessorList<float> list;
    list.Add(std::make_shared<AddConstant<float>>(1.0f));
    list.Add(std::make_shared<AddConstant<float>>(10.0f));

    Tensor<float> scores = Scores({0.0f, 5.0f});
    list.Process({}, scores);

    EXPECT_EQ(list.Size(), 2u);
    EXPECT_FLOAT_EQ(scores(0), 11.0f);
    EXPECT_FLOAT_EQ(scores(1), 16.0f);
}

TEST(ProcessorList, AppliesThemInOrder) {
    ProcessorList<float> add_then_scale;
    add_then_scale.Add(std::make_shared<AddConstant<float>>(1.0f));
    add_then_scale.Add(std::make_shared<Scale<float>>(2.0f));

    ProcessorList<float> scale_then_add;
    scale_then_add.Add(std::make_shared<Scale<float>>(2.0f));
    scale_then_add.Add(std::make_shared<AddConstant<float>>(1.0f));

    Tensor<float> a = Scores({3.0f});
    Tensor<float> b = Scores({3.0f});
    add_then_scale.Process({}, a);
    scale_then_add.Process({}, b);

    EXPECT_FLOAT_EQ(a(0), 8.0f);
    EXPECT_FLOAT_EQ(b(0), 7.0f) << "order is the semantics, not an implementation detail";
}

TEST(ProcessorList, AcceptsAnInitialiserList) {
    ProcessorList<float> list{std::make_shared<AddConstant<float>>(2.0f),
                              std::make_shared<Scale<float>>(3.0f)};

    Tensor<float> scores = Scores({1.0f});
    list.Process({}, scores);

    EXPECT_EQ(list.Size(), 2u);
    EXPECT_FLOAT_EQ(scores(0), 9.0f);
}

TEST(ProcessorList, RejectsANullProcessor) {
    ProcessorList<float> list;
    EXPECT_THROW(list.Add(nullptr), std::invalid_argument);
    EXPECT_TRUE(list.Empty()) << "a rejected Add must not partially apply";
}

TEST(ProcessorList, PassesHistoryThrough) {
    auto recorder = std::make_shared<RecordHistory<float>>();
    ProcessorList<float> list;
    list.Add(recorder);

    const std::vector<size_t> history{7, 8, 9};
    Tensor<float> scores = Scores({1.0f});
    list.Process(history, scores);

    EXPECT_EQ(recorder->calls, 1);
    EXPECT_EQ(recorder->seen, history) << "penalties need the tokens generated so far";
}

TEST(ProcessorList, IsItselfAProcessorSoListsNest) {
    auto inner = std::make_shared<ProcessorList<float>>();
    inner->Add(std::make_shared<AddConstant<float>>(1.0f));
    inner->Add(std::make_shared<AddConstant<float>>(2.0f));

    ProcessorList<float> outer;
    outer.Add(inner);
    outer.Add(std::make_shared<Scale<float>>(10.0f));

    Tensor<float> scores = Scores({0.0f});
    outer.Process({}, scores);

    EXPECT_FLOAT_EQ(scores(0), 30.0f);
}

TEST(ProcessorList, IsUsableThroughTheBaseInterface) {
    std::unique_ptr<LogitsProcessor<float>> processor = [] {
        auto list = std::make_unique<ProcessorList<float>>();
        list->Add(std::make_shared<AddConstant<float>>(4.0f));
        return list;
    }();

    Tensor<float> scores = Scores({1.0f});
    processor->Process({}, scores);

    EXPECT_FLOAT_EQ(scores(0), 5.0f)
        << "the call site must not need to know it holds a list";
}

TEST(Decoding, ProcessorsRunThenTheSamplerPicks) {
    class BoostLast : public LogitsProcessor<float> {
    public:
        void Process(std::span<const size_t>, Tensor<float>& scores) override {
            scores(scores.Shape()[0] - 1) += 100.0f;
        }
    };

    ProcessorList<float> list;
    list.Add(std::make_shared<BoostLast>());

    Tensor<float> scores = Scores({0.1f, 0.9f, 0.4f});
    Greedy<float> greedy;

    EXPECT_EQ(greedy.Pick(scores), 1u) << "before processing";
    list.Process({}, scores);
    EXPECT_EQ(greedy.Pick(scores), 2u) << "after processing";
}
