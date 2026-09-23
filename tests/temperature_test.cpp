#include <gtest/gtest.h>

#include <Generation/LogitsProcessor.h>
#include <Generation/Sampler.h>
#include <Generation/Temperature.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

using generation::Greedy;
using generation::LogitsProcessor;
using generation::ProcessorList;
using generation::Temperature;
using math::Tensor;

namespace {
Tensor<float> Scores(std::initializer_list<float> values) {
    Tensor<float> t({values.size()});
    size_t i = 0;
    for (float v : values) t(i++) = v;
    return t;
}

double Probability(const Tensor<float>& scores, size_t index) {
    double max_score = scores(0);
    for (size_t v = 1; v < scores.Shape()[0]; ++v) max_score = std::max(max_score, double(scores(v)));

    double sum = 0.0;
    for (size_t v = 0; v < scores.Shape()[0]; ++v) sum += std::exp(double(scores(v)) - max_score);
    return std::exp(double(scores(index)) - max_score) / sum;
}

}

TEST(TemperatureProcessor, DividesEveryLogit) {
    Temperature<float> temperature(2.0f);
    Tensor<float> scores = Scores({2.0f, 4.0f, -6.0f});

    temperature.Process({}, scores);

    EXPECT_FLOAT_EQ(scores(0), 1.0f);
    EXPECT_FLOAT_EQ(scores(1), 2.0f);
    EXPECT_FLOAT_EQ(scores(2), -3.0f) << "negative logits divide like any other";
}

TEST(TemperatureProcessor, OneIsTheIdentity) {
    Temperature<float> temperature(1.0f);
    Tensor<float> scores = Scores({0.25f, -1.5f, 3.0f});

    temperature.Process({}, scores);

    EXPECT_FLOAT_EQ(scores(0), 0.25f);
    EXPECT_FLOAT_EQ(scores(1), -1.5f);
    EXPECT_FLOAT_EQ(scores(2), 3.0f);
}

TEST(TemperatureProcessor, LeavesTheLogitCountAlone) {
    Temperature<float> temperature(0.5f);
    Tensor<float> scores = Scores({1.0f, 2.0f, 3.0f});

    temperature.Process({}, scores);

    EXPECT_EQ(scores.Shape(), (std::vector<size_t>{3})) << "a processor transforms, it does not truncate";
}

TEST(TemperatureProcessor, WritesInPlace) {
    Temperature<float> temperature(2.0f);
    Tensor<float> scores = Scores({8.0f});
    const float* before = scores.Data();

    temperature.Process({}, scores);

    EXPECT_EQ(scores.Data(), before) << "Process must not reallocate the caller's tensor";
    EXPECT_FLOAT_EQ(scores(0), 4.0f);
}

TEST(TemperatureProcessor, BelowOneSharpensTheDistribution) {
    Tensor<float> scores = Scores({1.0f, 2.0f, 3.0f});
    const double before = Probability(scores, 2);

    Temperature<float>(0.5f).Process({}, scores);

    EXPECT_GT(Probability(scores, 2), before) << "T < 1 concentrates mass on the leader";
}

TEST(TemperatureProcessor, AboveOneFlattensTheDistribution) {
    Tensor<float> scores = Scores({1.0f, 2.0f, 3.0f});
    const double before = Probability(scores, 2);

    Temperature<float>(2.0f).Process({}, scores);

    EXPECT_LT(Probability(scores, 2), before) << "T > 1 spreads mass toward the tail";
}

TEST(TemperatureProcessor, PreservesTheRanking) {
    Tensor<float> scores = Scores({0.1f, 0.9f, 0.4f, -2.0f});
    Greedy<float> greedy;
    const size_t before = greedy.Pick(scores);

    Temperature<float>(0.25f).Process({}, scores);
    EXPECT_EQ(greedy.Pick(scores), before) << "temperature only matters once you sample";

    Temperature<float>(9.0f).Process({}, scores);
    EXPECT_EQ(greedy.Pick(scores), before);
}

TEST(TemperatureProcessor, RejectsZero) {
    EXPECT_THROW(Temperature<float>(0.0f), std::invalid_argument);
}

TEST(TemperatureProcessor, RejectsNegativeTemperatures) {
    EXPECT_THROW(Temperature<float>(-1.0f), std::invalid_argument);
    EXPECT_THROW(Temperature<float>(-0.0001f), std::invalid_argument);
}

TEST(TemperatureProcessor, RejectsNaN) {
    EXPECT_THROW((Temperature<float>(std::numeric_limits<float>::quiet_NaN())), std::invalid_argument)
        << "the guard is written !(temp > 0) precisely so NaN is caught";
}

TEST(TemperatureProcessor, AcceptsSmallAndLargeTemperatures) {
    EXPECT_NO_THROW(Temperature<float>(0.01f));
    EXPECT_NO_THROW(Temperature<float>(100.0f));
}

TEST(TemperatureProcessor, RejectsScoresThatAreNotARow) {
    Temperature<float> temperature(2.0f);
    Tensor<float> batched({1, 4}, 1.0f);
    Tensor<float> full({1, 2, 4}, 1.0f);

    EXPECT_THROW(temperature.Process({}, batched), std::invalid_argument);
    EXPECT_THROW(temperature.Process({}, full), std::invalid_argument)
        << "a processor must not be handed [batch, seq, vocab]";
}

TEST(TemperatureProcessor, WritesThroughAViewIntoTheLogits) {
    Tensor<float> logits({1, 3, 4});
    for (size_t s = 0; s < 3; ++s)
        for (size_t v = 0; v < 4; ++v) logits(0, s, v) = static_cast<float>(10 * s + v);

    Tensor<float> row = logits.Row(0).Row(1);
    ASSERT_EQ(row.Shape(), (std::vector<size_t>{4}));
    ASSERT_TRUE(row.SharesStorage(logits));

    Temperature<float>(2.0f).Process({}, row);

    for (size_t v = 0; v < 4; ++v)
        EXPECT_FLOAT_EQ(logits(0, 1, v), static_cast<float>(10 + v) / 2.0f) << "v" << v;

    EXPECT_FLOAT_EQ(logits(0, 0, 3), 3.0f) << "earlier positions must be untouched";
    EXPECT_FLOAT_EQ(logits(0, 2, 0), 20.0f) << "later positions must be untouched";
}

TEST(TemperatureProcessor, IgnoresHistory) {
    Temperature<float> temperature(2.0f);
    const std::vector<size_t> history{5, 5, 5};

    Tensor<float> with_history = Scores({4.0f, 6.0f});
    Tensor<float> without = Scores({4.0f, 6.0f});
    temperature.Process(history, with_history);
    temperature.Process({}, without);

    EXPECT_FLOAT_EQ(with_history(0), without(0)) << "penalties use history; temperature does not";
    EXPECT_FLOAT_EQ(with_history(1), without(1));
}

TEST(TemperatureProcessor, WorksThroughTheBaseInterface) {
    std::unique_ptr<LogitsProcessor<float>> processor = std::make_unique<Temperature<float>>(4.0f);
    Tensor<float> scores = Scores({8.0f});

    processor->Process({}, scores);

    EXPECT_FLOAT_EQ(scores(0), 2.0f);
}

TEST(TemperatureProcessor, ComposesInAProcessorList) {
    ProcessorList<float> list;
    list.Add(std::make_shared<Temperature<float>>(2.0f));
    list.Add(std::make_shared<Temperature<float>>(5.0f));

    Tensor<float> scores = Scores({20.0f});
    list.Process({}, scores);

    EXPECT_FLOAT_EQ(scores(0), 2.0f) << "two divisions compose into one by 10";
}

TEST(TemperatureProcessor, WorksForDouble) {
    Temperature<double> temperature(4.0);
    Tensor<double> scores({1});
    scores(0) = 1.0;

    temperature.Process({}, scores);

    EXPECT_DOUBLE_EQ(scores(0), 0.25);
}
