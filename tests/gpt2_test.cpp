#include <gtest/gtest.h>

#include <Models/GPT2.h>
#include <Models/ModelConfigs.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using llm::GPT2;
using math::Tensor;
using transformer::TransformerConfig;

namespace {

TransformerConfig TinyConfig() {
    TransformerConfig c = models::gpt2::Small();
    c.vocab_size = 11;
    c.d_model = 8;
    c.num_layers = 2;
    c.num_heads = 2;
    c.ffn_dim = 32;
    c.max_seq_len = 6;
    return c;
}

void Seed(nn::Module<float>& m) {
    int k = 0;
    for (const auto& p : m.NamedParameters()) {
        float* data = p.tensor->Data();
        for (size_t i = 0; i < p.tensor->Size(); ++i)
            data[i] = 0.3f * static_cast<float>(std::sin(0.37 * ++k));
    }
}

Tensor<int64_t> Ids(size_t batch, size_t seq) {
    Tensor<int64_t> t({batch, seq});
    for (size_t b = 0; b < batch; ++b)
        for (size_t s = 0; s < seq; ++s) t(b, s) = static_cast<int64_t>((b * 3 + s * 2) % 11);
    return t;
}

}  // namespace

TEST(GPT2Construction, SmallPresetPassesValidation) {
    EXPECT_NO_THROW(models::gpt2::Small().Validate());
}

TEST(GPT2Construction, RejectsAnInvalidConfig) {
    TransformerConfig c = TinyConfig();
    c.vocab_size = 0;
    EXPECT_THROW(GPT2<float>{c}, std::invalid_argument);
}

TEST(GPT2Construction, BuildsFromTheConfig) {
    GPT2<float> model(TinyConfig());

    EXPECT_EQ(model.config.d_model, 8u);
    EXPECT_EQ(model.wte->weight->Shape(), (std::vector<size_t>{11, 8}));
    ASSERT_TRUE(model.wpe);
    EXPECT_EQ(model.wpe->weight->Shape(), (std::vector<size_t>{6, 8}));
    EXPECT_EQ(model.transformer->blocks.size(), 2u);
    EXPECT_FALSE(model.lm_head) << "GPT-2 ties its output to wte";
}

TEST(GPT2Construction, UntiedBuildsABiasFreeHead) {
    TransformerConfig c = TinyConfig();
    c.tie_word_embeddings = false;
    GPT2<float> model(c);

    ASSERT_TRUE(model.lm_head);
    EXPECT_EQ(model.lm_head->weight->Shape(), (std::vector<size_t>{11, 8}));
    EXPECT_FALSE(model.lm_head->bias);
}

TEST(GPT2Forward, GivesLogitsForEveryPosition) {
    GPT2<float> model(TinyConfig());
    Seed(model);

    EXPECT_EQ(model.Forward(Ids(2, 4)).Shape(), (std::vector<size_t>{2, 4, 11}));
}

TEST(GPT2Forward, RejectsIdsThatAreNot2D) {
    GPT2<float> model(TinyConfig());
    EXPECT_THROW(model.Forward(Tensor<int64_t>({4}, 1)), std::invalid_argument);
}

TEST(GPT2Forward, RejectsSequencesPastMaxSeqLen) {
    GPT2<float> model(TinyConfig());
    EXPECT_NO_THROW(model.Forward(Ids(1, 6)));
    EXPECT_THROW(model.Forward(Ids(1, 7)), std::invalid_argument);
    EXPECT_THROW(model.Forward(Ids(1, 2), 5), std::invalid_argument);
}

TEST(GPT2Forward, StartPosShiftsThePositions) {
    GPT2<float> model(TinyConfig());
    Seed(model);

    const Tensor<float> a = model.Forward(Ids(1, 2), 0);
    const Tensor<float> b = model.Forward(Ids(1, 2), 3);

    float largest = 0.0f;
    for (size_t i = 0; i < a.Size(); ++i) largest = std::max(largest, std::fabs(a.Data()[i] - b.Data()[i]));
    EXPECT_GT(largest, 1e-4f);
}

TEST(GPT2Parameters, SmallMatchesTheReleasedModelSize) {
    GPT2<float> model(models::gpt2::Small());

    size_t total = 0;
    for (const auto& p : model.NamedParameters()) total += p.tensor->Size();
    EXPECT_EQ(total, 124'439'808u);
}
