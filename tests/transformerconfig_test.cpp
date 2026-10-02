#include <gtest/gtest.h>

#include <Models/ModelConfigs.h>
#include <NeuralNet/Transformer/TransformerConfig.h>

#include <stdexcept>

using nn::transformer::config::TransformerConfig;

TEST(TransformerConfig, DefaultsAreNeutralAndFailValidation) {
    const TransformerConfig c;

    EXPECT_EQ(c.vocab_size, 0u);
    EXPECT_EQ(c.d_model, 0u);
    EXPECT_THROW(c.Validate(), std::invalid_argument);
}

TEST(TransformerConfig, ValidateRejectsHeadsThatDoNotGroup) {
    TransformerConfig c = models::gpt2::Small();
    c.num_kv_heads = 5;   // 12 query heads cannot split over 5 KV heads
    EXPECT_THROW(c.Validate(), std::invalid_argument);
}

TEST(TransformerConfig, DerivedHeadCounts) {
    TransformerConfig c = models::gpt2::Small();
    EXPECT_EQ(c.KVHeads(), 12u);
    EXPECT_EQ(c.HeadDim(), 64u);

    c.num_kv_heads = 4;
    c.head_dim = 32;
    EXPECT_EQ(c.KVHeads(), 4u);
    EXPECT_EQ(c.HeadDim(), 32u);
}

TEST(TransformerConfig, IsEosChecksEveryStopToken) {
    const TransformerConfig llama = models::llama::Llama32_1B();

    EXPECT_TRUE(llama.IsEos(128001));
    EXPECT_TRUE(llama.IsEos(128008));
    EXPECT_TRUE(llama.IsEos(128009));
    EXPECT_FALSE(llama.IsEos(128000));
    EXPECT_FALSE(llama.IsEos(0));
}

TEST(TransformerConfig, NoStopTokensMeansNothingIsEos) {
    const TransformerConfig c;

    EXPECT_TRUE(c.eos_token_ids.empty());
    EXPECT_FALSE(c.IsEos(0));
    EXPECT_FALSE(c.IsEos(50256));
}
