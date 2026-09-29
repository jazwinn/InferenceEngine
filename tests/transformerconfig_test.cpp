#include <gtest/gtest.h>

#include <Models/ModelConfigs.h>
#include <NeuralNet/Transformer/TransformerConfig.h>

#include <stdexcept>

using nn::transformer::config::TransformerConfig;

TEST(TransformerConfig, GPT2SmallPreset) {
    const TransformerConfig c = models::gpt2::Small();

    EXPECT_EQ(c.vocab_size, 50257u);
    EXPECT_EQ(c.max_seq_len, 1024u);
    EXPECT_EQ(c.d_model, 768u);
    EXPECT_EQ(c.num_layers, 12u);
    EXPECT_EQ(c.num_heads, 12u);
    EXPECT_EQ(c.ffn_dim, 4 * c.d_model);
    EXPECT_FLOAT_EQ(c.norm_eps, 1e-5f);
    EXPECT_EQ(c.eos_token_id, 50256);
    EXPECT_EQ(c.d_model % c.num_heads, 0u);
    EXPECT_NO_THROW(c.Validate());
}

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
