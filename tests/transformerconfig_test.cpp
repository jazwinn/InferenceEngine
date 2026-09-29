#include <gtest/gtest.h>

#include <Models/ModelConfigs.h>
#include <NeuralNet/Transformer/TransformerConfig.h>
#include <NeuralNet/Transformer/RoPE.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using nn::transformer::config::Activation;
using nn::transformer::config::MLPType;
using nn::transformer::config::NormPlacement;
using nn::transformer::config::NormType;
using nn::transformer::config::PositionEncoding;
using nn::transformer::config::RopeScalingType;
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
    EXPECT_EQ(c.eos_token_ids, (std::vector<int64_t>{50256}));
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

TEST(TransformerConfig, Llama32_1BPreset) {
    const TransformerConfig c = models::llama::Llama32_1B();

    EXPECT_EQ(c.vocab_size, 128256u);
    EXPECT_EQ(c.d_model, 2048u);
    EXPECT_EQ(c.num_layers, 16u);
    EXPECT_EQ(c.num_heads, 32u);
    EXPECT_EQ(c.KVHeads(), 8u);
    EXPECT_EQ(c.HeadDim(), 64u);
    EXPECT_EQ(c.ffn_dim, 8192u);
    EXPECT_EQ(c.max_seq_len, 131072u);
    EXPECT_NO_THROW(c.Validate());
}

TEST(TransformerConfig, Llama32_1BArchitecture) {
    const TransformerConfig c = models::llama::Llama32_1B();

    EXPECT_EQ(c.norm_type, NormType::RMSNorm);
    EXPECT_EQ(c.norm_placement, NormPlacement::Pre);
    EXPECT_FLOAT_EQ(c.norm_eps, 1e-5f);
    EXPECT_EQ(c.mlp_type, MLPType::Gated);
    EXPECT_EQ(c.activation, Activation::SiLU);
    EXPECT_FALSE(c.attention_bias);
    EXPECT_FALSE(c.output_bias);
    EXPECT_TRUE(c.tie_word_embeddings);
    EXPECT_TRUE(c.causal);
}

TEST(TransformerConfig, Llama32_1BUsesLlama3RopeScaling) {
    const TransformerConfig c = models::llama::Llama32_1B();

    EXPECT_EQ(c.position, PositionEncoding::Rotary);
    EXPECT_FLOAT_EQ(c.rope_theta, 500000.0f);
    EXPECT_EQ(c.rope_scaling.type, RopeScalingType::Llama3);
    EXPECT_DOUBLE_EQ(c.rope_scaling.factor, 32.0);
    EXPECT_DOUBLE_EQ(c.rope_scaling.low_freq_factor, 1.0);
    EXPECT_DOUBLE_EQ(c.rope_scaling.high_freq_factor, 4.0);
    EXPECT_EQ(c.rope_scaling.original_max_position, 8192u);
}

TEST(TransformerConfig, Llama32_1BSpecialTokens) {
    const TransformerConfig c = models::llama::Llama32_1B();

    EXPECT_EQ(c.bos_token_id, 128000);
    EXPECT_EQ(c.eos_token_ids, (std::vector<int64_t>{128001, 128008, 128009}));
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

TEST(TransformerConfig, Llama32_1BRopeSettingsBuildValidTables) {
    const TransformerConfig c = models::llama::Llama32_1B();

    EXPECT_NO_THROW(nn::transformer::RotaryEmbedding<float>(c.HeadDim(), 16, c.rope_theta, c.rope_scaling));
}
