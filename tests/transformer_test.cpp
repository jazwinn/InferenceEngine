#include <gtest/gtest.h>

#include <NeuralNet/Transformer/Transformer.h>

#include <stdexcept>

using transformer::Transformer;
using transformer::TransformerConfig;

TEST(TransformerConfig, DefaultsDescribeGPT2Small) {
    const TransformerConfig c;

    EXPECT_EQ(c.vocab_size, 50257u);
    EXPECT_EQ(c.d_model, 768u);
    EXPECT_EQ(c.num_layers, 12u);
    EXPECT_EQ(c.num_heads, 12u);
    EXPECT_EQ(c.ffn_dim, 4 * c.d_model);
    EXPECT_EQ(c.d_model % c.num_heads, 0u);
}

TEST(Transformer, BuildsAttentionFromTheConfig) {
    TransformerConfig c;
    c.d_model = 64;
    c.num_heads = 4;

    const Transformer<float> model(c);

    EXPECT_EQ(model.config.d_model, 64u);
    EXPECT_EQ(model.atten.num_heads, 4u);
    EXPECT_EQ(model.atten.head_dim, 16u);
}

TEST(Transformer, RejectsHeadsThatDoNotDivideTheModelWidth) {
    TransformerConfig c;
    c.d_model = 64;
    c.num_heads = 5;

    EXPECT_THROW(Transformer<float>{c}, std::invalid_argument);
}
