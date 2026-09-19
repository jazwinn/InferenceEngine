#include <gtest/gtest.h>

#include <Reflection/Reflection.h>
#include <Serialize/Serialize.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class Activation : int { ReLU = 0, GELU = 1, SiLU = 2 };

struct Layer {
    std::string name;
    size_t units = 0;

    REFLECTABLE(Layer, name, units)
};

struct Config {
    float learning_rate = 0.0f;
    double epsilon = 0.0;
    int seed = 0;
    size_t d_model = 0;
    bool causal = false;
    std::string label;
    Activation activation = Activation::ReLU;
    std::vector<int> dims;
    Layer head;
    std::vector<Layer> layers;

    REFLECTABLE(Config, learning_rate, epsilon, seed, d_model, causal, label, activation, dims, head, layers)
};

// A document with one Config entry holding every supported member type.
nlohmann::json FullDocument() {
    return nlohmann::json::parse(R"([
        { "Config": {
            "learning_rate": 0.5,
            "epsilon": 1e-5,
            "seed": -7,
            "d_model": 512,
            "causal": true,
            "label": "tiny",
            "activation": 2,
            "dims": [1, 2, 3],
            "head": { "name": "out", "units": 10 },
            "layers": [ { "name": "a", "units": 4 }, { "name": "b", "units": 8 } ]
        } }
    ])");
}

// Writes text to a uniquely named file in the temp directory, removed on destruction.
class TempFile {
public:
    explicit TempFile(const std::string& text)
        : m_path(std::filesystem::temp_directory_path() /
                 ("serialize_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".json")) {
        std::ofstream(m_path) << text;
    }
    ~TempFile() { std::filesystem::remove(m_path); }

    std::string Path() const { return m_path.string(); }

private:
    std::filesystem::path m_path;
};

}  // namespace

// --- reflection -------------------------------------------------------------

TEST(Reflection, NamesFollowDeclarationOrder) {
    EXPECT_EQ(Layer::Names(), (std::array<std::string, 2>{"name", "units"}));
    EXPECT_STREQ(Layer::classname(), "Layer");
    EXPECT_EQ(Config().membercount(), 10);
}

// --- loading each member type -----------------------------------------------

TEST(SerializeLoad, ReadsScalars) {
    const Config c = serialize::LoadJson<Config>(FullDocument());

    EXPECT_FLOAT_EQ(c.learning_rate, 0.5f);
    EXPECT_DOUBLE_EQ(c.epsilon, 1e-5);
    EXPECT_EQ(c.seed, -7);
    EXPECT_EQ(c.d_model, 512u);
    EXPECT_TRUE(c.causal);
    EXPECT_EQ(c.label, "tiny");
}

TEST(SerializeLoad, ReadsEnumsFromTheirIntegerValue) {
    EXPECT_EQ(serialize::LoadJson<Config>(FullDocument()).activation, Activation::SiLU);
}

TEST(SerializeLoad, ReadsVectorsOfScalars) {
    EXPECT_EQ(serialize::LoadJson<Config>(FullDocument()).dims, (std::vector<int>{1, 2, 3}));
}

TEST(SerializeLoad, ReadsNestedReflectableObjects) {
    const Config c = serialize::LoadJson<Config>(FullDocument());

    EXPECT_EQ(c.head.name, "out");
    EXPECT_EQ(c.head.units, 10u);
}

TEST(SerializeLoad, ReadsVectorsOfReflectableObjects) {
    const Config c = serialize::LoadJson<Config>(FullDocument());

    ASSERT_EQ(c.layers.size(), 2u);
    EXPECT_EQ(c.layers[0].name, "a");
    EXPECT_EQ(c.layers[0].units, 4u);
    EXPECT_EQ(c.layers[1].name, "b");
    EXPECT_EQ(c.layers[1].units, 8u);
}

TEST(SerializeLoad, AcceptsAnIntegerForAFloatingPointMember) {
    const auto doc = nlohmann::json::parse(R"([{ "Config": { "learning_rate": 2 } }])");
    EXPECT_FLOAT_EQ(serialize::LoadJson<Config>(doc).learning_rate, 2.0f);
}

// --- missing and mistyped values --------------------------------------------

TEST(SerializeLoad, MissingKeysKeepTheirDefaults) {
    const auto doc = nlohmann::json::parse(R"([{ "Config": { "seed": 3 } }])");
    const Config c = serialize::LoadJson<Config>(doc);

    EXPECT_EQ(c.seed, 3);
    EXPECT_EQ(c.d_model, 0u);
    EXPECT_FALSE(c.causal);
    EXPECT_TRUE(c.dims.empty());
    EXPECT_TRUE(c.head.name.empty());
}

TEST(SerializeLoad, WrongTypesAreIgnored) {
    const auto doc = nlohmann::json::parse(R"([{ "Config": {
        "learning_rate": "fast", "seed": 1.5, "causal": 1, "label": 4,
        "dims": { "not": "an array" }, "head": [1, 2]
    } }])");
    const Config c = serialize::LoadJson<Config>(doc);

    EXPECT_FLOAT_EQ(c.learning_rate, 0.0f);
    EXPECT_EQ(c.seed, 0);
    EXPECT_FALSE(c.causal);
    EXPECT_TRUE(c.label.empty());
    EXPECT_TRUE(c.dims.empty());
    EXPECT_TRUE(c.head.name.empty());
}

TEST(SerializeLoad, NegativeValuesAreRejectedForUnsignedMembers) {
    const auto doc = nlohmann::json::parse(R"([{ "Config": { "d_model": -1 } }])");
    EXPECT_EQ(serialize::LoadJson<Config>(doc).d_model, 0u);
}

// --- document shape ---------------------------------------------------------

TEST(SerializeLoad, EntriesForOtherClassesAreSkipped) {
    const auto doc = nlohmann::json::parse(R"([
        { "Other": { "seed": 99 } },
        { "Config": { "seed": 5 } }
    ])");
    EXPECT_EQ(serialize::LoadJson<Config>(doc).seed, 5);
}

TEST(SerializeLoad, LaterEntriesOverrideEarlierOnes) {
    const auto doc = nlohmann::json::parse(R"([
        { "Config": { "seed": 1, "label": "first" } },
        { "Config": { "seed": 2 } }
    ])");
    const Config c = serialize::LoadJson<Config>(doc);

    EXPECT_EQ(c.seed, 2);
    EXPECT_EQ(c.label, "first");
}

TEST(SerializeLoad, APlainObjectHoldsTheFieldsDirectly) {
    const nlohmann::json doc = FullDocument()[0]["Config"];
    const Config c = serialize::LoadJson<Config>(doc);

    EXPECT_FLOAT_EQ(c.learning_rate, 0.5f);
    EXPECT_EQ(c.seed, -7);
    EXPECT_EQ(c.d_model, 512u);
    EXPECT_EQ(c.activation, Activation::SiLU);
    EXPECT_EQ(c.dims, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(c.head.units, 10u);
    ASSERT_EQ(c.layers.size(), 2u);
    EXPECT_EQ(c.layers[1].name, "b");
}

TEST(SerializeLoad, APlainObjectIsNotSearchedForTheClassName) {
    // In the object form the top-level keys are the fields, so a class-name
    // wrapper is just an unknown key.
    const auto doc = nlohmann::json::parse(R"({ "Config": { "seed": 5 } })");
    EXPECT_EQ(serialize::LoadJson<Config>(doc).seed, 0);
}

TEST(SerializeLoad, AScalarDocumentGivesDefaults) {
    EXPECT_EQ(serialize::LoadJson<Config>(nlohmann::json(42)).seed, 0);
}

// --- files ------------------------------------------------------------------

TEST(SerializeLoadJsonFile, LoadsFromAFile) {
    const TempFile file(FullDocument().dump());
    const Config c = serialize::LoadJson<Config>(file.Path());

    EXPECT_EQ(c.d_model, 512u);
    ASSERT_EQ(c.layers.size(), 2u);
    EXPECT_EQ(c.layers[1].name, "b");
}

TEST(SerializeLoadJsonFile, LoadsAPlainObjectFromAFile) {
    const TempFile file(R"({ "d_model": 256, "causal": true })");
    const Config c = serialize::LoadJson<Config>(file.Path());

    EXPECT_EQ(c.d_model, 256u);
    EXPECT_TRUE(c.causal);
}

TEST(SerializeLoadJsonFile, ThrowsForAMissingFile) {
    EXPECT_THROW(serialize::LoadJson<Config>("does/not/exist.json"), std::runtime_error);
}

TEST(SerializeLoadJsonFile, ThrowsForMalformedJson) {
    const TempFile file("[{ \"Config\": ");
    EXPECT_THROW(serialize::LoadJson<Config>(file.Path()), nlohmann::json::parse_error);
}

// --- a real model config ----------------------------------------------------

namespace {

// Mirrors the config.json shipped with Qwen2-0.5B on Hugging Face.
struct Qwen2Config {
    std::vector<std::string> architectures;
    float attention_dropout = -1.0f;
    size_t bos_token_id = 0;
    size_t eos_token_id = 0;
    std::string hidden_act;
    size_t hidden_size = 0;
    float initializer_range = 0.0f;
    size_t intermediate_size = 0;
    size_t max_position_embeddings = 0;
    size_t max_window_layers = 0;
    std::string model_type;
    size_t num_attention_heads = 0;
    size_t num_hidden_layers = 0;
    size_t num_key_value_heads = 0;
    double rms_norm_eps = 0.0;
    double rope_theta = 0.0;
    size_t sliding_window = 0;
    bool tie_word_embeddings = false;
    std::string torch_dtype;
    std::string transformers_version;
    bool use_cache = false;
    bool use_mrope = true;
    bool use_sliding_window = true;
    size_t vocab_size = 0;

    REFLECTABLE(Qwen2Config, architectures, attention_dropout, bos_token_id, eos_token_id, hidden_act,
                hidden_size, initializer_range, intermediate_size, max_position_embeddings,
                max_window_layers, model_type, num_attention_heads, num_hidden_layers,
                num_key_value_heads, rms_norm_eps, rope_theta, sliding_window, tie_word_embeddings,
                torch_dtype, transformers_version, use_cache, use_mrope, use_sliding_window, vocab_size)
};

Qwen2Config LoadQwen2() {
    return serialize::LoadJson<Qwen2Config>(nlohmann::json::parse(R"(
    {
      "architectures": [
        "Qwen2ForCausalLM"
      ],
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151643,
      "hidden_act": "silu",
      "hidden_size": 896,
      "initializer_range": 0.02,
      "intermediate_size": 4864,
      "max_position_embeddings": 32768,
      "max_window_layers": 24,
      "model_type": "qwen2",
      "num_attention_heads": 14,
      "num_hidden_layers": 24,
      "num_key_value_heads": 2,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000.0,
      "sliding_window": 32768,
      "tie_word_embeddings": true,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.40.1",
      "use_cache": true,
      "use_mrope": false,
      "use_sliding_window": false,
      "vocab_size": 151936
    })"));
}

}  // namespace

TEST(SerializeQwen2Config, LoadsEveryField) {
    const Qwen2Config c = LoadQwen2();

    EXPECT_EQ(c.architectures, (std::vector<std::string>{"Qwen2ForCausalLM"}));
    EXPECT_FLOAT_EQ(c.attention_dropout, 0.0f);
    EXPECT_EQ(c.bos_token_id, 151643u);
    EXPECT_EQ(c.eos_token_id, 151643u);
    EXPECT_EQ(c.hidden_act, "silu");
    EXPECT_EQ(c.hidden_size, 896u);
    EXPECT_FLOAT_EQ(c.initializer_range, 0.02f);
    EXPECT_EQ(c.intermediate_size, 4864u);
    EXPECT_EQ(c.max_position_embeddings, 32768u);
    EXPECT_EQ(c.max_window_layers, 24u);
    EXPECT_EQ(c.model_type, "qwen2");
    EXPECT_EQ(c.num_attention_heads, 14u);
    EXPECT_EQ(c.num_hidden_layers, 24u);
    EXPECT_EQ(c.num_key_value_heads, 2u);
    EXPECT_DOUBLE_EQ(c.rms_norm_eps, 1e-6);
    EXPECT_DOUBLE_EQ(c.rope_theta, 1000000.0);
    EXPECT_EQ(c.sliding_window, 32768u);
    EXPECT_TRUE(c.tie_word_embeddings);
    EXPECT_EQ(c.torch_dtype, "bfloat16");
    EXPECT_EQ(c.transformers_version, "4.40.1");
    EXPECT_TRUE(c.use_cache);
    EXPECT_FALSE(c.use_mrope);
    EXPECT_FALSE(c.use_sliding_window);
    EXPECT_EQ(c.vocab_size, 151936u);
}

TEST(SerializeQwen2Config, DescribesAValidAttentionLayout) {
    const Qwen2Config c = LoadQwen2();

    ASSERT_GT(c.num_attention_heads, 0u);
    ASSERT_GT(c.num_key_value_heads, 0u);
    EXPECT_EQ(c.hidden_size % c.num_attention_heads, 0u) << "hidden_size must split evenly across heads";
    EXPECT_EQ(c.hidden_size / c.num_attention_heads, 64u) << "head_dim";
    EXPECT_EQ(c.num_attention_heads % c.num_key_value_heads, 0u)
        << "grouped-query attention needs whole query groups per KV head";
    EXPECT_EQ(c.num_attention_heads / c.num_key_value_heads, 7u) << "query heads per KV head";
}
