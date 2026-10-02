#include <gtest/gtest.h>

#include <Models/Tokenizer.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using models::Tokenizer;
using json = nlohmann::json;

namespace {

std::string SourcePath(const std::string& relative) {
    return std::string(IE_SOURCE_DIR) + "/" + relative;
}

const json& Cases() {
    static const json cases = [] {
        std::ifstream file(SourcePath("tests/data/tokenizer_cases.json"));
        return json::parse(file);
    }();
    return cases;
}

json TinyTokenizer(bool ignore_merges = false, json merges = json::array({"a b"})) {
    return json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"a", 0}, {"b", 1}, {"ab", 2}, {"c", 3}, {"abc", 4}, {"x", 5}}},
          {"merges", merges},
          {"ignore_merges", ignore_merges}}},
    };
}

std::vector<int64_t> Ids(const json& array) { return array.get<std::vector<int64_t>>(); }

void ExpectMatchesHuggingFace(const std::string& name, const std::string& path) {
    const std::string full = SourcePath(path);
    if (!std::filesystem::exists(full)) GTEST_SKIP() << full << " is not downloaded.";

    const Tokenizer tokenizer = Tokenizer::FromFile(full);
    for (const json& c : Cases().at(name)) {
        const std::string text = c.at("text").get<std::string>();
        SCOPED_TRACE("text: \"" + text + "\"");
        EXPECT_EQ(tokenizer.Encode(text, true), Ids(c.at("ids")));
        EXPECT_EQ(tokenizer.Encode(text, false), Ids(c.at("ids_no_special")));
        EXPECT_EQ(tokenizer.Decode(Ids(c.at("ids_no_special")), false), c.at("decoded").get<std::string>());
        EXPECT_EQ(tokenizer.Decode(Ids(c.at("ids_no_special")), true), c.at("decoded_skip").get<std::string>());
    }
}

}  // namespace

TEST(TokenizerHuggingFace, Llama3MatchesExactly) {
    ExpectMatchesHuggingFace("llama", "Models/llama3.2/tokenizer.json");
}

TEST(TokenizerHuggingFace, GPT2MatchesExactly) {
    ExpectMatchesHuggingFace("gpt2", "Models/gpt2/tokenizer.json");
}

TEST(TokenizerHuggingFace, Llama3VocabularyAndSpecialTokens) {
    const std::string path = SourcePath("Models/llama3.2/tokenizer.json");
    if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " is not downloaded.";
    const Tokenizer tokenizer = Tokenizer::FromFile(path);

    EXPECT_EQ(tokenizer.VocabSize(), 128256u);
    EXPECT_EQ(tokenizer.TokenToId("<|begin_of_text|>"), 128000);
    EXPECT_EQ(tokenizer.TokenToId("<|eot_id|>"), 128009);
    EXPECT_TRUE(tokenizer.IsSpecial(128009));
    EXPECT_FALSE(tokenizer.IsSpecial(9906));
    EXPECT_EQ(tokenizer.IdToToken(128001), "<|end_of_text|>");
    EXPECT_EQ(tokenizer.Encode("Hi").front(), 128000);
}

TEST(TokenizerHuggingFace, GPT2HasNoBosAndKnowsEndOfText) {
    const std::string path = SourcePath("Models/gpt2/tokenizer.json");
    if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " is not downloaded.";
    const Tokenizer tokenizer = Tokenizer::FromFile(path);

    EXPECT_EQ(tokenizer.VocabSize(), 50257u);
    EXPECT_EQ(tokenizer.TokenToId("<|endoftext|>"), 50256);
    EXPECT_EQ(tokenizer.Encode("Hello", true), tokenizer.Encode("Hello", false));
}

TEST(TokenizerHuggingFace, DecodingTokenByTokenConcatenatesToTheText) {
    const std::string path = SourcePath("Models/llama3.2/tokenizer.json");
    if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " is not downloaded.";
    const Tokenizer tokenizer = Tokenizer::FromFile(path);
    const std::string text = "Streaming 🙂 日本語 works.";

    std::string streamed;
    for (int64_t id : tokenizer.Encode(text, false)) streamed += tokenizer.Decode({id});

    EXPECT_EQ(streamed, text);
}

TEST(TokenizerBPE, AppliesMergesInRankOrder) {
    const Tokenizer tokenizer = Tokenizer::FromJson(TinyTokenizer());

    EXPECT_EQ(tokenizer.Encode("abc"), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tokenizer.Encode("ba"), (std::vector<int64_t>{1, 0}));
}

TEST(TokenizerBPE, AcceptsMergesAsPairs) {
    const Tokenizer strings = Tokenizer::FromJson(TinyTokenizer(false, json::array({"a b"})));
    const Tokenizer pairs = Tokenizer::FromJson(TinyTokenizer(false, json::array({json::array({"a", "b"})})));

    EXPECT_EQ(strings.Encode("abcab"), pairs.Encode("abcab"));
}

TEST(TokenizerBPE, IgnoreMergesUsesWholeWordsFromTheVocabulary) {
    EXPECT_EQ(Tokenizer::FromJson(TinyTokenizer(false)).Encode("abc"), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(Tokenizer::FromJson(TinyTokenizer(true)).Encode("abc"), (std::vector<int64_t>{4}));
}

TEST(TokenizerBPE, UnknownSymbolThrowsWithoutUnkToken) {
    EXPECT_THROW(Tokenizer::FromJson(TinyTokenizer()).Encode("z"), std::runtime_error);
}

TEST(TokenizerBPE, UnknownSymbolUsesUnkToken) {
    json doc = TinyTokenizer();
    doc["model"]["unk_token"] = "x";

    EXPECT_EQ(Tokenizer::FromJson(doc).Encode("azb"), (std::vector<int64_t>{0, 5, 1}));
}

TEST(TokenizerAddedTokens, AreSplitOutBeforeBPE) {
    json doc = TinyTokenizer();
    doc["added_tokens"] = json::array({{{"id", 10}, {"content", "<s>"}, {"special", true}}});

    EXPECT_EQ(Tokenizer::FromJson(doc).Encode("a<s>b"), (std::vector<int64_t>{0, 10, 1}));
}

TEST(TokenizerAddedTokens, LongestMatchWins) {
    json doc = TinyTokenizer();
    doc["added_tokens"] = json::array({{{"id", 10}, {"content", "<s>"}, {"special", true}},
                                       {{"id", 11}, {"content", "<s>>"}, {"special", true}}});

    EXPECT_EQ(Tokenizer::FromJson(doc).Encode("<s>><s>"), (std::vector<int64_t>{11, 10}));
}

TEST(TokenizerAddedTokens, DecodeKeepsOrSkipsSpecialTokens) {
    json doc = TinyTokenizer();
    doc["added_tokens"] = json::array({{{"id", 10}, {"content", "<s>"}, {"special", true}},
                                       {{"id", 11}, {"content", "<plain>"}, {"special", false}}});
    const Tokenizer tokenizer = Tokenizer::FromJson(doc);

    EXPECT_EQ(tokenizer.Decode({10, 0, 11, 1}), "<s>a<plain>b");
    EXPECT_EQ(tokenizer.Decode({10, 0, 11, 1}, true), "a<plain>b");
    EXPECT_EQ(tokenizer.VocabSize(), 12u);
}

TEST(TokenizerPostProcessor, TemplateAddsPrefixAndSuffix) {
    json doc = TinyTokenizer();
    doc["added_tokens"] = json::array({{{"id", 10}, {"content", "<s>"}, {"special", true}},
                                       {{"id", 11}, {"content", "</s>"}, {"special", true}}});
    doc["post_processor"] = {
        {"type", "TemplateProcessing"},
        {"single", json::array({{{"SpecialToken", {{"id", "<s>"}}}}, {{"Sequence", {{"id", "A"}}}},
                                {{"SpecialToken", {{"id", "</s>"}}}}})},
        {"special_tokens", {{"<s>", {{"ids", json::array({10})}}}, {"</s>", {{"ids", json::array({11})}}}}},
    };
    const Tokenizer tokenizer = Tokenizer::FromJson(doc);

    EXPECT_EQ(tokenizer.Encode("ab"), (std::vector<int64_t>{10, 2, 11}));
    EXPECT_EQ(tokenizer.Encode("ab", false), (std::vector<int64_t>{2}));
}

TEST(TokenizerPreTokenizer, SplitIsolatesMatchesAndGaps) {
    json doc = TinyTokenizer();
    doc["pre_tokenizer"] = {{"type", "Split"}, {"pattern", {{"Regex", "c"}}}, {"behavior", "Isolated"}};
    doc["model"]["vocab"]["bc"] = 6;
    doc["model"]["merges"] = json::array({"b c"});

    EXPECT_EQ(Tokenizer::FromJson(doc).Encode("bc"), (std::vector<int64_t>{1, 3}));
}

TEST(TokenizerErrors, RejectsUnsupportedComponents) {
    json model = TinyTokenizer();
    model["model"]["type"] = "Unigram";
    json normalizer = TinyTokenizer();
    normalizer["normalizer"] = {{"type", "NFC"}};
    json pre = TinyTokenizer();
    pre["pre_tokenizer"] = {{"type", "Metaspace"}};
    json decoder = TinyTokenizer();
    decoder["decoder"] = {{"type", "WordPiece"}};
    json fallback = TinyTokenizer();
    fallback["model"]["byte_fallback"] = true;

    for (const json& doc : {model, normalizer, pre, decoder, fallback})
        EXPECT_THROW(Tokenizer::FromJson(doc), std::runtime_error);
}

TEST(TokenizerErrors, RejectsInvalidRegexAndMalformedMerges) {
    json regex = TinyTokenizer();
    regex["pre_tokenizer"] = {{"type", "Split"}, {"pattern", {{"Regex", "(unclosed"}}}};
    json merges = TinyTokenizer(false, json::array({"nospace"}));

    EXPECT_THROW(Tokenizer::FromJson(regex), std::invalid_argument);
    EXPECT_THROW(Tokenizer::FromJson(merges), std::runtime_error);
}

TEST(TokenizerErrors, DecodeRejectsIdsOutsideTheVocabulary) {
    const Tokenizer tokenizer = Tokenizer::FromJson(TinyTokenizer());

    EXPECT_THROW(tokenizer.Decode({99}), std::out_of_range);
    EXPECT_THROW(tokenizer.Decode({-1}), std::out_of_range);
}

TEST(TokenizerErrors, RegexRejectsInvalidUtf8) {
    json doc = TinyTokenizer();
    doc["pre_tokenizer"] = {{"type", "Split"}, {"pattern", {{"Regex", "a"}}}, {"behavior", "Isolated"}};

    EXPECT_THROW(Tokenizer::FromJson(doc).Encode(std::string("a\xff")), std::invalid_argument);
}

TEST(TokenizerErrors, MissingFileThrows) {
    EXPECT_THROW(Tokenizer::FromFile("does/not/exist.json"), std::runtime_error);
}
