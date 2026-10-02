#include <gtest/gtest.h>

#include <Models/Tokenizer.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using models::Tokenizer;
using json = nlohmann::json;

namespace {

json TinyTokenizer(bool ignore_merges = false, json merges = json::array({"a b"})) {
    return json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"a", 0}, {"b", 1}, {"ab", 2}, {"c", 3}, {"abc", 4}, {"x", 5}}},
          {"merges", merges},
          {"ignore_merges", ignore_merges}}},
    };
}

}  // namespace

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
