#include <gtest/gtest.h>

#include <Models/ChatTemplate.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using models::ChatTemplate;
using models::Message;
using Json = nlohmann::json;

namespace {

std::string SourcePath(const std::string& relative) { return std::string(IE_SOURCE_DIR) + "/" + relative; }

const Json& Cases() {
    static const Json cases = [] {
        std::ifstream file(SourcePath("tests/data/chat_template_cases.json"));
        return Json::parse(file);
    }();
    return cases;
}

std::chrono::system_clock::time_point LocalNoon(int year, int month, int day) {
    std::tm t{};
    t.tm_year = year - 1900;
    t.tm_mon = month - 1;
    t.tm_mday = day;
    t.tm_hour = 12;
    t.tm_isdst = -1;
    return std::chrono::system_clock::from_time_t(std::mktime(&t));
}

std::vector<Message> Messages(const Json& array) {
    std::vector<Message> messages;
    for (const Json& m : array) messages.push_back({m.at("role").get<std::string>(), m.at("content").get<std::string>()});
    return messages;
}

ChatTemplate Family(const std::string& name) {
    const Json& family = Cases().at(name);
    return ChatTemplate::FromSource(family.at("template").get<std::string>(), family.at("bos_token").get<std::string>(),
                                    family.at("eos_token").get<std::string>());
}

void ExpectMatchesHuggingFace(const std::string& name) {
    const ChatTemplate chat = Family(name);
    const Json& cases = Cases().at(name).at("cases");
    ASSERT_FALSE(cases.empty());
    for (const Json& c : cases) {
        SCOPED_TRACE(name + " / " + c.at("name").get<std::string>());
        EXPECT_EQ(chat.Apply(Messages(c.at("messages")), c.at("add_generation_prompt").get<bool>(), LocalNoon(2026, 10, 2)),
                  c.at("expected").get<std::string>());
    }
}

class TempDir {
public:
    TempDir()
        : m_path(std::filesystem::temp_directory_path() /
                 ("chattemplate_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)))) {
        std::filesystem::create_directories(m_path);
    }
    ~TempDir() { std::filesystem::remove_all(m_path); }

    std::string Write(const std::string& name, const std::string& contents) const {
        std::ofstream(m_path / name, std::ios::binary) << contents;
        return (m_path / name).string();
    }

private:
    std::filesystem::path m_path;
};

const char* kSimple = "{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}</>{% endfor %}"
                      "{% if add_generation_prompt %}<assistant>{% endif %}";

}  // namespace

TEST(ChatTemplateHuggingFace, Qwen25MatchesExactly) { ExpectMatchesHuggingFace("qwen2.5"); }
TEST(ChatTemplateHuggingFace, SmolLM2MatchesExactly) { ExpectMatchesHuggingFace("smollm2"); }
TEST(ChatTemplateHuggingFace, Gemma2MatchesExactly) { ExpectMatchesHuggingFace("gemma2"); }
TEST(ChatTemplateHuggingFace, Phi35MatchesExactly) { ExpectMatchesHuggingFace("phi3.5"); }
TEST(ChatTemplateHuggingFace, MistralMatchesExactly) { ExpectMatchesHuggingFace("mistral"); }

TEST(ChatTemplateApply, GenerationPromptIsOptional) {
    const ChatTemplate chat = ChatTemplate::FromSource(kSimple, "<s>");

    EXPECT_EQ(chat.Apply({{"user", "hi"}}), "<s><user>hi</><assistant>");
    EXPECT_EQ(chat.Apply({{"user", "hi"}}, false), "<s><user>hi</>");
}

TEST(ChatTemplateApply, MergesASystemMessageWhenTheTemplateHasNoSystemRole) {
    const ChatTemplate chat = Family("gemma2");
    const std::string prompt = chat.Apply({{"system", "Be brief."}, {"user", "hello!"}});

    const size_t system = prompt.find("Be brief.");
    const size_t user = prompt.find("hello!");
    ASSERT_NE(system, std::string::npos);
    ASSERT_NE(user, std::string::npos);
    EXPECT_LT(system, user);
    EXPECT_EQ(prompt.find("<start_of_turn>system"), std::string::npos);
}

TEST(ChatTemplateApply, TemplateRaisedErrorsBecomeRuntimeErrors) {
    const ChatTemplate chat = ChatTemplate::FromSource("{{ raise_exception('roles must alternate') }}");

    try {
        chat.Apply({{"user", "hi"}});
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("roles must alternate"), std::string::npos);
    }
}

TEST(ChatTemplateLoad, ReadsTokensFromTheConfig) {
    const Json config = {{"chat_template", kSimple},
                         {"bos_token", {{"content", "<s>"}, {"lstrip", false}}},
                         {"eos_token", "</s>"}};
    const ChatTemplate chat = ChatTemplate::FromJson(config);

    EXPECT_EQ(chat.BosToken(), "<s>");
    EXPECT_EQ(chat.EosToken(), "</s>");
    EXPECT_EQ(chat.Source(), kSimple);
    EXPECT_EQ(chat.Apply({{"user", "hi"}}, false), "<s><user>hi</>");
}

TEST(ChatTemplateLoad, MissingTokensAreEmpty) {
    const ChatTemplate chat = ChatTemplate::FromJson({{"chat_template", kSimple}, {"bos_token", nullptr}});

    EXPECT_EQ(chat.BosToken(), "");
    EXPECT_EQ(chat.Apply({{"user", "hi"}}, false), "<user>hi</>");
}

TEST(ChatTemplateLoad, PicksTheDefaultNamedTemplate) {
    const Json config = {{"chat_template", Json::array({{{"name", "tool_use"}, {"template", "tools"}},
                                                        {{"name", "default"}, {"template", "plain"}}})}};

    EXPECT_EQ(ChatTemplate::FromJson(config).Apply({{"user", "hi"}}), "plain");
}

TEST(ChatTemplateLoad, RejectsConfigsWithoutAUsableTemplate) {
    const Json named = {{"chat_template", Json::array({{{"name", "tool_use"}, {"template", "tools"}}})}};

    EXPECT_THROW(ChatTemplate::FromJson({{"bos_token", "<s>"}}), std::runtime_error);
    EXPECT_THROW(ChatTemplate::FromJson({{"chat_template", nullptr}}), std::runtime_error);
    EXPECT_THROW(ChatTemplate::FromJson({{"chat_template", 3}}), std::runtime_error);
    EXPECT_THROW(ChatTemplate::FromJson(named), std::runtime_error);
}

TEST(ChatTemplateLoad, RejectsInvalidTemplateSyntax) {
    EXPECT_THROW(ChatTemplate::FromSource("{% for m in messages %}never closed"), std::runtime_error);
}

TEST(ChatTemplateLoad, ReadsTheTemplateFromATokenizerConfigFile) {
    const TempDir dir;
    const std::string path = dir.Write("tokenizer_config.json", Json{{"chat_template", kSimple}, {"bos_token", "<s>"}}.dump());

    EXPECT_EQ(ChatTemplate::FromFile(path).Apply({{"user", "hi"}}), "<s><user>hi</><assistant>");
}

TEST(ChatTemplateLoad, FallsBackToAChatTemplateJinjaFile) {
    const TempDir dir;
    const std::string path = dir.Write("tokenizer_config.json", Json{{"bos_token", "<s>"}}.dump());
    dir.Write("chat_template.jinja", kSimple);

    EXPECT_EQ(ChatTemplate::FromFile(path).Apply({{"user", "hi"}}), "<s><user>hi</><assistant>");
}

TEST(ChatTemplateLoad, BaseModelConfigWithoutATemplateThrows) {
    const TempDir dir;
    const std::string path = dir.Write("tokenizer_config.json", Json{{"bos_token", "<|begin_of_text|>"}}.dump());

    EXPECT_THROW(ChatTemplate::FromFile(path), std::runtime_error);
}

TEST(ChatTemplateLoad, MissingFileThrows) {
    EXPECT_THROW(ChatTemplate::FromFile("does/not/exist.json"), std::runtime_error);
}
