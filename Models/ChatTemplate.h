#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4702)
#endif
#include <minja/chat-template.hpp>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace models
{

    struct Message
    {
        std::string role;
        std::string content;
    };

    class ChatTemplate
    {
    public:
        static ChatTemplate FromFile(const std::string &tokenizer_config_path)
        {
            const std::filesystem::path path(tokenizer_config_path);
            const nlohmann::json config = nlohmann::json::parse(ReadText(path));
            if (HasTemplate(config))
            {
                return FromJson(config);
            }

            const std::filesystem::path sibling = path.parent_path() / "chat_template.jinja";
            if (std::filesystem::exists(sibling))
            {
                return FromSource(ReadText(sibling), TokenText(config, "bos_token"), TokenText(config, "eos_token"));
            }
            throw Error("'" + path.string() + "' has no chat_template and there is no chat_template.jinja beside it; "
                        "base models usually have none.");
        }

        static ChatTemplate FromJson(const nlohmann::json &config)
        {
            if (!HasTemplate(config))
            {
                throw Error("tokenizer config has no chat_template; base models usually have none.");
            }
            return FromSource(SelectTemplate(config.at("chat_template")), TokenText(config, "bos_token"),
                              TokenText(config, "eos_token"));
        }

        static ChatTemplate FromSource(const std::string &source, const std::string &bos_token = "",
                                       const std::string &eos_token = "")
        {
            ChatTemplate chat;
            try
            {
                chat.m_template = std::make_shared<const minja::chat_template>(source, bos_token, eos_token);
            }
            catch (const std::exception &error)
            {
                throw Error(std::string("cannot parse the template: ") + error.what());
            }
            return chat;
        }

        std::string Apply(const std::vector<Message> &messages, bool add_generation_prompt = true,
                          std::chrono::system_clock::time_point now = std::chrono::system_clock::now()) const
        {
            minja::chat_template_inputs inputs;
            inputs.messages = nlohmann::ordered_json::array();
            for (const Message &message : messages)
            {
                inputs.messages.push_back({{"role", message.role}, {"content", message.content}});
            }
            inputs.add_generation_prompt = add_generation_prompt;
            inputs.now = now;

            try
            {
                return m_template->apply(inputs);
            }
            catch (const std::exception &error)
            {
                throw Error(std::string("cannot render the template: ") + error.what());
            }
        }

        const std::string &Source() const { return m_template->source(); }
        const std::string &BosToken() const { return m_template->bos_token(); }
        const std::string &EosToken() const { return m_template->eos_token(); }

    private:
        ChatTemplate() = default;

        static std::runtime_error Error(const std::string &message)
        {
            return std::runtime_error("ChatTemplate: " + message);
        }

        static bool HasTemplate(const nlohmann::json &config)
        {
            const auto entry = config.find("chat_template");
            return entry != config.end() && !entry->is_null();
        }

        static std::string SelectTemplate(const nlohmann::json &value)
        {
            if (value.is_string())
            {
                return value.get<std::string>();
            }
            if (value.is_array())
            {
                for (const nlohmann::json &entry : value)
                {
                    if (entry.value("name", "") == "default")
                    {
                        return entry.at("template").get<std::string>();
                    }
                }
                throw Error("chat_template lists named templates but none is called 'default'.");
            }
            throw Error("chat_template must be a string or a list of named templates.");
        }

        static std::string TokenText(const nlohmann::json &config, const char *key)
        {
            const auto entry = config.find(key);
            if (entry == config.end() || entry->is_null())
            {
                return "";
            }
            if (entry->is_object())
            {
                return entry->at("content").get<std::string>();
            }
            return entry->get<std::string>();
        }

        static std::string ReadText(const std::filesystem::path &path)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file)
            {
                throw Error("cannot open '" + path.string() + "'.");
            }
            return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        }

        std::shared_ptr<const minja::chat_template> m_template;
    };

}
