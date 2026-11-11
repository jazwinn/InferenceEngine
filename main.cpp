#include "Models/CausalModel.h"
#include "Models/ChatTemplate.h"
#include "Models/ModelConfigs.h"
#include "Models/Tokenizer.h"
#include "Models/WeightLoader.h"
#include "Generation/Sampler.h"
#include "UI/ChatUI.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using generation::Greedy;
using models::CausalModel;
using models::ChatTemplate;
using models::LoadWeights;
using models::Message;
using models::Tokenizer;
using nn::transformer::KVCache;

using Precision = float;

int main() {
  ui::EnableTerminal();

  try {
    const std::string modelDir = "Models/llama3.2-instruct/";
    const std::string systemPrompt = "You are a helpful assistant.";

    ui::Status("Loading tokenizer...");
    const Tokenizer tokenizer = Tokenizer::FromFile(modelDir + "tokenizer.json");
    const ChatTemplate chat = ChatTemplate::FromFile(modelDir + "tokenizer_config.json");
    std::vector<Message> history = {{"system", systemPrompt}};

    ui::Status("Loading model weights...");
    CausalModel<Precision> model(models::llama::Llama32_1B());
    LoadWeights<Precision>(model, modelDir + "model.safetensors", models::llama::Rules());
    const tensor::Device device = tensor::Device::CUDA;
    model.To(device);
    const size_t contextSize = std::min<size_t>(2048, model.config.max_seq_len);
    KVCache<Precision> cache = model.MakeCache(1, contextSize);

    Greedy<Precision> greedy;

    std::cout << ui::kClearLine;
    ui::Banner("Llama 3.2 1B Instruct (CUDA)");

    std::string input;
    while (ui::ReadPrompt(input)) {
      input.erase(0, input.find_first_not_of(" \t\r"));
      input.erase(input.find_last_not_of(" \t\r") + 1);
      if (input.empty()) continue;

      if (input == "/exit" || input == "/quit") break;
      if (input == "/clear") {
        history.assign({{"system", systemPrompt}});
        ui::Info("Conversation cleared.");
        continue;
      }
      if (input[0] == '/') {
        ui::Info("Unknown command. Use /clear or /exit.");
        continue;
      }

      history.push_back({"user", input});
      std::vector<int64_t> ids = tokenizer.Encode(chat.Apply(history), false);
      while (ids.size() >= contextSize && history.size() > 2) {
        history.erase(history.begin() + 1, history.begin() + 3);
        ids = tokenizer.Encode(chat.Apply(history), false);
      }
      if (ids.size() >= contextSize) {
        history.pop_back();
        ui::Info("Prompt is too long for the context window.");
        continue;
      }

      ui::Status(std::string(ui::kAccent) + ui::kIcon + ui::kReset + ui::kDim + " Thinking...");

      cache.Reset();

      tensor::Tensor<int64_t> prompt({1, ids.size()});
      for (size_t i = 0; i < ids.size(); ++i) prompt(0, i) = ids[i];

      tensor::Tensor<Precision> logits = model.Forward(prompt.To(device), &cache);
      size_t last = ids.size() - 1;

      std::cout << ui::kClearLine << ui::kAccent << ui::kDot << ui::kReset << " " << std::flush;

      std::string answer;
      bool atLineStart = false;
      while (ids.size() < contextSize) {
        const int64_t best = static_cast<int64_t>(greedy.Pick(logits.Row(0).Row(last).To(tensor::Device::CPU)));
        if (model.config.IsEos(best)) break;
        ids.push_back(best);

        const std::string piece = tokenizer.Decode({best});
        answer += piece;

        // Indent continuation lines so the reply stays aligned under the bullet.
        for (const char c : piece) {
          if (atLineStart) std::cout << "  ";
          std::cout << c;
          atLineStart = c == '\n';
        }
        std::cout << std::flush;

        if (ids.size() >= contextSize) break;

        tensor::Tensor<int64_t> next({1, 1});
        next(0, 0) = best;
        logits = model.Forward(next.To(device), &cache);
        last = 0;
      }

      std::cout << "\n\n";
      history.push_back({"assistant", std::move(answer)});
    }

    std::cout << "\n" << ui::kDim << "Goodbye!" << ui::kReset << "\n";
  } catch (const std::exception& error) {
    std::cerr << ui::kClearLine << "\nerror: " << error.what() << "\n";
    return 1;
  }
}
