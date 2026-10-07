#include "Models/CausalModel.h"
#include "Models/ChatTemplate.h"
#include "Models/ModelConfigs.h"
#include "Models/Tokenizer.h"
#include "Models/WeightLoader.h"
#include "Generation/Sampler.h"

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

int main() {
  try {
    const std::string modelDir = "Models/llama3.2-instruct/";

    const Tokenizer tokenizer = Tokenizer::FromFile(modelDir + "tokenizer.json");
    const ChatTemplate chat = ChatTemplate::FromFile(modelDir + "tokenizer_config.json");
    std::vector<Message> history = {{"system", "You are a helpful assistant."}};

    CausalModel<float> model(models::llama::Llama32_1B());
    const size_t contextSize = std::min<size_t>(2048, model.config.max_seq_len);
    KVCache<float> cache(model.config.num_layers, 1, model.config.KVHeads(), contextSize, model.config.HeadDim());
    LoadWeights<float>(model, modelDir + "model.safetensors", models::llama::Rules());

    Greedy<float> greedy;

    while (true) {
      std::string input;
      std::cout << "Insert Prompt: ";
      if (!std::getline(std::cin, input)) break;
      if (input.empty()) continue;

      history.push_back({"user", input});
      std::vector<int64_t> ids = tokenizer.Encode(chat.Apply(history), false);
      while (ids.size() >= contextSize && history.size() > 2) {
        history.erase(history.begin() + 1, history.begin() + 3);
        ids = tokenizer.Encode(chat.Apply(history), false);
      }
      if (ids.size() >= contextSize) {
        history.pop_back();
        std::cout << "prompt is too long for the context window\n";
        continue;
      }

      cache.Reset();

      tensor::Tensor<int64_t> prompt({1, ids.size()});
      for (size_t i = 0; i < ids.size(); ++i) prompt(0, i) = ids[i];

      tensor::Tensor<float> logits = model.Forward(prompt, &cache);
      size_t last = ids.size() - 1;

      std::string answer;
      while (ids.size() < contextSize) {
        const int64_t best = static_cast<int64_t>(greedy.Pick(logits.Row(0).Row(last)));
        if (model.config.IsEos(best)) break;
        ids.push_back(best);

        const std::string piece = tokenizer.Decode({best});
        std::cout << piece << std::flush;
        answer += piece;

        if (ids.size() >= contextSize) break;

        tensor::Tensor<int64_t> next({1, 1});
        next(0, 0) = best;
        logits = model.Forward(next, &cache);
        last = 0;
      }

      std::cout << "\n";
      history.push_back({"assistant", std::move(answer)});
    }
  } catch (const std::exception& error) {
    std::cerr << "\nerror: " << error.what() << "\n";
    return 1;
  }
}
