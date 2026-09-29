#include "Models/gpt2/GPT2.h"
#include "Models/ModelConfigs.h"
#include "Generation/Sampler.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

using llm::GPT2;
using llm::GPT2Tokenizer;
using nn::transformer::TransformerConfig;
using nn::transformer::KVCache;
using generation::Greedy;

int main() {
  try {
    const GPT2Tokenizer tokenizer("Models/gpt2/vocab.json", "Models/gpt2/merges.txt");

    GPT2<float> model(models::gpt2::Small());
    KVCache<float> cache(model.config.num_layers, 1, model.config.KVHeads(),
                         model.config.max_seq_len, model.config.HeadDim());

    llm::LoadGPT2Weights(model, "Models/gpt2/model.safetensors");

    while (true) {
      std::string input;
      std::cout << "Insert Prompt: ";
      if (!std::getline(std::cin, input)) break;

      std::vector<int64_t> ids = tokenizer.Encode(input);
      if (ids.empty()) continue;
      if (ids.size() >= model.config.max_seq_len) {
        std::cout << "prompt is too long for the context window\n";
        continue;
      }

      cache.Reset();

      Greedy<float> greedy;

      math::Tensor<int64_t> prompt({1, ids.size()});
      for (size_t i = 0; i < ids.size(); ++i) prompt(0, i) = ids[i];

      math::Tensor<float> logits = model.Forward(prompt, &cache);
      size_t last = ids.size() - 1;

      while (ids.size() < model.config.max_seq_len) {
          const int64_t best = static_cast<int64_t>(greedy.Pick(logits.Row(0).Row(last)));

          if (model.config.eos_token_id && best == *model.config.eos_token_id) break;
          ids.push_back(best);
          std::cout << tokenizer.Decode({best}) << std::flush;

          if (ids.size() >= model.config.max_seq_len) break;

          math::Tensor<int64_t> next({1, 1});
          next(0, 0) = best;
          logits = model.Forward(next, &cache);
          last   = 0;
      }

      std::cout << "\n";
    }

  } catch (const std::exception& error) {
    std::cerr << "\nerror: " << error.what() << "\n";
    return 1;
  }
}
