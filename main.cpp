#include "Models/CausalModel.h"
#include "Models/ModelConfigs.h"
#include "Models/Tokenizer.h"
#include "Models/gpt2/Checkpoint.h"
#include "Models/WeightLoader.h"
#include "Generation/Sampler.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

using models::CausalModel;
using nn::transformer::config::TransformerConfig;
using nn::transformer::KVCache;
using generation::Greedy;
using models::Tokenizer;
using models::gpt2::ReadCheckpoint;
using models::LoadWeights;

int main() {
  try {
    const Tokenizer tokenizer = Tokenizer::FromFile("Models/gpt2/tokenizer.json");

    CausalModel<float> GPT2Model(models::gpt2::Small());
    KVCache<float> cache(GPT2Model.config.num_layers, 1, GPT2Model.config.KVHeads(),
                         GPT2Model.config.max_seq_len, GPT2Model.config.HeadDim());
    LoadWeights<float>(GPT2Model, ReadCheckpoint<float>("Models/gpt2/model.safetensors"));
                    

    while (true) {
      std::string input;
      std::cout << "Insert Prompt: ";
      if (!std::getline(std::cin, input)) break;

      std::vector<int64_t> ids = tokenizer.Encode(input);
      if (ids.empty()) continue;
      if (ids.size() >= GPT2Model.config.max_seq_len) {
        std::cout << "prompt is too long for the context window\n";
        continue;
      }

      cache.Reset();

      Greedy<float> greedy;

      math::Tensor<int64_t> prompt({1, ids.size()});
      for (size_t i = 0; i < ids.size(); ++i) prompt(0, i) = ids[i];

      math::Tensor<float> logits = GPT2Model.Forward(prompt, &cache);
      size_t last = ids.size() - 1;

      while (ids.size() < GPT2Model.config.max_seq_len) {
          const int64_t best = static_cast<int64_t>(greedy.Pick(logits.Row(0).Row(last)));

          if (GPT2Model.config.IsEos(best)) break;
          ids.push_back(best);
          std::cout << tokenizer.Decode({best}) << std::flush;

          if (ids.size() >= GPT2Model.config.max_seq_len) break;

          math::Tensor<int64_t> next({1, 1});
          next(0, 0) = best;
          logits = GPT2Model.Forward(next, &cache);
          last   = 0;
      }

      std::cout << "\n";
    }

  } catch (const std::exception& error) {
    std::cerr << "\nerror: " << error.what() << "\n";
    return 1;
  }
}
