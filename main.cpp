#include "Models/gpt2/GPT2.h"
#include "Models/ModelConfigs.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

using llm::GPT2;
using llm::GPT2Tokenizer;
using transformer::TransformerConfig;

int main() {
  try {
    const GPT2Tokenizer tokenizer("Models/gpt2/vocab.json", "Models/gpt2/merges.txt");

    GPT2<float> model(models::gpt2::Small());
    llm::LoadGPT2Weights(model, "Models/gpt2/model.safetensors");

    std::vector<int64_t> ids = tokenizer.Encode("Hello, world!");
    std::cout << tokenizer.Decode(ids) << std::flush;

    while (ids.size() < model.config.max_seq_len) {
        math::Tensor<int64_t> input({1, ids.size()});
        for (size_t i = 0; i < ids.size(); ++i) input(0, i) = ids[i];

        const math::Tensor<float> logits = model.Forward(input);

        const size_t last = ids.size() - 1;
        int64_t best = 0;
        float best_score = logits(0, last, 0);
        for (size_t v = 1; v < model.config.vocab_size; ++v) {
            if (logits(0, last, v) > best_score) {
                best_score = logits(0, last, v);
                best = static_cast<int64_t>(v);
            }
        }

        if (model.config.eos_token_id && best == *model.config.eos_token_id) break;

        ids.push_back(best);
        std::cout << tokenizer.Decode({best}) << std::flush;
    }

    std::cout << "\n";
  } catch (const std::exception& error) {
    std::cerr << "\nerror: " << error.what() << "\n";
    return 1;
  }
}
