#include "Models/CausalModel.h"
#include "Models/ModelConfigs.h"
#include "Models/Tokenizer.h"
#include "Models/WeightLoader.h"
#include "Generation/Sampler.h"
#include "CUDA/Memory.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Precision = float;
using Clock = std::chrono::steady_clock;

namespace {

double MsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double Median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::vector<int64_t> PromptOfLength(const models::Tokenizer& tokenizer, size_t length) {
    const std::string text =
        "The history of computing is a story of steadily shrinking machines and growing ambitions. "
        "Early mechanical calculators gave way to vacuum tubes, then transistors, then integrated circuits, "
        "and each step made it possible to ask harder questions of the machines we built. ";
    std::vector<int64_t> ids;
    while (ids.size() < length) {
        const std::vector<int64_t> piece = tokenizer.Encode(text, false);
        ids.insert(ids.end(), piece.begin(), piece.end());
    }
    ids.resize(length);
    return ids;
}

struct RunResult {
    double prefill_ms;
    double decode_ms_per_token;
};

RunResult Run(models::CausalModel<Precision>& model, nn::transformer::KVCache<Precision>& cache,
              const std::vector<int64_t>& prompt_ids, size_t decode_tokens) {
    const tensor::Device device = tensor::Device::CUDA;
    generation::Greedy<Precision> greedy;
    cache.Reset();

    tensor::Tensor<int64_t> prompt({1, prompt_ids.size()});
    for (size_t i = 0; i < prompt_ids.size(); ++i) prompt(0, i) = prompt_ids[i];

    auto start = Clock::now();
    tensor::Tensor<Precision> logits = model.Forward(prompt.To(device), &cache);
    int64_t next_id = static_cast<int64_t>(
        greedy.Pick(logits.Row(0).Row(prompt_ids.size() - 1).To(tensor::Device::CPU)));
    const double prefill_ms = MsSince(start);

    start = Clock::now();
    for (size_t step = 0; step < decode_tokens; ++step) {
        tensor::Tensor<int64_t> next({1, 1});
        next(0, 0) = next_id;
        logits = model.Forward(next.To(device), &cache);
        next_id = static_cast<int64_t>(greedy.Pick(logits.Row(0).Row(0).To(tensor::Device::CPU)));
    }
    const double decode_ms = MsSince(start);

    return {prefill_ms, decode_ms / static_cast<double>(decode_tokens)};
}

}

int main(int argc, char** argv) {
    const std::string model_dir = argc > 1 ? argv[1] : "Models/llama3.2-instruct/";
    const size_t decode_tokens = 64;
    const int repeats = 3;
    const std::vector<size_t> prompt_lengths = {16, 128, 512};

    try {
        const models::Tokenizer tokenizer = models::Tokenizer::FromFile(model_dir + "tokenizer.json");

        auto start = Clock::now();
        models::CausalModel<Precision> model(models::llama::Llama32_1B());
        models::LoadWeights<Precision>(model, model_dir + "model.safetensors", models::llama::Rules());
        model.To(tensor::Device::CUDA);
        const double load_ms = MsSince(start);

        const size_t context = std::min<size_t>(2048, model.config.max_seq_len);
        nn::transformer::KVCache<Precision> cache = model.MakeCache(1, context);

        Run(model, cache, PromptOfLength(tokenizer, 16), 8);

        std::printf("Llama 3.2 1B, fp32, greedy, %zu decode tokens, median of %d runs\n", decode_tokens, repeats);
        std::printf("load: %.0f ms\n\n", load_ms);
        std::printf("| prompt tokens | prefill (ms) | prefill (tok/s) | decode (ms/tok) | decode (tok/s) |\n");
        std::printf("|---:|---:|---:|---:|---:|\n");

        for (const size_t length : prompt_lengths) {
            const std::vector<int64_t> ids = PromptOfLength(tokenizer, length);
            std::vector<double> prefill, decode;
            for (int r = 0; r < repeats; ++r) {
                const RunResult result = Run(model, cache, ids, decode_tokens);
                prefill.push_back(result.prefill_ms);
                decode.push_back(result.decode_ms_per_token);
            }
            const double p = Median(prefill), d = Median(decode);
            std::printf("| %zu | %.1f | %.0f | %.2f | %.1f |\n",
                        length, p, 1000.0 * static_cast<double>(length) / p, d, 1000.0 / d);
        }

        std::printf("\nGPU memory held by the allocator: %.2f GB (%.2f GB of it cached for reuse)\n",
                    static_cast<double>(cuda::AllocatedBytes()) / 1e9,
                    static_cast<double>(cuda::CachedBytes()) / 1e9);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
