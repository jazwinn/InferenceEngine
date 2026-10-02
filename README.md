# InferenceEngine

A transformer inference engine written from scratch in C++20, with no ML dependencies. It builds decoder-only models from a config, loads Hugging Face `.safetensors` checkpoints, tokenizes with the model's own `tokenizer.json`, formats chats with the model's own chat template, and generates text. It runs GPT-2 (124M) and Llama 3.2 1B Instruct.

## Status

Llama 3.2 1B Instruct chats end to end: chat template → tokenize → forward pass with KV cache → greedy decode → streamed detokenize, with multi-turn history. The output is correct but slow, because `MatMul` is still a naive single-threaded loop (see Performance in the roadmap).

## Layout

| Directory | Contents |
|---|---|
| `Math/` | `Tensor<T>`: shapes, strides, views (reshape, slice, permute, transpose, squeeze/unsqueeze), broadcasting, batched `MatMul`, reductions, masked fill |
| `NeuralNet/` | `Module` base (named parameters, submodules), `Linear`, `Embedding`, `Sequential`, activations (ReLU, GELU, GELU-tanh, SiLU, SwiGLU, Sigmoid, Tanh, Softmax) |
| `NeuralNet/Transformer/` | `GroupQueryAttention` (MHA/GQA/MQA), `RotaryEmbedding` (RoPE with scaling), `MLP` (plain or gated), `KVCache`, `DecoderBlock`/`Decoder`, `TransformerConfig` |
| `NeuralNet/Transformer/Normal/` | `Norm` base, `LayerNorm`, `RMSNorm`, `MakeNorm` (picks one from the config) |
| `Models/` | `CausalModel`, generic `Tokenizer` (`tokenizer.json`), `ChatTemplate`, generic `WeightLoader`, config presets and Llama rename rules (`ModelConfigs.h`), GPT-2 checkpoint conversion (`gpt2/Checkpoint.h`) |
| `Generation/` | `LogitsProcessor` pipeline, `Temperature`, `Greedy` sampler |
| `Serialize/` | `.safetensors` reader (F32/F64/F16/BF16/int/bool), reflection-based JSON loading |
| `Reflection/` | `REFLECTABLE` macro: member names and iteration for JSON config structs |
| `tests/` | 803 GoogleTest cases |

## Implemented

- **Tensors** — views and broadcasting without copying; `MatMul` over batched operands of any rank.
- **Attention** — grouped-query attention covering MHA, GQA and MQA through a 5-D broadcast (no K/V repetition); causal and explicit masks normalised to one shape; masked rows filled with the lowest finite value so fully masked rows stay finite; KV cache that stores the compact KV-head layout.
- **Positions** — learned embeddings (GPT-2) or rotary embeddings, with `default`, `linear` and `llama3` frequency scaling (`llama3` checked against Hugging Face's inverse frequencies).
- **Model** — one config-driven `CausalModel`: pre- or post-norm blocks, LayerNorm or RMSNorm, plain or gated MLP with GELU/GELU-tanh/SiLU/ReLU, optional biases, tied or separate output head. GPT-2 small and Llama 3.2 1B are presets in `ModelConfigs.h`.
- **Checkpoints** — `.safetensors` loading with dtype conversion (including BF16). A generic loader fills any `Module` by parameter name, using rename rules with `{i}` layer placeholders; it moves tensors instead of copying them, and checks every parameter, shape and checkpoint tensor before changing the model. Llama loads through rename rules alone; GPT-2 first converts its checkpoint (Conv1D transposes, fused `c_attn` split into q/k/v, mask buffers dropped). GPT-2 logits match Hugging Face to within 6e-5.
- **Tokenizer** — reads `tokenizer.json`: added/special tokens split out first (longest match wins), `Split`/`ByteLevel`/`Sequence` pre-tokenizers with Unicode regexes (PCRE2), BPE with id-based merging and a per-word cache, `TemplateProcessing` for BOS/EOS, and the `ByteLevel` decoder. Matches Hugging Face `tokenizers` exactly on GPT-2 and Llama 3. Unsupported components are rejected with an error rather than ignored.
- **Chat templates** — renders the model's own Jinja chat template (from `tokenizer_config.json` or `chat_template.jinja`) with [minja](https://github.com/google/minja), including `strftime_now` and a merged system message for templates without a system role. Matches Hugging Face on Qwen2.5, SmolLM2, Gemma 2, Phi-3.5 and Mistral.
- **Generation** — greedy decoding over a logits-processor pipeline, multiple stop tokens, streamed output, and multi-turn chat that drops the oldest turns to stay within the context window.
- **Config** — JSON model configs loaded through compile-time reflection.

## Build

Requires CMake 3.21+, a C++20 compiler (MSVC with Visual Studio 2026 or 2022 tested) and vcpkg. vcpkg installs gtest, nlohmann-json and pcre2 from `vcpkg.json`; CMake downloads minja (header-only, pinned by hash) during configure.

```
cmake --preset default
cmake --build --preset release
cmake --build --preset debug
ctest --preset debug
```

With Visual Studio 2022, use `vs2022` and `vs2022-release`/`vs2022-debug` instead.

## Run

`main.cpp` runs Llama 3.2 1B Instruct as a chat. Put these files from [`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) (an ungated copy of `meta-llama/Llama-3.2-1B-Instruct`) in `Models/llama3.2-instruct/`:

- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json` (holds the chat template)

Then run `build-vs/Release/inference.exe` from the repository root, so the relative paths resolve. Loading peaks at about 10 GB of memory, and with the current `MatMul` the first reply takes many minutes.

## Roadmap

**Performance** — the current bottleneck. Llama 3.2 1B runs correctly but slowly: `MatMul` reaches about 0.075 GFLOP/s on its projections, so a short chat prompt takes roughly 20+ minutes to the first token.
- [x] KV cache — reuse past keys/values so each step processes one token instead of the whole sequence
- [ ] `Linear` kernel — `Linear` multiplies by a transposed (strided) weight view, so the inner loop jumps 8–32 KB per step and cannot vectorise; a contiguous row-dot-product kernel is the single biggest win
- [ ] Multithreading — `MatMul` runs on one core
- [ ] SIMD (AVX2/FMA, AVX-512), blocked/cache-aware `MatMul`, fused attention
- [ ] Last-position-only output projection — prefill currently computes the 128K-vocabulary logits for every prompt position (about 20% of Llama prefill) and uses only the last row
- [ ] Prefix caching — keep the KV cache across chat turns and encode only the new turn instead of the whole history
- [ ] Memory — load weights straight into parameters (loading currently peaks at about 2× model size), memory-mapped weights, arena allocation, no allocation in the hot loop
- [ ] Quantization — int8/int4 weights, BF16/FP16 storage and compute paths
- [ ] GPU backend (CUDA/Vulkan) behind a device abstraction

**Architectures**
- [x] RoPE, with `default`, `linear` and `llama3` frequency scaling
- [ ] Remaining RoPE scaling types — dynamic NTK, YaRN, LongRoPE (currently rejected)
- [x] Grouped-query and multi-query attention
- [x] Gated MLP (SwiGLU) and config-driven norm/activation/placement selection
- [ ] QK-norm (Qwen3, Gemma 3), sandwich norms and `(1 + w)` RMSNorm (Gemma)
- [ ] Mixture of Experts — router, expert dispatch, top-k gating
- [ ] Sliding-window and chunked attention
- [ ] Encoder and encoder-decoder support (cross-attention)

**Models**
- [x] Config-driven model: one generic `DecoderBlock`/`Decoder`/`CausalModel` assembled from `TransformerConfig`
- [x] Generic weight loader — rename rules with `{i}` layer indices, all-or-nothing loading with missing/unused/shape checks
- [x] GPT-2 checkpoint conversion (Conv1D transposes, fused QKV split, mask buffers dropped) and Llama rename rules
- [ ] More families — Qwen2/2.5, Mistral, SmolLM, Gemma
- [ ] Hugging Face `config.json` → `TransformerConfig` translation, and picking the weight adapter from `"architectures"`
- [ ] Sharded checkpoints (`model-00001-of-*.safetensors` + index)

**Tokenizers**
- [x] Generic `tokenizer.json` BPE — added tokens, `Split`/`ByteLevel`/`Sequence` pre-tokenizers, both merge formats, `TemplateProcessing` BOS/EOS, `ByteLevel` decoder; matches Hugging Face exactly on GPT-2 and Llama 3
- [x] Special-token matching before splitting; BOS/EOS post-processing
- [x] Chat templates — the model's own Jinja template (`tokenizer_config.json` or `chat_template.jinja`) rendered with minja; matches Hugging Face on Qwen2.5, SmolLM2, Gemma 2, Phi-3.5 and Mistral
- [ ] Normalizers (NFC/NFKC, lowercase), `Metaspace` and `byte_fallback` — SentencePiece-style BPE for Llama 2, Mistral and Gemma
- [ ] Unigram and WordPiece models
- [ ] Streaming-safe decode — hold back incomplete UTF-8 sequences between tokens

**Generation**

*Decoding* — a composable logit pipeline (penalties → temperature → truncation → sample), so each stage stays independent and testable:
- [x] Greedy search — argmax each step. Worth keeping behind a flag: being deterministic, it is the only decoding mode that gives bit-exact regression tests. It is also degenerate on its own, falling into repetition loops because the continuation becomes a fixed function of the prefix.
- [x] `LogitsProcessor` pipeline
- [x] Temperature — divide logits by `T` before the softmax; `T < 1` sharpens, `T > 1` flattens
- [ ] Top-k sampling — keep the k highest-scoring tokens, renormalise, sample
- [ ] Top-p (nucleus) sampling — keep the smallest set whose cumulative probability exceeds p, so the candidate count adapts to how confident the model is
- [ ] Min-p sampling — keep tokens within a ratio of the top probability; cheaper than top-p and needs no sort
- [ ] Repetition, frequency and presence penalties — discount tokens already generated
- [ ] Beam search with length normalisation — for tasks where the likeliest sequence is actually wanted
- [ ] Contrastive search — a deterministic alternative to greedy that penalises tokens too similar to the existing context
- [ ] Seeded RNG so sampled runs stay reproducible

*Other*
- [x] Multiple stop tokens (`eos_token_ids`) and token-by-token streaming output
- [x] Multi-turn chat with history trimming to the context window
- [ ] Stop sequences (multi-token strings)
- [ ] Reusable chat session class instead of the loop in `main.cpp`
- [ ] Batched generation
- [ ] Speculative decoding — draft with a small model, verify in one pass with the large one

**Correctness and tooling**
- [ ] Reference-parity harness: compare per-layer activations and logits against Hugging Face for each supported model (GPT-2 logits were verified to within 6e-5; the model-level parity tests were removed from the unit suite)
- [ ] Benchmarks: tokens/sec, time to first token, memory, per-layer timing
- [ ] CLI with model-directory/prompt/sampling flags instead of hard-coded paths; optional server API
