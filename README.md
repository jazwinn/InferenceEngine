# InferenceEngine

A transformer inference engine written from scratch in C++20, with no ML dependencies. It builds decoder-only models from a config, loads Hugging Face `.safetensors` checkpoints, tokenizes with the model's own `tokenizer.json`, formats chats with the model's own chat template, and generates text, on the CPU or on an NVIDIA GPU through its own CUDA backend. It runs GPT-2 (124M) and Llama 3.2 1B Instruct.

## Status

Llama 3.2 1B Instruct chats end to end: chat template → tokenize → forward pass with KV cache → greedy decode → streamed detokenize, with multi-turn history.

The whole forward pass now runs on CUDA: embeddings, norms, linear layers, RoPE, the KV cache, attention and the output head. On small Llama-style and GPT-2-style models, GPU logits match the CPU path through a prompt and several decode steps. The GPU kernels are still naive (one thread per output element, no shared-memory tiling), and the weights are fp32, so Llama 3.2 1B uses about 4.9 GB of a 6 GB card. The CPU path is correct but slow, because its `MatMul` is a single-threaded loop. See Performance in the roadmap.

## Layout

| Directory | Contents |
|---|---|
| `Tensor/` | `Tensor<T>`: shapes, strides, views (reshape, slice, permute, transpose, squeeze/unsqueeze), broadcasting, batched `MatMul`, reductions, masked fill |
| `NeuralNet/` | `Module` base (named parameters, submodules), `Linear`, `Embedding`, `Sequential`, activations (ReLU, GELU, GELU-tanh, SiLU, SwiGLU, Sigmoid, Tanh, Softmax) |
| `NeuralNet/Transformer/` | `GroupQueryAttention` (MHA/GQA/MQA), `RotaryEmbedding` (RoPE with scaling), `MLP` (plain or gated), `KVCache`, `DecoderBlock`/`Decoder`, `TransformerConfig` |
| `NeuralNet/Transformer/Normal/` | `Norm` base, `LayerNorm`, `RMSNorm`, `MakeNorm` (picks one from the config) |
| `Models/` | `CausalModel`, generic `Tokenizer` (`tokenizer.json`), `ChatTemplate`, generic `WeightLoader`, config presets and Llama rename rules (`ModelConfigs.h`), GPT-2 checkpoint conversion (`gpt2/Checkpoint.h`) |
| `Generation/` | `LogitsProcessor` pipeline, `Temperature`, `Greedy` sampler |
| `Serialize/` | `.safetensors` reader (F32/F64/F16/BF16/int/bool), reflection-based JSON loading |
| `Reflection/` | `REFLECTABLE` macro: member names and iteration for JSON config structs |
| `CUDA/` | CUDA kernels and launchers (elementwise and strided ops, fill and masked fill, reductions, softmax, activations, RoPE, embedding lookup, strided copy, linear, batched strided matmul) and a caching device allocator |
| `tests/` | 1,385 GoogleTest cases, including GPU-versus-CPU checks for every CUDA op and for the full model |

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
- **CUDA backend** — the device is a property of the tensor: `Tensor::To(Device::CUDA)` and `Module::To()` move parameters and registered buffers (such as the RoPE tables), and each op runs on the device of its inputs, throwing on mixed devices. Kernels follow one pattern (an op functor, a kernel templated on it, a launcher that switches on an enum) and work on strided views directly through a shared layout helper, so transposes, slices and broadcasts need no copy. Reductions and softmax use one block per output with a shared `BlockReduce`. Batched `MatMul` takes broadcast batch strides plus row and column strides, so `Q·Kᵀ` over a sliced KV cache with grouped-query broadcasting runs with no copies. A caching allocator reuses freed blocks (rounded to 512 B under 1 MiB and to 2 MiB above, capped at 256 MiB cached), clears the cache and retries on out-of-memory, and rejects double frees. `CausalModel::MakeCache` creates the KV cache on the model's device.

## Build

Requires CMake 3.21+, a C++20 compiler (MSVC with Visual Studio 2026 or 2022 tested), vcpkg, and the CUDA Toolkit (13.x tested) for the GPU backend. vcpkg installs gtest, nlohmann-json and pcre2 from `vcpkg.json`; CMake downloads minja (header-only, pinned by hash) during configure.

```
cmake --preset default
cmake --build --preset release
cmake --build --preset debug
ctest --preset debug
```

With Visual Studio 2022, use `vs2022` and `vs2022-release`/`vs2022-debug` instead.

The CUDA backend is on by default and compiles for the GPU in the build machine (`-arch=native`). To build without it, configure with `-DIE_ENABLE_CUDA=OFF`.

## Run

`main.cpp` runs Llama 3.2 1B Instruct as a chat. Put these files from [`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) (an ungated copy of `meta-llama/Llama-3.2-1B-Instruct`) in `Models/llama3.2-instruct/`:

- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json` (holds the chat template)

Then run `build-vs/Release/inference.exe` from the repository root, so the relative paths resolve.

The `device` constant in `main.cpp` picks the device, and is set to `tensor::Device::CUDA`. Loading peaks at about 10 GB of host memory. On the GPU, the fp32 weights take about 4.9 GB of device memory; if allocation fails, lower `contextSize`. On the CPU, the first reply takes many minutes.

## Roadmap

**Performance** — the current bottleneck. On the CPU, `MatMul` reaches about 0.075 GFLOP/s on Llama 3.2 1B's projections, so a short chat prompt takes roughly 20+ minutes to the first token. The GPU path is correct but its kernels are not yet optimised.
- [x] KV cache — reuse past keys/values so each step processes one token instead of the whole sequence
- [ ] `Linear` kernel — `Linear` multiplies by a transposed (strided) weight view, so the inner loop jumps 8–32 KB per step and cannot vectorise; a contiguous row-dot-product kernel is the single biggest win
- [ ] Multithreading — `MatMul` runs on one core
- [ ] SIMD (AVX2/FMA, AVX-512), blocked/cache-aware `MatMul`, fused attention
- [ ] Last-position-only output projection — prefill currently computes the 128K-vocabulary logits for every prompt position (about 20% of Llama prefill) and uses only the last row
- [ ] Prefix caching — keep the KV cache across chat turns and encode only the new turn instead of the whole history
- [ ] Memory — load weights straight into parameters (loading currently peaks at about 2× model size), memory-mapped weights, arena allocation, no allocation in the hot loop
- [ ] Quantization — int8/int4 weights, FP16 storage and full BF16/FP16 compute paths
- [x] CUDA backend — every op of the forward pass, including attention, on the GPU, with a caching allocator
- [ ] FlashAttention — fused, tiled attention that never builds the score matrix or a mask tensor, then flash-decoding (split the KV sequence across blocks for single-token steps)
- [ ] Tiled shared-memory `MatMul`/`Linear` kernels, or cuBLAS
- [ ] BF16 weights on the GPU — store `Linear` and embedding weights as `__nv_bfloat16` and accumulate in fp32, halving Llama 3.2 1B to about 2.5 GB; needs a mixed-type parameter registry in `Module`
- [ ] Skip the memset when a kernel overwrites its whole output
- [ ] Fused RMSNorm/LayerNorm CUDA kernels — one block per row reusing `BlockReduce`; replaces the ~6 launches and 3 full-size temporaries of the composed-op path (about 7N memory traffic down to 2–3N)

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
