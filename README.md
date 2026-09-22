# InferenceEngine

A transformer inference engine written from scratch in C++20, with no ML dependencies. It currently loads and runs GPT-2 (124M) from a Hugging Face `.safetensors` checkpoint and generates text.

## Status

Running GPT-2 small end to end: tokenize → forward pass → greedy decode → detokenize.

## Layout

| Directory | Contents |
|---|---|
| `Math/` | `Tensor<T>`: shapes, strides, views (reshape, slice, permute, transpose), broadcasting, `MatMul`, reductions, masked fill |
| `NeuralNet/` | `Module` base (named parameters, submodules), `Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, `Sequential`, activations (ReLU, GELU, Sigmoid, Tanh, Softmax) |
| `NeuralNet/Transformer/` | `MultiHeadAttention` (causal masking, separate q/k/v inputs), `TransformerBlock` (pre-norm, two residuals), `Transformer` (block stack + final norm), `TransformerConfig` |
| `Models/` | `GPT2` model, checkpoint loader, byte-level BPE tokenizer, model config presets |
| `Serialize/` | `.safetensors` reader (F32/F64/F16/BF16/int/bool), reflection-based JSON loading |
| `Reflection/` | `REFLECTABLE` macro: member names and iteration for JSON config structs |
| `tests/` | 574 GoogleTest cases |

## Implemented

- **Tensors** — views and broadcasting without copying; `MatMul` over batched 2-D/3-D operands.
- **Attention** — multi-head, causal masking, explicit masks, separate query/key/value inputs with bottom-right causal alignment (ready for a KV cache).
- **Model** — GPT-2: token + learned position embeddings, 12 pre-norm blocks, final LayerNorm, output projection tied to the token embedding.
- **Checkpoints** — `.safetensors` header parsing and tensor loading with dtype conversion; GPT-2 weight mapping (fused `c_attn` split into q/k/v, Conv1D `[in, out]` → `[out, in]` transposes, LayerNorm `weight`/`bias` → `gamma`/`beta`), with checks that every parameter is filled and every tensor used.
- **Tokenizer** — GPT-2 byte-level BPE: byte↔character mapping, word splitting, rank-ordered merges, per-word cache, exact round-trips over UTF-8 and all byte values.
- **Config** — JSON model configs loaded through compile-time reflection.

## Build

Requires CMake 3.21+, a C++20 compiler (MSVC 19.4+ tested), and vcpkg (gtest, nlohmann-json).

```
cmake --preset vs2022
cmake --build --preset vs2022-release
ctest --preset vs2022-release
```

## Run

Place a GPT-2 checkpoint (`config.json`, `model.safetensors`, `vocab.json`, `merges.txt`) in `Models/gpt2/`, then run `inference` from the repository root.

## Roadmap

**Performance**
- KV cache — reuse past keys/values so each step processes one token instead of the whole sequence
- Last-position-only output projection during generation
- Kernels — blocked/cache-aware `MatMul`, SIMD (AVX2/AVX-512), multithreading, fused attention
- Quantization — int8/int4 weights, BF16/FP16 compute paths
- Memory — memory-mapped weights, arena allocation, no allocation in the hot loop
- GPU backend (CUDA/Vulkan) behind a device abstraction

**Architectures**
- RoPE (rotary position embeddings)
- Grouped-query and multi-query attention
- Gated MLP (SwiGLU) and config-driven norm/activation/placement selection
- Mixture of Experts — router, expert dispatch, top-k gating
- Sliding-window and chunked attention
- Encoder and encoder-decoder support (cross-attention)

**Models**
- Config-driven model builder: one generic block assembled from `TransformerConfig`
- Per-model weight mappers (Qwen2, LLaMA, Mistral) beside the GPT-2 one
- Hugging Face `config.json` → `TransformerConfig` translation per family
- Sharded checkpoints (`model-00001-of-*.safetensors` + index)

**Tokenizers**
- Generic byte-level BPE with pluggable pre-tokenizer rules (GPT-2 and tiktoken-style)
- Special-token matching before splitting; BOS/EOS post-processing
- `tokenizer.json` parsing (covers most modern models in one path)
- SentencePiece (LLaMA-1/2, Mistral, Gemma)
- Chat templates for instruct models

**Generation**

*Decoding* — a composable logit pipeline (penalties → temperature → truncation → sample), so each stage stays independent and testable:
- Greedy search — argmax each step. Implemented, and worth keeping behind a flag: being deterministic, it is the only decoding mode that gives bit-exact regression tests. It is also degenerate on its own, falling into repetition loops because the continuation becomes a fixed function of the prefix.
- Temperature — divide logits by `T` before the softmax; `T < 1` sharpens, `T > 1` flattens
- Top-k sampling — keep the k highest-scoring tokens, renormalise, sample
- Top-p (nucleus) sampling — keep the smallest set whose cumulative probability exceeds p, so the candidate count adapts to how confident the model is
- Min-p sampling — keep tokens within a ratio of the top probability; cheaper than top-p and needs no sort
- Repetition, frequency and presence penalties — discount tokens already generated
- Beam search with length normalisation — for tasks where the likeliest sequence is actually wanted
- Contrastive search — a deterministic alternative to greedy that penalises tokens too similar to the existing context
- Seeded RNG so sampled runs stay reproducible

*Other*
- Stop sequences and streaming output
- Batched generation
- Speculative decoding — draft with a small model, verify in one pass with the large one

**Correctness and tooling**
- Reference-parity harness: compare per-layer activations and logits against Hugging Face
- Benchmarks: tokens/sec, memory, per-layer timing
- CLI with prompt/sampling flags; optional server API
