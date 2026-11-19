# Linear on cuBLAS: before and after

`llama_bench` (bench/llama_bench.cpp). Llama 3.2 1B Instruct, fp32 weights, greedy decoding, 64 decode tokens, median of 3 runs after one warm-up run.
NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), Windows 11, CUDA 13.5, MSVC 19.52, Release build.

- **Before:** `bc521c5`, where `Linear` uses the naive one-thread-per-output kernel
- **After:** `e9e0efe`, where `Linear` uses cuBLAS with the bias accumulated through `beta = 1`

Two runs of each were taken in opposite orders. The table uses the second run; the first "before" run's 16-token prefill (1481 ms) was GPU warm-up.

| prompt tokens | prefill before | prefill after | speed-up | decode before | decode after | speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 366 ms | 49 ms | 7.5× | 126.1 ms/tok (7.9 tok/s) | 39.2 ms/tok (25.5 tok/s) | 3.2× |
| 128 | 2723 ms | 151 ms | 18× | 126.3 ms/tok (7.9 tok/s) | 39.3 ms/tok (25.5 tok/s) | 3.2× |
| 512 | 11020 ms | 852 ms | 13× | 126.4 ms/tok (7.9 tok/s) | 40.4 ms/tok (24.7 tok/s) | 3.1× |

GPU memory held by the allocator: 5.37 GB in both (0.26 GB of it cached for reuse).

Decode is memory-bound. About 4.9 GB of fp32 weights over the 4050's ~192 GB/s puts the floor near 26 ms/token, so decode now runs about 1.5× above that floor, down from about 5×.

Load time (12.6–17.4 s) is dominated by reading the file from disk and depends on the OS file cache, so it is not compared.
