#!/usr/bin/env bash

export LLAMA_SET_ROWS=1
export GGML_CUDA_GRAPH_OPT=1

# Q4_K_XL (~16.4 GiB model, 4x GTX 1080 Ti = 44 GiB VRAM)
# Max tested: 131K ctx @ 145 t/s pp, 65K ctx @ 204 t/s pp
# Layer split — row split OOMs past 32K
llama-bench \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL \
    --numa distribute \
    -ngl 99 \
    -sm layer \
    -b 2048 -ub 128 \
    -fa 1 \
    -p 131072 -n 128 \
    -r 3 -o md

# Q8_K_XL (~33 GiB model) — q4_0 KV cache to fit in VRAM
# Max tested: 8K ctx @ 151 t/s pp (OOMs at 16K)
llama-bench \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q8_K_XL \
    --numa distribute \
    -ngl 99 \
    -sm layer \
    -b 512 -ub 128 \
    -ctk q4_0 -ctv q4_0 \
    -fa 1 \
    -p 8192 -n 128 \
    -r 3 -o md
