#!/usr/bin/env bash

export LLAMA_SET_ROWS=1
export GGML_CUDA_GRAPH_OPT=1

# Qwen3.5-27B dense Q4_K_XL — max context, layer split
llama-bench \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL \
    --numa distribute \
    -ngl 99 \
    -sm layer \
    -b 2048 -ub 128 \
    -fa 1 \
    -p 131072 -n 128 \
    -r 3 -o md

# Qwen3.5-27B dense Q8_K_XL — q4_0 KV cache, 8K max
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

# Qwen3.5-35B-A3B MoE Q8_K_XL — ngl 36 (4 layers on CPU), 16K context
llama-bench \
    -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q8_K_XL \
    --numa distribute \
    -ngl 36 \
    -sm layer \
    -b 2048 -ub 512 \
    -fa 1 \
    -p 16384 -n 128 \
    -r 3 -o md
