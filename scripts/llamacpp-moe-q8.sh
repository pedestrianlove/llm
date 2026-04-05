#!/usr/bin/env bash

export LLAMA_SET_ROWS=1
export GGML_CUDA_GRAPH_OPT=1

# Qwen3.5-35B-A3B MoE Q8_K_XL (45.33 GiB, 3B active per token)
# 4x GTX 1080 Ti — ngl 36 (4 layers on CPU), 16K context
# pp16K: ~103 t/s | tg: ~16.7 t/s
llama-server \
    -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q8_K_XL \
    --ctx-size 16384 \
    -ngl 36 \
    -sm layer \
    -b 2048 -ub 512 \
    -fa on \
    --numa distribute \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --host 0.0.0.0 \
    --port 8080
