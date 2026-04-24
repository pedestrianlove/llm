#!/usr/bin/env bash

export LLAMA_SET_ROWS=1
export GGML_CUDA_GRAPH_OPT=1

# Qwen3.5-27B Q4_K_XL on 4x GTX 1080 Ti (44 GiB VRAM)
# Model: 16.4 GiB — leaves ~28 GiB for KV cache
# Max context tested: 131K @ 145 t/s pp, 13.6 t/s tg
# Layer split optimal for long context; row split gives better tg but OOMs past 32K
llama-server \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL \
    --ctx-size 131072 \
    -ngl 99 \
    -sm layer \
    -b 2048 -ub 128 \
    -fa on \
    --numa distribute \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --host 0.0.0.0 \
    --port 30000
