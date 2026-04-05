#!/usr/bin/env bash

export LLAMA_SET_ROWS=1
export GGML_CUDA_GRAPH_OPT=1

# Qwen3.5-27B Q8_K_XL on 4x GTX 1080 Ti (44 GiB VRAM)
# Model: 33 GiB — only ~11 GiB left for KV cache
# Max context tested: 8K @ 151 t/s pp, ~8.5 t/s tg (with q4_0 KV cache)
# q4_0 KV cache required to fit any meaningful context in remaining VRAM
llama-server \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q8_K_XL \
    --ctx-size 8192 \
    -ngl 99 \
    -sm layer \
    -b 512 -ub 128 \
    -ctk q4_0 -ctv q4_0 \
    -fa on \
    --numa distribute \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --host 0.0.0.0 \
    --port 8080
