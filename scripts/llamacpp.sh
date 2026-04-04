#! /usr/bin/env bash

export LLAMA_SET_ROWS=1
export GGML_CUDA_GRAPH_OPT=1

llama-server \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL \
    --ctx-size 16384 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --flash-attn 1 \
    --numa distribute
