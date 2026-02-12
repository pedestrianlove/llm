#! /usr/bin/env bash

llama-server \
    -hf unsloth/Qwen3-Coder-Next-GGUF:UD-Q4_K_XL \
    --port 8001 \
    --ctx-size 32072 \
    --ubatch-size 4096 \
    --batch-size 4096 \
    --flash-attn on \
    --fit on \
    --seed 3407 \
    --temp 1.0 \
    --top-p 0.95 \
    --min-p 0.01 \
    --top-k 40 \
    --numa distribute --jinja --ctx-checkpoints 128
