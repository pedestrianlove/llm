#! /usr/bin/env bash

llama-cli \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q8_K_XL \
    --ctx-size 16384 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --numa distribute
