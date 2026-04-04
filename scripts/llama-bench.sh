#!/usr/bin/env bash

export LLAMA_SET_ROWS=1
export GGML_CUDA_GRAPH_OPT=1

llama-bench \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL \
    --numa distribute \
    -ngl 99 \
    -b 128,256,512 -ub 128 \
    -fa 1 \
    -p 512 -n 128 \
    -r 3 -o md
