# Setup LLM inference

- Model: Qwen3.6-27B-UD-Q4_K_XL.

## 1. Install Pixi
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

## 2. Build llama.cpp
```bash
pixi run build
```

## 3. Run inference
```bash
pixi run serve
```
