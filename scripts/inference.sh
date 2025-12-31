#!/usr/bin/env bash

#SBATCH --account=MST114546
#SBATCH --partition=gp1d
#SBATCH --job-name=run-inference
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --switches=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --mem=720G
#SBATCH --mail-user=geniusjvm@gmail.com
#SBATCH --mail-type=ALL

# Launch Pixi with the exact Slurm CPU count
pixi run python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 8 \
    --ep-size 8 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
