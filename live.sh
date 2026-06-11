#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

./build-sdl3/vvm visualize-live-train \
  --task mnist-01 \
  --mnist-dir resources/mnist \
  --state-dim 816 \
  --class-registers 32 \
  --ops 256 \
  --candidates 16 \
  --sample-candidates 0 \
  --hard-retrieval \
  --sample-frames 8 \
  --window 8 \
  --class-start-frame 1 \
  --train-samples 1024 \
  --test-samples 512 \
  --lr 0.00025 \
  --momentum 0.9 \
  --recency-decay 1.0 \
  --max-grad-norm 0.1 \
  --update-scale 0.1 \
  --rejection-scale 0.02 \
  --hard-refractory-ticks 4
