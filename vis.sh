#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

checkpoint="${VVM_CHECKPOINT:-artifacts/checkpoints/mnist01_probe_hard_refractory.vvmckpt}"

if [[ ! -f "$checkpoint" ]]; then
  cat <<EOF
Missing checkpoint:
  $checkpoint

Train it first with:

./build-sdl3/vvm train-task \\
  --task mnist-01 \\
  --mnist-dir resources/mnist \\
  --state-dim 816 \\
  --class-registers 32 \\
  --ops 256 \\
  --candidates 16 \\
  --sample-candidates 0 \\
  --hard-retrieval \\
  --sample-frames 8 \\
  --window 8 \\
  --class-start-frame 1 \\
  --epochs 5 \\
  --train-samples 1024 \\
  --test-samples 512 \\
  --lr 0.00025 \\
  --lr-decay 0.65 \\
  --momentum 0.9 \\
  --recency-decay 1.0 \\
  --max-grad-norm 0.1 \\
  --update-scale 2.0 \\
  --rejection-scale 0.02 \\
  --rejection-decay 0.5 \\
  --op-anchor-scale 0.1 \\
  --anchor-to-best \\
  --restore-best \\
  --hard-refractory-ticks 4 \\
  --save-model "$checkpoint"
EOF
  exit 1
fi

./build-sdl3/vvm visualize-probe \
  --load-model "$checkpoint" \
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
  --lr-decay 0.65 \
  --momentum 0.9 \
  --recency-decay 1.0 \
  --max-grad-norm 0.1 \
  --update-scale 2.0 \
  --rejection-scale 0.02 \
  --rejection-decay 0.5 \
  --op-anchor-scale 0.1 \
  --anchor-to-best \
  --restore-best \
  --hard-refractory-ticks 4
