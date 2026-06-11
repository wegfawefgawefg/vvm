#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

frames_dir="${VVM_VIDEO_FRAMES:-resources/video/ricardo-28x28}"

if [[ ! -d "$frames_dir" ]] || ! find "$frames_dir" -maxdepth 1 -name '*.pgm' -print -quit | grep -q .; then
  cat <<EOF
Missing Ricardo frame directory:
  $frames_dir

Prepare it with:
  ./scripts/fetch_ricardo_frames.sh
EOF
  exit 1
fi

./build-sdl3/vvm visualize-live-train \
  --task video-next \
  --video-frames-dir "$frames_dir" \
  --state-dim 1568 \
  --ops 256 \
  --candidates 16 \
  --sample-candidates 0 \
  --hard-retrieval \
  --activation leaky-relu \
  --activation-leak 0.05 \
  --sample-frames 4 \
  --window 16 \
  --train-samples 2048 \
  --test-samples 256 \
  --lr 0.006 \
  --momentum 0.9 \
  --recency-decay 1.0 \
  --max-grad-norm 1.0 \
  --update-scale 2.0 \
  --activation-threshold 0.0 \
  --video-foreground-weight 8.0 \
  --video-input-to-output-scale 1.0 \
  --state-heat 0.0 \
  --rejection-scale 0.0 \
  --hard-refractory-ticks 4
