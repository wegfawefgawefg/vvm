#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

url="${1:-https://www.youtube.com/watch?v=2Od-X7FQT2Q}"
frames_dir="${2:-resources/video/ricardo-28x28}"
fps="${3:-12}"
work_dir="resources/video/ricardo-source"

command -v ffmpeg >/dev/null
if [[ -x ".cache/yt-dlp-venv/bin/yt-dlp" ]]; then
  ytdlp=".cache/yt-dlp-venv/bin/yt-dlp"
else
  command -v yt-dlp >/dev/null
  ytdlp="yt-dlp"
fi

mkdir -p "$work_dir"
rm -rf "$frames_dir"
mkdir -p "$frames_dir"

"$ytdlp" \
  -f 'bv*[height<=480]+ba/b[height<=480]/b' \
  --merge-output-format mp4 \
  -o "$work_dir/source.%(ext)s" \
  "$url"

source_video="$(find "$work_dir" -maxdepth 1 -type f -name 'source.*' | sort | tail -n 1)"

ffmpeg \
  -y \
  -i "$source_video" \
  -vf "fps=${fps},scale=28:28:force_original_aspect_ratio=increase,crop=28:28,format=gray" \
  "$frames_dir/frame_%05d.pgm"

count="$(find "$frames_dir" -maxdepth 1 -type f -name '*.pgm' | wc -l)"
echo "wrote $count frames to $frames_dir"
