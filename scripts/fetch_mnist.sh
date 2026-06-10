#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-resources/mnist}"
base_url="https://storage.googleapis.com/cvdf-datasets/mnist"

mkdir -p "$out_dir"

files=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

for file in "${files[@]}"; do
  gz_path="$out_dir/$file"
  raw_path="$out_dir/${file%.gz}"
  if [[ ! -f "$raw_path" ]]; then
    curl -L --fail --retry 3 -o "$gz_path" "$base_url/$file"
    gzip -dc "$gz_path" > "$raw_path"
  fi
done

echo "MNIST IDX files are ready in $out_dir"
