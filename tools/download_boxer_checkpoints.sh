#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="$ROOT/models/boxer/ckpts"
BASE_URL="https://huggingface.co/facebook/boxer/resolve/main"
mkdir -p "$CKPT_DIR"

for file in \
  boxernet_hw960in2x6d768-c88128f8.ckpt \
  dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth \
  owlv2-base-patch16-ensemble.pt; do
  if [[ -s "$CKPT_DIR/$file" ]]; then
    echo "Already downloaded: $file"
    continue
  fi
  echo "Downloading Boxer checkpoint: $file"
  curl --fail --location --retry 3 --output "$CKPT_DIR/$file.part" "$BASE_URL/$file"
  mv "$CKPT_DIR/$file.part" "$CKPT_DIR/$file"
done

echo "Boxer checkpoints are in $CKPT_DIR"
