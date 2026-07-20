#!/usr/bin/env bash
set -euo pipefail

model_dir="${1:-models/gpt2}"
base_url="https://huggingface.co/openai-community/gpt2/resolve/main"

mkdir -p "$model_dir"
curl -L --fail --output "$model_dir/config.json" "$base_url/config.json"
curl -L --fail --output "$model_dir/tokenizer.json" "$base_url/tokenizer.json"
curl -L --fail --output "$model_dir/model.safetensors" "$base_url/model.safetensors"

echo "Downloaded GPT-2 artifacts to $model_dir"
