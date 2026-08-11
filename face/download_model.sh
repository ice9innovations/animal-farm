#!/bin/bash
# Download the BlazeFace ONNX model used by the Face service.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MODEL_PATH="${FACE_MODEL_PATH:-$SCRIPT_DIR/../models/face/blaze.onnx}"
MODEL_URL="${FACE_MODEL_URL:-https://huggingface.co/garavv/blazeface-onnx/resolve/main/blaze.onnx}"

mkdir -p "$(dirname "$MODEL_PATH")"
tmp_path="${MODEL_PATH}.tmp"

curl -fL "$MODEL_URL" -o "$tmp_path"
test -s "$tmp_path"
mv "$tmp_path" "$MODEL_PATH"

echo "Downloaded Face ONNX model to $MODEL_PATH"
