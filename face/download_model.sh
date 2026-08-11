#!/bin/bash
# Download the BlazeFace ONNX model used by the Face service.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MODEL_PATH="${FACE_MODEL_PATH:-$SCRIPT_DIR/../models/face/face_detection_back_256x256_float32.onnx}"

if [ -z "${FACE_MODEL_URL:-}" ]; then
    echo "FACE_MODEL_URL is required." >&2
    echo "Expected model output format: MediaPipe BlazeFace back/full-range 256x256 ONNX." >&2
    echo "Destination: $MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$(dirname "$MODEL_PATH")"
tmp_path="${MODEL_PATH}.tmp"

curl -fL "$FACE_MODEL_URL" -o "$tmp_path"
test -s "$tmp_path"
mv "$tmp_path" "$MODEL_PATH"

echo "Downloaded Face ONNX model to $MODEL_PATH"
