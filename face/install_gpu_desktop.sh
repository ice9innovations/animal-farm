#!/bin/bash
# Install the Face service for desktop Nvidia GPUs into face/venv.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
FACE_ORT_PACKAGE="${FACE_ORT_PACKAGE:-onnxruntime-gpu==1.22.1}" "$SCRIPT_DIR/install.sh"
FACE_MODEL_PATH="${FACE_MODEL_PATH:-$SCRIPT_DIR/../models/face/face_detection_back_256x256_float32.onnx}"

set_env_value() {
    local key="$1"
    local value="$2"
    local env_file="$SCRIPT_DIR/.env"
    if [ ! -f "$env_file" ]; then
        return
    fi
    if grep -q "^$key=" "$env_file"; then
        sed -i "s|^$key=.*|$key=$value|" "$env_file"
    else
        echo "$key=$value" >> "$env_file"
    fi
}

set_env_value "FACE_BACKEND" "onnx"
set_env_value "FACE_MODEL_PATH" "$(realpath -m "$FACE_MODEL_PATH")"
set_env_value "USE_GPU" "true"
set_env_value "REQUIRE_GPU" "true"
set_env_value "ONNX_PROVIDER_ORDER" "cuda,tensorrt,cpu"
if [ ! -f "$FACE_MODEL_PATH" ]; then
    echo ""
    echo "Warning: Face ONNX model is missing: $FACE_MODEL_PATH" >&2
    echo "The service will not start with FACE_BACKEND=onnx until this model exists." >&2
fi
