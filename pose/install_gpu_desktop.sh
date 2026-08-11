#!/bin/bash
# Install the Pose service for desktop Nvidia GPUs into pose/venv.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
POSE_ORT_PACKAGE="${POSE_ORT_PACKAGE:-onnxruntime-gpu==1.22.1}" "$SCRIPT_DIR/install.sh"

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

set_env_value "USE_GPU" "true"
set_env_value "REQUIRE_GPU" "true"
set_env_value "ONNX_PROVIDER_ORDER" "cuda,tensorrt,cpu"
