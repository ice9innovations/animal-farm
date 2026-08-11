#!/bin/bash
# Install the Face service with the BlazeFace ONNX CPU backend only.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/venv"
FACE_MODEL_PATH="${FACE_MODEL_PATH:-$SCRIPT_DIR/../models/face/blaze.onnx}"

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

rm -rf "$VENV"
python3 -m venv "$VENV"

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements-onnx.txt"

set_env_value "FACE_BACKEND" "onnx"
set_env_value "FACE_MODEL_PATH" "$(realpath -m "$FACE_MODEL_PATH")"
set_env_value "USE_GPU" "false"
set_env_value "REQUIRE_GPU" "false"
if [ ! -f "$FACE_MODEL_PATH" ]; then
    FACE_MODEL_PATH="$FACE_MODEL_PATH" "$SCRIPT_DIR/download_model.sh"
fi

echo ""
echo "ONNX CPU venv ready."
