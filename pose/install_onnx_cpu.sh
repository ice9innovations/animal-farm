#!/bin/bash
# Install the Pose service with the BlazePose ONNX CPU backend only.
# Use this on CPU-only hosts such as Raspberry Pi, or as a portable fallback.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/venv"

rm -rf "$VENV"
python3 -m venv "$VENV"

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements-onnx.txt"

echo ""
echo "ONNX CPU venv ready."
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

set_env_value "USE_GPU" "false"
set_env_value "REQUIRE_GPU" "false"
echo "Set POSE_BACKEND=onnx, USE_GPU=false, and REQUIRE_GPU=false in $SCRIPT_DIR/.env before starting."
