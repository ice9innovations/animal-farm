#!/bin/bash
# Install the Face service with the BlazeFace ONNX CPU backend only.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/venv"
FACE_MODEL_PATH="${FACE_MODEL_PATH:-$SCRIPT_DIR/../models/face/blaze.onnx}"

if [ -z "${PYTHON_BIN:-}" ]; then
    for candidate in python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi

if [ -z "${PYTHON_BIN:-}" ] || ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "install_onnx_cpu.sh requires Python >= 3.10. Set PYTHON_BIN=/path/to/python3.10+ if needed." >&2
    exit 1
fi

set_env_value() {
    local key="$1"
    local value="$2"
    local env_file="$SCRIPT_DIR/.env"
    if [ ! -f "$env_file" ]; then
        : > "$env_file"
    fi
    if grep -q "^$key=" "$env_file"; then
        sed -i "s|^$key=.*|$key=$value|" "$env_file"
    else
        echo "$key=$value" >> "$env_file"
    fi
}

rm -rf "$VENV"
"$PYTHON_BIN" -m venv "$VENV"

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements-onnx.txt"

set_env_value "PORT" "7772"
set_env_value "PRIVATE" "false"
set_env_value "AUTO_UPDATE" "true"
set_env_value "TIMEOUT" "2.0"
set_env_value "FACE_BACKEND" "onnx"
set_env_value "FACE_MODEL_PATH" "$(realpath -m "$FACE_MODEL_PATH")"
set_env_value "USE_GPU" "false"
set_env_value "ONNX_PROVIDER_ORDER" "cpu"
if [ ! -f "$FACE_MODEL_PATH" ]; then
    FACE_MODEL_PATH="$FACE_MODEL_PATH" "$SCRIPT_DIR/download_model.sh"
fi

echo ""
echo "ONNX CPU venv ready."
