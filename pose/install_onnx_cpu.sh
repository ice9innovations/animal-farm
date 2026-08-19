#!/bin/bash
# Install the Pose service with the BlazePose ONNX CPU backend only.
# Use this on CPU-only hosts such as Raspberry Pi, or as a portable fallback.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/venv"
POSE_DETECTION_MODEL="${POSE_DETECTION_MODEL:-$SCRIPT_DIR/../models/pose/pose_detection.onnx}"
POSE_LANDMARK_MODEL="${POSE_LANDMARK_MODEL:-$SCRIPT_DIR/../models/pose/pose_landmark_heavy.onnx}"

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

rm -rf "$VENV"
"$PYTHON_BIN" -m venv "$VENV"

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements-onnx.txt"

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

set_env_value "PORT" "7786"
set_env_value "PRIVATE" "false"
set_env_value "AUTO_UPDATE" "true"
set_env_value "TIMEOUT" "2.0"
set_env_value "POSE_BACKEND" "onnx"
set_env_value "POSE_DETECTION_MODEL" "$(realpath -m "$POSE_DETECTION_MODEL")"
set_env_value "POSE_LANDMARK_MODEL" "$(realpath -m "$POSE_LANDMARK_MODEL")"
set_env_value "USE_GPU" "false"
set_env_value "ONNX_PROVIDER_ORDER" "cpu"

echo ""
echo "ONNX CPU venv ready."
