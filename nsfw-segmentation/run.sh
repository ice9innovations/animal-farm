#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Error: $SCRIPT_DIR/.env is missing. Run install.sh or install_jetson.sh first." >&2
    exit 1
fi
if [ ! -x "$SCRIPT_DIR/venv/bin/python" ]; then
    echo "Error: service virtualenv is missing. Run the appropriate installer first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-nsfw-segmentation}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics-nsfw-segmentation}"
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"

# Desktop PyTorch wheels keep CUDA libraries below site-packages/nvidia.
NVIDIA_SITE_LIBS="$(find "$SCRIPT_DIR/venv/lib" -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | paste -sd: -)"
if [ -n "$NVIDIA_SITE_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_SITE_LIBS:${LD_LIBRARY_PATH:-}"
fi

cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/venv/bin/python" REST.py
