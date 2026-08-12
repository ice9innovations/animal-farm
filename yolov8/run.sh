#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "$SCRIPT_DIR/.env"

cd "$SCRIPT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-yolov8}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${TMPDIR:-/tmp}/ultralytics-yolov8}"
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"

NVIDIA_SITE_LIBS="$(find "$SCRIPT_DIR/venv/lib" "$SCRIPT_DIR/jetson_venv/lib" -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | paste -sd: -)"
if [ -n "$NVIDIA_SITE_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_SITE_LIBS:${LD_LIBRARY_PATH:-}"
fi

"$SCRIPT_DIR/venv/bin/python" REST.py
