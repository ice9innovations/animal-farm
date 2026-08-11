#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Missing $SCRIPT_DIR/.env. Run install.sh, install_jetson.sh, or install_onnx_cpu.sh first." >&2
    exit 1
fi

set -a
source "$SCRIPT_DIR/.env"
set +a

cd "$SCRIPT_DIR"
if [ -n "${FACE_VENV:-}" ] && [ -x "$SCRIPT_DIR/$FACE_VENV/bin/python" ]; then
    "$SCRIPT_DIR/$FACE_VENV/bin/python" REST.py
elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    "$SCRIPT_DIR/venv/bin/python" REST.py
elif [ -x "$SCRIPT_DIR/face_venv/bin/python" ]; then
    "$SCRIPT_DIR/face_venv/bin/python" REST.py
else
    echo "No Face virtualenv found. Run install.sh, install_gpu_desktop.sh, install_onnx_cpu.sh, or install_jetson.sh first." >&2
    exit 1
fi
