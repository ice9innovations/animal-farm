#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

set -a
source "$SCRIPT_DIR/.env"
set +a

cd "$SCRIPT_DIR"
if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    "$SCRIPT_DIR/venv/bin/python" REST.py
elif [ -x "$SCRIPT_DIR/pose_venv/bin/python" ]; then
    "$SCRIPT_DIR/pose_venv/bin/python" REST.py
else
    echo "No Pose virtualenv found. Run install.sh, install_onnx_cpu.sh, or install_jetson.sh first." >&2
    exit 1
fi
