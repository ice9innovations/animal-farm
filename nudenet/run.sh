#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "$SCRIPT_DIR/.env"

# onnxruntime-gpu links against system CUDA libs, not bundled ones.
# Ensure /usr/local/cuda/lib64 is on LD_LIBRARY_PATH so libcublas etc. are found.
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

cd "$SCRIPT_DIR"
"$SCRIPT_DIR/venv/bin/python" REST.py
