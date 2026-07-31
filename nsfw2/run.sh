#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "$SCRIPT_DIR/.env"

cd "$SCRIPT_DIR"

if [ "$MODE" = "gpu" ]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    if [ -d "$CUDA_HOME/bin" ]; then
        export PATH="$CUDA_HOME/bin:$PATH"
    fi
    if [ -d "$CUDA_HOME/lib64" ]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    fi
fi

"$SCRIPT_DIR/venv/bin/python" REST.py
