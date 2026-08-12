#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "$SCRIPT_DIR/.env"

cd "$SCRIPT_DIR"

unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

if [ "$MODE" = "gpu" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    if [ -d "$CUDA_HOME/bin" ]; then
        export PATH="$CUDA_HOME/bin:$PATH"
    fi
    NVIDIA_SITE_LIBS="$(find "$SCRIPT_DIR/venv/lib" -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | paste -sd: -)"
    if [ -n "$NVIDIA_SITE_LIBS" ]; then
        export LD_LIBRARY_PATH="$NVIDIA_SITE_LIBS:${LD_LIBRARY_PATH:-}"
    fi
    if [ -d "$CUDA_HOME/lib64" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib64"
    fi
else
    export CUDA_VISIBLE_DEVICES="-1"
fi

"$SCRIPT_DIR/venv/bin/python" REST.py
