#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "$SCRIPT_DIR/.env"

SITE_PACKAGES="$("$SCRIPT_DIR/venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
CUDA_LIB_ROOT="$SITE_PACKAGES/nvidia"
CUDNN_LIB_DIR="$CUDA_LIB_ROOT/cudnn/lib"
CUBLAS_LIB_DIR="$CUDA_LIB_ROOT/cublas/lib"
CUDA_NVRTC_LIB_DIR="$CUDA_LIB_ROOT/cuda_nvrtc/lib"

# Paddle looks up generic sonames like libcudnn.so / libcublas.so at runtime.
# The NVIDIA wheel installs only versioned files, so add the expected symlinks
# before starting the service.
if [ -d "$CUDNN_LIB_DIR" ]; then
    ln -sf libcudnn.so.9 "$CUDNN_LIB_DIR/libcudnn.so"
    ln -sf libcudnn_adv.so.9 "$CUDNN_LIB_DIR/libcudnn_adv.so"
    ln -sf libcudnn_cnn.so.9 "$CUDNN_LIB_DIR/libcudnn_cnn.so"
    ln -sf libcudnn_ops.so.9 "$CUDNN_LIB_DIR/libcudnn_ops.so"
    ln -sf libcudnn_graph.so.9 "$CUDNN_LIB_DIR/libcudnn_graph.so"
    ln -sf libcudnn_heuristic.so.9 "$CUDNN_LIB_DIR/libcudnn_heuristic.so"
    ln -sf libcudnn_engines_runtime_compiled.so.9 "$CUDNN_LIB_DIR/libcudnn_engines_runtime_compiled.so"
    ln -sf libcudnn_engines_precompiled.so.9 "$CUDNN_LIB_DIR/libcudnn_engines_precompiled.so"
fi

if [ -d "$CUBLAS_LIB_DIR" ]; then
    ln -sf libcublas.so.12 "$CUBLAS_LIB_DIR/libcublas.so"
    ln -sf libcublasLt.so.12 "$CUBLAS_LIB_DIR/libcublasLt.so"
    ln -sf libnvblas.so.12 "$CUBLAS_LIB_DIR/libnvblas.so"
fi

export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:$CUBLAS_LIB_DIR:$CUDA_NVRTC_LIB_DIR:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/venv/bin/python" REST.py
