#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "$SCRIPT_DIR/.env"

if [ -n "${NUDENET_VENV:-}" ]; then
    VENV="$NUDENET_VENV"
elif [ "$(uname -m)" = "aarch64" ] && [ -x "$SCRIPT_DIR/nudenet_venv/bin/python" ]; then
    VENV="$SCRIPT_DIR/nudenet_venv"
elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    VENV="$SCRIPT_DIR/venv"
elif [ -x "$SCRIPT_DIR/nudenet_venv/bin/python" ]; then
    VENV="$SCRIPT_DIR/nudenet_venv"
else
    echo "Error: no NudeNet virtualenv found." >&2
    echo "Run ./install_jetson.sh on Jetson or ./install.sh on other systems." >&2
    exit 1
fi

PYTHON="$VENV/bin/python"
SITE_PACKAGES="$("$PYTHON" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

CUDA_LIB_ROOT="$SITE_PACKAGES/nvidia"
CUDNN_LIB_DIR="$CUDA_LIB_ROOT/cudnn/lib"
CUBLAS_LIB_DIR="$CUDA_LIB_ROOT/cublas/lib"
CUDA_NVRTC_LIB_DIR="$CUDA_LIB_ROOT/cuda_nvrtc/lib"

# onnxruntime-gpu looks up generic sonames like libcudnn.so / libcublas.so.
# The NVIDIA wheels provide versioned files, so add the expected symlinks here.
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
"$PYTHON" REST.py
