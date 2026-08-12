#!/bin/bash
# Install CUDA-enabled PyTorch for desktop Linux OCR.
# Do not use this on Jetson; install NVIDIA's JetPack-matched PyTorch wheel or
# use an l4t-pytorch container there.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV_PIP="$SCRIPT_DIR/venv/bin/pip"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

if [ ! -x "$VENV_PIP" ]; then
    "$PYTHON_BIN" -m venv "$SCRIPT_DIR/venv"
    "$VENV_PIP" install --upgrade pip
fi

if [ -z "$TORCH_INDEX_URL" ]; then
    CUDA_VERSION="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1\2/p' | head -1)"
    case "$CUDA_VERSION" in
        128|129|13*) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" ;;
        121|122|123|124|125|126|127) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" ;;
        *) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118" ;;
    esac
fi

echo "Installing CUDA PyTorch from: $TORCH_INDEX_URL"
"$VENV_PIP" uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
"$VENV_PIP" install --no-cache-dir \
    --force-reinstall \
    torch torchvision \
    --index-url "$TORCH_INDEX_URL"

"$VENV_PYTHON" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch install failed or no NVIDIA GPU is visible.")
print("CUDA device:", torch.cuda.get_device_name(0))
PY

echo "CUDA PyTorch is installed. Run: bash $SCRIPT_DIR/install.sh"
