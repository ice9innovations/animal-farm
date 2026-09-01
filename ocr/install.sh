#!/bin/bash
# Single-command, platform-aware installer for the OCR Flask API.
#
# Detects Jetson (aarch64) vs. desktop/server NVIDIA GPU vs. CPU-only,
# installs the matching PyTorch build (delegating to install_jetson.sh or
# enable_gpu_desktop.sh as needed), then installs EasyOCR and the rest of
# the service's dependencies, and generates the systemd service file.
#
# Usage:
#   bash install.sh
#
# After install, start the service with:
#   bash run.sh  (RunPod)
#   systemctl start ocr  (systemd)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="ocr"
CURRENT_USER="$(whoami)"
VENV="$SCRIPT_DIR/venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if [ -z "${WORKSPACE_DIR:-}" ]; then
    if [ -d /workspace ] && [ -w /workspace ]; then
        WORKSPACE_DIR=/workspace
    else
        WORKSPACE_DIR="$SCRIPT_DIR/.workspace"
    fi
fi

export TMPDIR="${TMPDIR:-$WORKSPACE_DIR/tmp}"
mkdir -p "$TMPDIR"

# --- Platform detection ---------------------------------------------------
# /etc/nv_tegra_release is the canonical JetPack/L4T marker and is present
# only on Jetson hardware — aarch64 alone also matches non-Jetson boards
# (e.g. a Raspberry Pi), which have no CUDA and must go through the
# CPU-only path instead.
if [ -e /etc/nv_tegra_release ]; then
    PLATFORM="jetson"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    PLATFORM="gpu"
else
    PLATFORM="cpu"
fi
echo "Detected platform: $PLATFORM"

# Jetson has its own venv layout (--system-site-packages, Python 3.10) and a
# complete install flow already, so hand off to it entirely instead of
# duplicating it here.
if [ "$PLATFORM" = "jetson" ]; then
    exec bash "$SCRIPT_DIR/install_jetson.sh"
fi

# --- Recreate the venv if it's missing or broken --------------------------
venv_is_valid() {
    [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import sys" >/dev/null 2>&1
}

if [ -e "$VENV" ] && ! venv_is_valid; then
    echo "Existing venv at $VENV is broken; removing it."
    rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip

# --- Platform-specific PyTorch ---------------------------------------------
torch_cuda_ok() {
    "$VENV/bin/python" -c "import torch, torchvision; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1
}

if [ "$PLATFORM" = "gpu" ]; then
    if ! torch_cuda_ok; then
        echo "Installing CUDA-enabled PyTorch for desktop/server NVIDIA GPU..."
        bash "$SCRIPT_DIR/enable_gpu_desktop.sh"
    fi
else
    if ! "$VENV/bin/python" -c "import torch, torchvision" >/dev/null 2>&1; then
        echo "No usable NVIDIA GPU detected; installing CPU-only PyTorch..."
        "$VENV/bin/pip" install --no-cache-dir torch torchvision
    fi
fi

# --- Remaining dependencies -------------------------------------------------
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

if [ "$PLATFORM" = "gpu" ] && ! torch_cuda_ok; then
    cat >&2 <<'EOF'
Error: CUDA-enabled PyTorch is required on this platform but is not
available after installation. Check the enable_gpu_desktop.sh output above
for the failure.
EOF
    exit 1
fi

"$VENV/bin/pip" install --no-cache-dir --no-deps easyocr==1.7.2

"$VENV/bin/python" -c "import easyocr, torch; print('EasyOCR dependency check passed (CUDA available:', torch.cuda.is_available(), ')')"

"$VENV/bin/python" -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm EasyOCR GPU Text Extraction Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
EnvironmentFile=$SCRIPT_DIR/.env
ExecStart=$SCRIPT_DIR/run.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "Generated $SERVICE_FILE"

if [ "$(id -u)" = "0" ]; then
    cp "$SERVICE_FILE" /etc/systemd/system/
    systemctl daemon-reload
    echo "Service installed. Run: systemctl start $SERVICE_NAME"
else
    echo ""
    echo "To install the service, run:"
    echo "  sudo cp $SERVICE_FILE /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl start $SERVICE_NAME"
fi

if [ "$PLATFORM" = "cpu" ]; then
    echo ""
    echo "No usable NVIDIA GPU was detected; installed CPU-only PyTorch."
    echo "Set USE_GPU=false in .env to run this service in CPU mode."
fi
