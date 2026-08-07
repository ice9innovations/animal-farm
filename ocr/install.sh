#!/bin/bash
# Install EasyOCR Flask API dependencies into ocr/venv for desktop/server CUDA.
# Run enable_gpu_desktop.sh first so the venv has CUDA-enabled PyTorch.
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
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

export TMPDIR="${TMPDIR:-$WORKSPACE_DIR/tmp}"
mkdir -p "$TMPDIR"

if [ ! -x "$SCRIPT_DIR/venv/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$SCRIPT_DIR/venv"
fi

"$SCRIPT_DIR/venv/bin/pip" install --upgrade pip
"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

if ! "$SCRIPT_DIR/venv/bin/python" -c "import torch, torchvision; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    cat >&2 <<'EOF'
CUDA-enabled PyTorch is required before installing OCR.

Desktop NVIDIA example:
  bash enable_gpu_desktop.sh

Jetson:
  Use install_jetson.sh instead. It uses system Torch from JetPack.
EOF
    exit 1
fi

"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir --no-deps easyocr==1.7.2

"$SCRIPT_DIR/venv/bin/python" -c "import easyocr, torch; assert torch.cuda.is_available(); print('EasyOCR GPU dependency check passed')"

"$SCRIPT_DIR/venv/bin/python" -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

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
