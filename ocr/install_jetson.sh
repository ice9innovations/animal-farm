#!/bin/bash
# Install OCR service for Nvidia Jetson Orin / JetPack.
#
# Differences from install.sh:
#   - Uses Python 3.10 by default, matching JetPack 6
#   - Creates ocr/venv with --system-site-packages
#   - Uses the system NVIDIA Torch/torchvision install
#   - Installs EasyOCR with --no-deps so pip does not replace Torch
#   - Verifies CUDA before generating the service file
#
# Usage:
#   bash install_jetson.sh
set -eu

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="ocr"
VENV="$SCRIPT_DIR/venv"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
SERVICE_USER="${SUDO_USER:-$(id -un)}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "Error: install_jetson.sh requires an aarch64 Jetson system." >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN was not found. JetPack 6 should provide Python 3.10." >&2
    echo "Set PYTHON_BIN=/path/to/python3.10 if needed." >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"Error: install_jetson.sh expects Python 3.10 on JetPack 6; "
        f"got Python {sys.version_info.major}.{sys.version_info.minor}"
    )
PY

rm -rf "$VENV"
"$PYTHON_BIN" -m venv --system-site-packages "$VENV"

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

"$VENV/bin/python" - <<'PY'
import torch
import torchvision

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Error: system Torch is not CUDA-enabled or CUDA is not visible.")
print("CUDA device:", torch.cuda.get_device_name(0))
PY

"$VENV/bin/python" -m pip install --no-cache-dir --no-deps easyocr==1.7.2

"$VENV/bin/python" - <<'PY'
import easyocr
import torch

if not torch.cuda.is_available():
    raise SystemExit("Error: CUDA disappeared after EasyOCR install.")
print("EasyOCR GPU dependency check passed")
PY

"$VENV/bin/python" -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm EasyOCR GPU Text Extraction Service
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$SCRIPT_DIR
EnvironmentFile=$SCRIPT_DIR/.env
ExecStart=$SCRIPT_DIR/run.sh
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "Generated $SERVICE_FILE for $SERVICE_USER:$SERVICE_GROUP"

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
