#!/bin/bash
# Install rembg/BEN2 Flask API dependencies into rembg/venv.
# Run once before first use. Requires Python 3.11.
#
# Two backends available (set BACKEND in .env):
#   rembg  — CPU/ONNX, no GPU required
#   ben2   — GPU/PyTorch, requires CUDA
#
# Usage:
#   bash install.sh
#
# After install, start the service with:
#   bash run.sh  (RunPod)
#   systemctl start rembg  (systemd)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="rembg"
CURRENT_USER="$(whoami)"

export TMPDIR=/workspace/tmp
mkdir -p "$TMPDIR"

# Read BACKEND from .env to download only what's needed
BACKEND="all"
if [ -f "$SCRIPT_DIR/.env" ]; then
    _backend=$(grep -s '^BACKEND=' "$SCRIPT_DIR/.env" | cut -d= -f2 | tr -d ' \r')
    [ -n "$_backend" ] && BACKEND="$_backend"
fi

# Clone BEN2 model code to network volume
BEN2_CODE_DIR="/workspace/rembg/BEN2"
if [ ! -f "$BEN2_CODE_DIR/BEN2.py" ]; then
    echo "Cloning BEN2 model code to $BEN2_CODE_DIR..."
    mkdir -p "$(dirname "$BEN2_CODE_DIR")"
    git clone https://huggingface.co/PramaLLC/BEN2 "$BEN2_CODE_DIR"
else
    echo "BEN2 code already at $BEN2_CODE_DIR — skipping clone."
fi

# Download model weights
bash "$SCRIPT_DIR/download_models.sh" /workspace/rembg/models "$BACKEND"

rm -rf "$SCRIPT_DIR/venv"
python3.11 -m venv "$SCRIPT_DIR/venv"

"$SCRIPT_DIR/venv/bin/pip" install --upgrade pip
"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm Background Removal Service
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
