#!/bin/bash
# Install Florence-2 service dependencies into florence2/venv.
# Run once before first use. Requires Python 3.11.
#
# Usage:
#   bash install.sh
#
# After install, start the service with:
#   systemctl start florence2
#   (or: bash run.sh)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="florence2"
CURRENT_USER="$(whoami)"

python3.11 -m venv "$SCRIPT_DIR/venv"
source "$SCRIPT_DIR/venv/bin/activate"

# Route pip temp files to network volume — CUDA wheels exhaust the overlay /tmp
export TMPDIR="${TMPDIR:-/workspace/tmp}"
mkdir -p "$TMPDIR"

pip install --upgrade pip

# PyTorch with CUDA 12.8 (confirmed working for Florence-2)
pip install --no-cache-dir \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# All other dependencies (torch already installed above, pip will skip it)
pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm Florence-2 Vision Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
EnvironmentFile=$SCRIPT_DIR/.env
ExecStart=$SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/REST.py
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
