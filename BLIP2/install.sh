#!/bin/bash
# Install BLIP2 service: clones LAVIS and installs all dependencies into BLIP2/venv.
# Run once before first use. Requires Python 3.11.
#
# Usage:
#   bash install.sh
#
# After install, start the service with:
#   systemctl start blip2
#   (or: bash run.sh)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LAVIS_DIR="$SCRIPT_DIR/LAVIS"
SERVICE_NAME="blip2"
CURRENT_USER="$(whoami)"

# Clone LAVIS if not already present
if [ ! -d "$LAVIS_DIR" ]; then
    echo "Cloning LAVIS..."
    git clone https://github.com/salesforce/LAVIS.git "$LAVIS_DIR"
else
    echo "LAVIS already present at $LAVIS_DIR"
fi

python3.11 -m venv "$SCRIPT_DIR/venv"
source "$SCRIPT_DIR/venv/bin/activate"

# Route pip temp files to network volume — CUDA wheels exhaust the overlay /tmp
export TMPDIR="${TMPDIR:-/workspace/tmp}"
mkdir -p "$TMPDIR"

pip install --upgrade pip

# PyTorch with CUDA 11.8 (works on CUDA 12.x hardware)
pip install --no-cache-dir \
    torch==2.7.1+cu118 \
    torchvision==0.22.1+cu118 \
    torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

# Install LAVIS itself (no deps — already installed above)
pip install -e "$LAVIS_DIR" --no-deps

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm BLIP2 Caption Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
Environment=PYTHONPATH=$LAVIS_DIR
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
