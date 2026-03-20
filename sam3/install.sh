#!/bin/bash
# Install SAM3 Flask API dependencies into sam3/venv.
# Run once before first use. Requires Python 3.11.
#
# Prerequisites:
#   export HF_TOKEN=<your-token>  (facebook/sam3 is a restricted model)
#
# Usage:
#   bash install.sh
#
# After install, start the service with:
#   bash run.sh  (RunPod)
#   systemctl start sam3  (systemd)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="sam3"
CURRENT_USER="$(whoami)"

export TMPDIR=/workspace/tmp
mkdir -p "$TMPDIR"

# Clone SAM3 model code to network volume
SAM3_CODE_DIR="/workspace/sam3/sam3"
if [ ! -d "$SAM3_CODE_DIR" ]; then
    echo "Cloning SAM3 model code to $SAM3_CODE_DIR..."
    mkdir -p "$(dirname "$SAM3_CODE_DIR")"
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_CODE_DIR"
else
    echo "SAM3 code already at $SAM3_CODE_DIR — skipping clone."
fi

# Download model checkpoint (requires HF_TOKEN — facebook/sam3 is restricted)
bash "$SCRIPT_DIR/download_models.sh" /workspace/sam3/models

rm -rf "$SCRIPT_DIR/venv"
python3.11 -m venv "$SCRIPT_DIR/venv"

"$SCRIPT_DIR/venv/bin/pip" install --upgrade pip
"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir -e "$SAM3_CODE_DIR[notebooks]"
"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir \
    Flask==3.1.1 \
    flask-cors==6.0.1 \
    python-dotenv \
    requests \
    pillow \
    numpy

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm SAM3 Segmentation Service
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
