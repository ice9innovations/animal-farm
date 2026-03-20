#!/bin/bash
# Install nsfw2 Flask API dependencies into nsfw2/venv.
# Run once before first use. Requires Python 3.11.
#
# This service runs on CPU only (MODE=cpu in .env).
#
# Usage:
#   bash install.sh
#
# After install, start the service with:
#   bash run.sh  (RunPod)
#   systemctl start nsfw2  (systemd)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="nsfw2"
CURRENT_USER="$(whoami)"

export TMPDIR=/workspace/tmp
mkdir -p "$TMPDIR"

rm -rf "$SCRIPT_DIR/venv"
python3.11 -m venv "$SCRIPT_DIR/venv"

"$SCRIPT_DIR/venv/bin/pip" install --upgrade pip
"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm NSFW2 Detection Service
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
