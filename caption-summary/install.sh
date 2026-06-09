#!/bin/bash
# Install caption-summary dependencies and generate a systemd service file.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="caption-summary"
CURRENT_USER="$(whoami)"
VENV_DIR="${CAPTION_SUMMARY_VENV:-$SCRIPT_DIR/venv}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Caption Summary REST API
After=network.target qwen-cpp.service
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=10
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
EnvironmentFile=$SCRIPT_DIR/.env
ExecStart=$SCRIPT_DIR/rest.sh

[Install]
WantedBy=multi-user.target
EOF

echo "Generated $SERVICE_FILE"
echo ""
echo "To install the service, run:"
echo "  sudo cp $SERVICE_FILE /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl restart $SERVICE_NAME"
