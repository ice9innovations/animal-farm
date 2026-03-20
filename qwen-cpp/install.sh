#!/bin/bash
# Install qwen-cpp Flask API dependencies into qwen-cpp/venv.
# Run once before first use. Requires Python 3.11.
#
# NOTE: This installs only the Flask REST API wrapper.
# The llama-server binary must be compiled separately and available
# on PATH or set via LLAMA_SERVER_BIN in .env.
#
# Usage:
#   bash install.sh
#
# After install, start the service with:
#   bash run.sh  (RunPod)
#   systemctl start qwen-cpp  (systemd)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="qwen-cpp"
CURRENT_USER="$(whoami)"

bash "$(dirname "$(realpath "$0")")/../llama-cpp/build_server.sh"

rm -rf "$SCRIPT_DIR/venv"
python3.11 -m venv "$SCRIPT_DIR/venv"
source "$SCRIPT_DIR/venv/bin/activate"

pip install --upgrade pip
pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

"$SCRIPT_DIR/venv/bin/python" -c "import nltk; nltk.download('punkt')"

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm Qwen-CPP Vision Service
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
