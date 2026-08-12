#!/bin/bash
# Install yolov8 Flask API dependencies on Jetson.
# Uses python3 and the system torch/torchvision packages from the JetPack image.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="yolov8"
CURRENT_USER="$(whoami)"
VENV="$SCRIPT_DIR/venv"

set_env_value() {
    local key="$1"
    local value="$2"
    local env_file="$SCRIPT_DIR/.env"
    if [ ! -f "$env_file" ]; then
        : > "$env_file"
    fi
    if grep -q "^$key=" "$env_file"; then
        sed -i "s|^$key=.*|$key=$value|" "$env_file"
    else
        echo "$key=$value" >> "$env_file"
    fi
}

rm -rf "$VENV"
python3 -m venv --system-site-packages "$VENV"

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/python" - <<'PY'
import torch
import torchvision

print("System torch:", torch.__version__)
print("System torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

set_env_value "PORT" "7773"
set_env_value "PRIVATE" "false"
set_env_value "AUTO_UPDATE" "true"
set_env_value "TIMEOUT" "2.0"
set_env_value "YOLO_MODEL_PATH" "yolov8l.pt"
set_env_value "CONFIDENCE_THRESHOLD" "0.25"

SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm YOLOv8 Object Detection Service
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
