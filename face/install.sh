#!/bin/bash
# Install face detection Flask API dependencies into face/venv.
# Run once before first use. Requires Python 3.11.
#
# Usage:
#   bash install.sh
#
# Optional override:
#   FACE_ORT_PACKAGE=onnxruntime-gpu==1.22.1 bash install.sh
#
# After install, start the service with:
#   bash run.sh  (RunPod)
#   systemctl start face  (systemd)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="face"
CURRENT_USER="$(whoami)"
FACE_ORT_PACKAGE="${FACE_ORT_PACKAGE:-onnxruntime-gpu==1.22.1}"
FACE_MODEL_PATH="${FACE_MODEL_PATH:-$SCRIPT_DIR/../models/face/blaze.onnx}"

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

rm -rf "$SCRIPT_DIR/venv"
python3.11 -m venv "$SCRIPT_DIR/venv"

"$SCRIPT_DIR/venv/bin/pip" install --upgrade pip
"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"
"$SCRIPT_DIR/venv/bin/pip" uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir "$FACE_ORT_PACKAGE"

"$SCRIPT_DIR/venv/bin/python" - <<'PY'
try:
    import onnxruntime as ort
except ImportError:
    print("ONNX Runtime not installed")
else:
    print("ONNX Runtime version:", ort.__version__)
    print("ONNX Runtime providers:", ", ".join(ort.get_available_providers()))
PY

set_env_value "PORT" "7772"
set_env_value "PRIVATE" "false"
set_env_value "AUTO_UPDATE" "true"
set_env_value "TIMEOUT" "2.0"
set_env_value "FACE_BACKEND" "onnx"
set_env_value "FACE_MODEL_PATH" "$(realpath -m "$FACE_MODEL_PATH")"
set_env_value "USE_GPU" "true"
set_env_value "ONNX_PROVIDER_ORDER" "cuda,cpu"
if [ ! -f "$FACE_MODEL_PATH" ]; then
    FACE_MODEL_PATH="$FACE_MODEL_PATH" "$SCRIPT_DIR/download_model.sh"
fi

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm Face Detection Service
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
