#!/bin/bash
# Install pose detection Flask API dependencies into pose/venv.
# Run once before first use. Requires Python 3.11.
#
# Usage:
#   bash install.sh
#
# Optional override:
#   POSE_ORT_PACKAGE=onnxruntime-gpu==1.22.1 bash install.sh
#
# After install, start the service with:
#   bash run.sh  (RunPod)
#   systemctl start pose  (systemd)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="pose"
CURRENT_USER="$(whoami)"
POSE_ORT_PACKAGE="${POSE_ORT_PACKAGE:-onnxruntime-gpu==1.22.1}"
POSE_DETECTION_MODEL="${POSE_DETECTION_MODEL:-$SCRIPT_DIR/../models/pose/pose_detection.onnx}"
POSE_LANDMARK_MODEL="${POSE_LANDMARK_MODEL:-$SCRIPT_DIR/../models/pose/pose_landmark_heavy.onnx}"

set_env_value() {
    local key="$1"
    local value="$2"
    local env_file="$SCRIPT_DIR/.env"
    if [ ! -f "$env_file" ]; then
        return
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
"$SCRIPT_DIR/venv/bin/pip" install --no-cache-dir "$POSE_ORT_PACKAGE"

"$SCRIPT_DIR/venv/bin/python" - <<'PY'
import onnxruntime as ort

print("ONNX Runtime version:", ort.__version__)
print("ONNX Runtime providers:", ", ".join(ort.get_available_providers()))
PY

set_env_value "POSE_BACKEND" "onnx"
set_env_value "POSE_DETECTION_MODEL" "$(realpath -m "$POSE_DETECTION_MODEL")"
set_env_value "POSE_LANDMARK_MODEL" "$(realpath -m "$POSE_LANDMARK_MODEL")"
set_env_value "USE_GPU" "true"
set_env_value "REQUIRE_GPU" "true"
set_env_value "ONNX_PROVIDER_ORDER" "cuda,tensorrt,cpu"

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm Pose Detection Service
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
