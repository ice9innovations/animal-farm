#!/bin/bash
# Install pose detection service for Nvidia Jetson Orin (JetPack 6, CUDA 12.6, TRT 10.3).
#
# Differences from install.sh:
#   - Pins numpy<2 (Jetson ONNX wheel requires NumPy 1.x ABI)
#   - Uses opencv-python 4.9 (opencv 4.12 requires numpy>=2)
#   - Uses requirements-jetson.txt to avoid desktop MediaPipe/ONNX Runtime pins
#   - Installs onnxruntime-gpu from Jetson index (PyPI wheel lacks nvgpu support)
#   - Generates a systemd service for this checkout path
#
# Usage:
#   bash install_jetson.sh
#
# After install:
#   sudo systemctl start pose-api
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/venv"
SERVICE_SRC="$SCRIPT_DIR/pose-api.service"
CURRENT_USER="$(whoami)"
JETSON_ORT_INDEX="${JETSON_ORT_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
JETSON_ORT_PACKAGE="${JETSON_ORT_PACKAGE:-onnxruntime-gpu}"

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

rm -rf "$VENV"
python3 -m venv "$VENV"

"$VENV/bin/pip" install --upgrade pip

# Install service requirements without the desktop onnxruntime / mediapipe pins.
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements-jetson.txt"

# Install Jetson GPU ONNX Runtime separately. Leave the package unpinned by default
# because available versions vary by JetPack image and Python minor version.
"$VENV/bin/pip" uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
echo "Installing Jetson ONNX Runtime package: $JETSON_ORT_PACKAGE"
echo "Using Jetson ONNX Runtime index: $JETSON_ORT_INDEX"
"$VENV/bin/pip" install --no-cache-dir "$JETSON_ORT_PACKAGE" \
    --index-url "$JETSON_ORT_INDEX"

"$VENV/bin/python" - <<'PY'
import onnxruntime as ort
import subprocess
import sys

providers = set(ort.get_available_providers())
required = {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
print("ONNX Runtime version:", ort.__version__)
print("ONNX Runtime providers:", ", ".join(sorted(providers)))
subprocess.run(
    [sys.executable, "-m", "pip", "show", "onnxruntime", "onnxruntime-gpu"],
    check=False,
)
if not providers.intersection(required):
    raise SystemExit(
        "Jetson ONNX Runtime GPU provider check failed. "
        "This venv is still using a CPU-only ONNX Runtime build. "
        "Set JETSON_ORT_PACKAGE/JETSON_ORT_INDEX to a JetPack-matched "
        "onnxruntime-gpu wheel before rerunning install_jetson.sh."
    )
PY

echo ""
echo "venv ready."

set_env_value "USE_GPU" "true"
set_env_value "REQUIRE_GPU" "true"
set_env_value "ONNX_PROVIDER_ORDER" "cuda,tensorrt,cpu"
cat > "$SERVICE_SRC" <<EOF
[Unit]
Description=Pose Estimation REST API Service
After=network.target
StartLimitBurst=3
StartLimitIntervalSec=300

[Service]
Type=simple
Restart=always
RestartSec=5
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/run.sh
EnvironmentFile=$SCRIPT_DIR/.env
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

if [ "$(id -u)" = "0" ]; then
    cp "$SERVICE_SRC" /etc/systemd/system/pose-api.service
    systemctl daemon-reload
    echo "Service installed. Run: systemctl start pose-api"
else
    echo "To install the service, run:"
    echo "  sudo cp $SERVICE_SRC /etc/systemd/system/pose-api.service"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl start pose-api"
fi
