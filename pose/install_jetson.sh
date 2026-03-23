#!/bin/bash
# Install pose detection service for Nvidia Jetson Orin (JetPack 6, CUDA 12.6, TRT 10.3).
#
# Differences from install.sh:
#   - Uses pose_venv (not venv) to match pose.sh
#   - Pins numpy<2 (Jetson ONNX wheel requires NumPy 1.x ABI)
#   - Uses opencv-python 4.9 (opencv 4.12 requires numpy>=2)
#   - Installs onnxruntime-gpu from Jetson index (PyPI wheel lacks nvgpu support)
#   - Copies pre-configured services/pose-api.service instead of generating one
#
# Usage:
#   bash install_jetson.sh
#
# After install:
#   sudo systemctl start pose-api
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/pose_venv"
SERVICE_SRC="$SCRIPT_DIR/services/pose-api.service"

rm -rf "$VENV"
python3 -m venv "$VENV"

"$VENV/bin/pip" install --upgrade pip

# Install requirements (already pins numpy<2 and opencv-python==4.9.0.80)
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

# Replace PyPI onnxruntime with Jetson GPU wheel (supports nvgpu / TensorRT)
"$VENV/bin/pip" uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
"$VENV/bin/pip" install --no-cache-dir onnxruntime-gpu \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

echo ""
echo "pose_venv ready."

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
