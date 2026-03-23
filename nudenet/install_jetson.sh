#!/bin/bash
# Install nudenet service for Nvidia Jetson Orin (JetPack 6, CUDA 12.6, TRT 10.3).
#
# Differences from install.sh:
#   - Uses nudenet_venv (not venv) to match nudenet.sh
#   - Skips onnxruntime-gpu from requirements (nudenet pulls in CPU onnxruntime as dep)
#   - Installs onnxruntime-gpu from Jetson index after, then pins numpy<2
#   - Removes TMPDIR override (RunPod-specific, not needed on Jetson)
#   - Copies pre-configured services/nudenet-api.service instead of generating one
#
# Usage:
#   bash install_jetson.sh
#
# After install:
#   sudo systemctl start nudenet-api
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/nudenet_venv"
SERVICE_SRC="$SCRIPT_DIR/services/nudenet-api.service"

rm -rf "$VENV"
python3 -m venv "$VENV"

"$VENV/bin/pip" install --upgrade pip

# Install requirements (excluding onnxruntime-gpu — handled below)
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

# nudenet pulls in CPU-only onnxruntime as a transitive dep — remove it,
# then install the Jetson GPU wheel (nvgpu / TensorRT support).
# Must also pin numpy<2 after, since the Jetson wheel is compiled against NumPy 1.x ABI.
"$VENV/bin/pip" uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
"$VENV/bin/pip" install --no-cache-dir onnxruntime-gpu \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
"$VENV/bin/pip" install --no-cache-dir "numpy<2"

echo ""
echo "nudenet_venv ready."

if [ "$(id -u)" = "0" ]; then
    cp "$SERVICE_SRC" /etc/systemd/system/nudenet-api.service
    systemctl daemon-reload
    echo "Service installed. Run: systemctl start nudenet-api"
else
    echo "To install the service, run:"
    echo "  sudo cp $SERVICE_SRC /etc/systemd/system/nudenet-api.service"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl start nudenet-api"
fi
