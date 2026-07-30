#!/bin/bash
# Install nudenet service for Nvidia Jetson Orin (JetPack 6, CUDA 12.6, TRT 10.3).
#
# Differences from install.sh:
#   - Uses nudenet_venv (not venv) for Jetson-specific dependencies
#   - Uses Python 3.10, the Python version shipped with JetPack 6
#   - Installs Jetson-compatible NumPy/OpenCV constraints
#   - Installs NudeNet without its CPU-only onnxruntime dependency
#   - Installs onnxruntime-gpu from the Jetson index
#   - Removes TMPDIR override (RunPod-specific, not needed on Jetson)
#   - Generates a systemd service for the invoking user and current install path
#
# Usage:
#   bash install_jetson.sh
#
# After install:
#   sudo systemctl start nudenet-api
set -eu

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/nudenet_venv"
SERVICE_FILE="$SCRIPT_DIR/nudenet-api.service"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
JETSON_INDEX="${JETSON_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
JETSON_ORT_VERSION="${JETSON_ORT_VERSION:-1.23.0}"
SERVICE_USER="${SUDO_USER:-$(id -un)}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "Error: install_jetson.sh requires an aarch64 Jetson system." >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN was not found. JetPack 6 should provide Python 3.10." >&2
    echo "Set PYTHON_BIN=/path/to/python3.10 if it is installed in a non-standard location." >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"Error: install_jetson.sh requires Python 3.10 on JetPack 6; "
        f"got Python {sys.version_info.major}.{sys.version_info.minor}"
    )
PY

rm -rf "$VENV"
if ! "$PYTHON_BIN" -m venv "$VENV"; then
    echo "Error: failed to create virtualenv at $VENV." >&2
    echo "Install the Python 3.10 venv package for JetPack 6, then rerun this script." >&2
    exit 1
fi

if [ ! -x "$VENV/bin/pip" ]; then
    echo "Error: virtualenv was created without pip at $VENV." >&2
    exit 1
fi

"$VENV/bin/pip" install --upgrade pip

# Install the compatible dependency set first. NudeNet is installed without
# dependencies because its metadata requires the CPU "onnxruntime" package.
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements-jetson.txt"
"$VENV/bin/pip" install --no-cache-dir --no-deps "nudenet>=3.4.0"
"$VENV/bin/pip" install --no-cache-dir \
    "onnxruntime-gpu==$JETSON_ORT_VERSION" \
    --no-deps \
    --index-url "$JETSON_INDEX"

"$VENV/bin/python" - <<'PY'
import cv2
import numpy
import onnxruntime as ort
from nudenet import NudeDetector

providers = ort.get_available_providers()
print(f"NumPy {numpy.__version__}; OpenCV {cv2.__version__}; ONNX Runtime {ort.__version__}")
print(f"ONNX Runtime providers: {providers}")
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("Error: the installed ONNX Runtime has no CUDAExecutionProvider")
PY

echo ""
echo "nudenet_venv ready."

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=NudeNet+ REST API Service
After=network.target
StartLimitBurst=3
StartLimitIntervalSec=300

[Service]
Type=simple
Restart=on-failure
RestartSec=5
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/run.sh
EnvironmentFile=$SCRIPT_DIR/.env
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "Generated $SERVICE_FILE for $SERVICE_USER:$SERVICE_GROUP"

if [ "$(id -u)" = "0" ]; then
    cp "$SERVICE_FILE" /etc/systemd/system/nudenet-api.service
    systemctl daemon-reload
    echo "Service installed. Run: systemctl start nudenet-api"
else
    echo "To install the service, run:"
    echo "  sudo cp $SERVICE_FILE /etc/systemd/system/nudenet-api.service"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl start nudenet-api"
fi
