#!/usr/bin/env bash
# Install on NVIDIA Jetson with JetPack 6 and its system Torch/Torchvision.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/venv"
SERVICE_NAME="nsfw-segmentation"
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
SERVICE_USER="${SUDO_USER:-$(id -un)}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"

set_env_value() {
    local key="$1" value="$2" env_file="$SCRIPT_DIR/.env"
    touch "$env_file"
    if grep -q "^${key}=" "$env_file"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$env_file"
    else
        printf '%s=%s\n' "$key" "$value" >> "$env_file"
    fi
}

if [ "$(uname -m)" != "aarch64" ]; then
    echo "Error: install_jetson.sh requires an aarch64 NVIDIA Jetson." >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN was not found. JetPack 6 normally provides Python 3.10." >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Error: JetPack 6 installation requires Python 3.10; got {sys.version}")
PY

rm -rf "$VENV"
if ! "$PYTHON_BIN" -m venv --system-site-packages "$VENV"; then
    echo "Error: failed to create virtualenv. Install python3.10-venv and retry." >&2
    exit 1
fi

"$VENV/bin/pip" install --upgrade pip
# Torch and Torchvision must come from JetPack, not PyPI. Existing system-site
# packages satisfy Ultralytics while the remaining dependencies stay in the venv.
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements-jetson.txt"

set_env_value PORT "${PORT:-7806}"
set_env_value PRIVATE "${PRIVATE:-false}"
set_env_value MODEL_DIR "${MODEL_DIR:-$SCRIPT_DIR/models}"
set_env_value MODEL_SIZE "${MODEL_SIZE:-s}"
set_env_value CONFIDENCE_THRESHOLD "${CONFIDENCE_THRESHOLD:-0.1}"
set_env_value IOU_THRESHOLD "${IOU_THRESHOLD:-0.45}"
set_env_value IMAGE_SIZE "${IMAGE_SIZE:-832}"
set_env_value DEVICE "${DEVICE:-cuda}"
set_env_value ALLOW_PRIVATE_URLS "${ALLOW_PRIVATE_URLS:-false}"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
missing=false
for task in breast vagina penis; do
    if [ ! -f "$MODEL_DIR/nsfw-seg-${task}-${MODEL_SIZE}.pt" ]; then
        missing=true
    fi
done
if [ "$missing" = true ]; then
    MODEL_DIR="$MODEL_DIR" MODEL_SIZE="$MODEL_SIZE" "$SCRIPT_DIR/download_models.sh"
fi

MODEL_DIR="$MODEL_DIR" MODEL_SIZE="$MODEL_SIZE" "$VENV/bin/python" - <<'PY'
import os
import torch
import torchvision
from ultralytics import YOLO

print("JetPack PyTorch:", torch.__version__)
print("JetPack Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "Error: JetPack PyTorch cannot see CUDA. Install the NVIDIA JetPack "
        "Torch/Torchvision packages before rerunning this installer."
    )
for task in ("breast", "vagina", "penis"):
    path = os.path.join(
        os.environ["MODEL_DIR"], f"nsfw-seg-{task}-{os.environ['MODEL_SIZE']}.pt"
    )
    model = YOLO(path)
    print(f"Validated {task} checkpoint: {path} ({model.names})")
print("Ultralytics, Jetson CUDA, and checkpoint checks passed")
PY

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm NSFW Segmentation Moderation Service
After=network.target
StartLimitBurst=3
StartLimitIntervalSec=300

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$SCRIPT_DIR
EnvironmentFile=$SCRIPT_DIR/.env
ExecStart=$SCRIPT_DIR/run.sh
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "Generated $SERVICE_FILE for $SERVICE_USER:$SERVICE_GROUP"
if [ "$(id -u)" = "0" ]; then
    cp "$SERVICE_FILE" "/etc/systemd/system/$SERVICE_NAME.service"
    systemctl daemon-reload
    echo "Service installed. Run: systemctl start $SERVICE_NAME"
else
    echo "To install the service:"
    echo "  sudo cp $SERVICE_FILE /etc/systemd/system/$SERVICE_NAME.service"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl start $SERVICE_NAME"
fi
