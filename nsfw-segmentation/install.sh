#!/usr/bin/env bash
# Install the NSFW segmentation service on an NVIDIA desktop/server.
# Requires Python 3.11. CUDA wheels default to cu128 for current NVIDIA GPUs.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/venv"
SERVICE_NAME="nsfw-segmentation"
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0+cu128}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0+cu128}"
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

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN was not found. Install Python 3.11 and its venv package." >&2
    exit 1
fi

rm -rf "$VENV"
if ! "$PYTHON_BIN" -m venv "$VENV"; then
    echo "Error: failed to create virtualenv. Install the Python venv package and retry." >&2
    exit 1
fi

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --no-cache-dir \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "$PYTORCH_INDEX"
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

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

DEVICE="$DEVICE" MODEL_DIR="$MODEL_DIR" MODEL_SIZE="$MODEL_SIZE" "$VENV/bin/python" - <<'PY'
import os
import torch
import torchvision
from ultralytics import YOLO

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if os.environ["DEVICE"].startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit(
        "Error: DEVICE requests CUDA, but PyTorch cannot see a CUDA GPU. "
        "Check the NVIDIA driver or rerun with DEVICE=cpu."
    )
for task in ("breast", "vagina", "penis"):
    path = os.path.join(
        os.environ["MODEL_DIR"], f"nsfw-seg-{task}-{os.environ['MODEL_SIZE']}.pt"
    )
    model = YOLO(path)
    print(f"Validated {task} checkpoint: {path} ({model.names})")
print("Ultralytics, accelerator, and checkpoint checks passed")
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

echo "Generated $SERVICE_FILE"
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
