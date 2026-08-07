#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
CURRENT_USER="$(id -un)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.sample" "$SCRIPT_DIR/.env"
fi

source "$SCRIPT_DIR/.env"

DEFAULT_REPO_URL="https://github.com/fpgaminer/joycaption.git"
REPO_URL="${JOYCAPTION_REPO_URL:-$DEFAULT_REPO_URL}"
REPO_DIR="$SCRIPT_DIR/joycaption-src"

if [ -z "${JOYCAPTION_CACHE_ROOT:-}" ]; then
    if [ -d "/mnt/models/workspace" ] && [ -w "/mnt/models/workspace" ]; then
        JOYCAPTION_CACHE_ROOT="/mnt/models/workspace"
    else
        JOYCAPTION_CACHE_ROOT="$SCRIPT_DIR/.cache"
    fi
fi

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$JOYCAPTION_CACHE_ROOT/pip-cache}"
export TMPDIR="${TMPDIR:-$JOYCAPTION_CACHE_ROOT/tmp}"
export MODEL_DIR="${MODEL_DIR:-$JOYCAPTION_CACHE_ROOT/huggingface}"
export HF_HOME="$MODEL_DIR"
VENV_DIR="${JOYCAPTION_VENV_DIR:-$SCRIPT_DIR/venv}"

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$MODEL_DIR" "$(dirname "$VENV_DIR")"

if [[ "$REPO_URL" == /* || "$REPO_URL" == ./* || "$REPO_URL" == ../* ]]; then
    if [ ! -d "$REPO_URL/.git" ]; then
        echo "Configured local JoyCaption repo does not exist: $REPO_URL"
        echo "Falling back to https://github.com/fpgaminer/joycaption.git"
        REPO_URL="https://github.com/fpgaminer/joycaption.git"
    fi
fi

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi

rm -rf "$VENV_DIR"
if [ "$VENV_DIR" != "$SCRIPT_DIR/venv" ] && [ -L "$SCRIPT_DIR/venv" ]; then
    rm -f "$SCRIPT_DIR/venv"
fi
python3.11 -m venv "$VENV_DIR"
if [ "$VENV_DIR" != "$SCRIPT_DIR/venv" ]; then
    rm -rf "$SCRIPT_DIR/venv"
    ln -s "$VENV_DIR" "$SCRIPT_DIR/venv"
fi
source "$SCRIPT_DIR/venv/bin/activate"

choose_torch_defaults() {
    local compute_cap
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 || true)"

    if [[ "$compute_cap" == 12.* ]]; then
        TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
        TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
        TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    else
        TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
        TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
        TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
    fi

    if [[ "$compute_cap" == 12.* && "$TORCH_INDEX_URL" != *cu128* ]]; then
        echo "Detected compute capability $compute_cap, which needs PyTorch CUDA 12.8 wheels."
        echo "Current TORCH_INDEX_URL is '$TORCH_INDEX_URL'."
        echo "Remove TORCH_INDEX_URL from .env or set it to https://download.pytorch.org/whl/cu128"
        exit 1
    fi
}

choose_torch_defaults

pip install --upgrade pip
pip install --no-cache-dir "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"
pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

chmod +x "$SCRIPT_DIR/run.sh" "$SCRIPT_DIR/rest.sh" "$SCRIPT_DIR/joycaption.sh" "$SCRIPT_DIR/download_model.sh"

mkdir -p "$SCRIPT_DIR/services"
cat > "$SCRIPT_DIR/services/joycaption-api.service" <<EOF
[Unit]
Description=JoyCaption Vision REST API
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=10
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/rest.sh

[Install]
WantedBy=multi-user.target
EOF

echo "JoyCaption installed. Edit $SCRIPT_DIR/.env if needed, then run:"
echo "  $SCRIPT_DIR/run.sh"
echo ""
echo "Cache root: $JOYCAPTION_CACHE_ROOT"
echo "Hugging Face cache: $MODEL_DIR"
echo "pip cache: $PIP_CACHE_DIR"
echo "temp dir: $TMPDIR"
echo "venv: $VENV_DIR"
echo ""
echo "Generated systemd unit:"
echo "  $SCRIPT_DIR/services/joycaption-api.service"
