#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Missing venv. Run install.sh first:"
    echo "  $SCRIPT_DIR/install.sh"
    exit 1
fi

source "$SCRIPT_DIR/venv/bin/activate"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

if [ -z "${JOYCAPTION_CACHE_ROOT:-}" ]; then
    if [ -d "/mnt/models/workspace" ] && [ -w "/mnt/models/workspace" ]; then
        JOYCAPTION_CACHE_ROOT="/mnt/models/workspace"
    else
        JOYCAPTION_CACHE_ROOT="$SCRIPT_DIR/.cache"
    fi
fi

export MODEL_DIR="${MODEL_DIR:-$JOYCAPTION_CACHE_ROOT/huggingface}"
export HF_HOME="$MODEL_DIR"
export MODEL_ID="${MODEL_ID:-fancyfeast/llama-joycaption-beta-one-hf-llava}"

# This command is specifically for populating the cache, so allow network access
# even when normal runtime is configured for offline operation.
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

mkdir -p "$MODEL_DIR"

python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
hf_home = os.environ["HF_HOME"]
cache_dir = os.path.join(hf_home, "hub")

print(f"Downloading {model_id}")
print(f"Hugging Face home: {hf_home}")
print(f"Hugging Face hub cache: {cache_dir}")
path = snapshot_download(repo_id=model_id, cache_dir=cache_dir, resume_download=True)
print(f"Model snapshot ready: {path}")
PY
