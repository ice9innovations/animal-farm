#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"
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

export MODEL_ID="${MODEL_ID:-fancyfeast/llama-joycaption-beta-one-hf-llava}"
export MODEL_DIR="${MODEL_DIR:-$JOYCAPTION_CACHE_ROOT/huggingface}"
export HF_HOME="$MODEL_DIR"

model_cached() {
    if [ -d "$MODEL_ID" ]; then
        return 0
    fi

    cache_name="models--${MODEL_ID//\//--}"
    snapshot_dir="$HF_HOME/hub/$cache_name/snapshots"
    [ -d "$snapshot_dir" ] && [ -n "$(find "$snapshot_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)" ]
}

if ! model_cached; then
    echo "JoyCaption model is not cached in $HF_HOME; downloading $MODEL_ID..." >&2
    "$SCRIPT_DIR/download_model.sh"
fi

if ! model_cached; then
    cat >&2 <<EOF
JoyCaption model download did not create a usable local snapshot.

Model: $MODEL_ID
MODEL_DIR: $MODEL_DIR
Expected cache: $HF_HOME/hub/models--${MODEL_ID//\//--}/snapshots
EOF
    exit 1
fi

exec python3 REST.py
