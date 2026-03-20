#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
source "$SCRIPT_DIR/.env"
source "$SCRIPT_DIR/venv/bin/activate"

LLAMA_BIN="${LLAMA_SERVER_BIN:-llama-server}"

# Start llama-server in the background
"$LLAMA_BIN" \
    --model "${MODEL_PATH}" \
    --mmproj "${MMPROJ_PATH}" \
    --ctx-size 2048 \
    --n-gpu-layers "${N_GPU_LAYERS:-99}" \
    --port "${LLAMA_SERVER_PORT}" \
    --host 127.0.0.1 \
    --no-webui &

LLAMA_PID=$!
trap 'kill $LLAMA_PID 2>/dev/null' EXIT

echo "Waiting for llama-server to start..."
until curl -sf "http://127.0.0.1:${LLAMA_SERVER_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "llama-server ready."

cd "$SCRIPT_DIR"
exec python REST.py
