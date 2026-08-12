#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
source "$SCRIPT_DIR/.env"

LLAMA_BIN="${LLAMA_SERVER_BIN:-${WORKSPACE_DIR:-/workspace}/llama-server/build/bin/llama-server}"
if [ ! -x "$LLAMA_BIN" ]; then
    echo "llama-server binary not found or not executable: $LLAMA_BIN"
    echo "Run: $SCRIPT_DIR/build_server.sh"
    exit 1
fi

LLAMA_BIN_DIR="$(dirname "$(realpath "$LLAMA_BIN")")"
export LD_LIBRARY_PATH="$LLAMA_BIN_DIR:${LD_LIBRARY_PATH:-}"

echo "Waiting for llama-server to start..."
if curl -sf "http://127.0.0.1:${LLAMA_SERVER_PORT}/health" > /dev/null 2>&1; then
    echo "llama-server already ready."
else
    # Start llama-server in the background
    "$LLAMA_BIN" \
        --model "${MODEL_PATH}" \
        --mmproj "${MMPROJ_PATH}" \
        --ctx-size 5120 \
        --n-gpu-layers "${N_GPU_LAYERS:-99}" \
        --port "${LLAMA_SERVER_PORT}" \
        --host 127.0.0.1 \
        --no-webui &

    LLAMA_PID=$!
    trap 'kill $LLAMA_PID 2>/dev/null' EXIT

    until curl -sf "http://127.0.0.1:${LLAMA_SERVER_PORT}/health" > /dev/null 2>&1; do
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            wait "$LLAMA_PID"
            exit $?
        fi
        sleep 2
    done
fi
echo "llama-server ready."

cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/venv/bin/python" REST.py
