#!/bin/bash
set -e

# Start llama-server in the background
llama-server \
    --model "${MODEL_PATH}" \
    --mmproj "${MMPROJ_PATH}" \
    --ctx-size 2048 \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --port "${LLAMA_SERVER_PORT}" \
    --host 127.0.0.1 \
    --no-webui &

LLAMA_PID=$!

# Wait for llama-server to be ready
echo "Waiting for llama-server to start..."
until curl -sf "http://127.0.0.1:${LLAMA_SERVER_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "llama-server ready."

# Start Flask
exec python3 REST.py
