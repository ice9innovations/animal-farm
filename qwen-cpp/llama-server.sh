#!/bin/bash
# Starts llama-server with Qwen3-VL model and vision projector.
# This must be running before REST.py is started.
cd "$(dirname "$0")"
source .env

/home/sd/llama.cpp/build/bin/llama-server \
    --model "${MODEL_PATH}" \
    --mmproj "${MMPROJ_PATH}" \
    --ctx-size 2048 \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --port "${LLAMA_SERVER_PORT}" \
    --host 127.0.0.1 \
    --no-webui
