#!/bin/bash
cd "$(dirname "$0")"

source .env

if [ "$MODE" = "gpu" ]; then
    export CUDA_HOME=/usr/local/cuda-12.2
    export PATH=/usr/local/cuda-12.2/bin:$PATH
    CUDNN_PATH="$(dirname "$0")/nsfw2/lib/python3.9/site-packages/nvidia/cudnn/lib"
    export LD_LIBRARY_PATH="$CUDNN_PATH:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
fi

source nsfw2/bin/activate
python REST.py
