#!/bin/bash
cd "$(dirname "$0")"

# Set CUDA paths first
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin:$PATH

# Set CuDNN library path for TensorFlow compatibility (AFTER CUDA paths so it takes precedence)
CUDNN_PATH="$(dirname "$0")/nsfw2/lib/python3.9/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$CUDNN_PATH:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

# Suppress duplicate library warnings
export TF_CPP_MIN_LOG_LEVEL=2

source nsfw2/bin/activate
python REST.py
