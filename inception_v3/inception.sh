#!/bin/bash
cd /home/sd/inception_v3
source inception_v3_venv/bin/activate
source .env
export LD_LIBRARY_PATH="$(dirname "$0")/inception_v3_venv/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
python3 REST.py
