#!/bin/bash
set -e
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
python3.11 -m venv "$SCRIPT_DIR/venv"
source "$SCRIPT_DIR/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$SCRIPT_DIR/requirements.txt"
