#!/bin/bash
# Install the Pose service with the BlazePose ONNX CPU backend only.
# Use this on CPU-only hosts such as Raspberry Pi, or as a portable fallback.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/venv"

rm -rf "$VENV"
python3 -m venv "$VENV"

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements-onnx.txt"

echo ""
echo "ONNX CPU venv ready."
echo "Set POSE_BACKEND=onnx and USE_GPU=false in $SCRIPT_DIR/.env before starting."
