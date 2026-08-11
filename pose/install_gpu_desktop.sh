#!/bin/bash
# Install the Pose service for desktop Nvidia GPUs into pose/venv.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
POSE_ORT_PACKAGE="${POSE_ORT_PACKAGE:-onnxruntime-gpu==1.22.1}" "$SCRIPT_DIR/install.sh"

if [ -f "$SCRIPT_DIR/.env" ] && ! grep -q '^REQUIRE_GPU=' "$SCRIPT_DIR/.env"; then
    {
        echo ""
        echo "REQUIRE_GPU=true"
    } >> "$SCRIPT_DIR/.env"
fi
