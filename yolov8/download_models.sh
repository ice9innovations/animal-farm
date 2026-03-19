#!/bin/bash
# Download YOLOv8l weights from Ultralytics.
# Run once on the RunPod volume before first container start.
# Files are volume-mounted into the container at /models/yolov8.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

echo "Connecting to github.com for yolov8l.pt (84MB)..."
wget -c --connect-timeout=30 --progress=bar:force \
    -O "$MODELS_DIR/yolov8l.pt" \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l.pt"

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.pt
