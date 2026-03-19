#!/bin/bash
# Download YOLOv8l weights from Ultralytics.
# Run once on the RunPod volume before first container start.
# Files are volume-mounted into the container at /models/yolov8.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

pip install -q ultralytics

echo "Downloading yolov8l.pt..."
python3 - <<EOF
import os
from ultralytics import YOLO
os.chdir("$MODELS_DIR")
YOLO("yolov8l.pt")  # downloads to cwd if not present
EOF

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.pt
