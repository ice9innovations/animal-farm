#!/bin/bash
# Download PaddleOCR model weights.
# Run once on the RunPod volume before first container start.
# Files are volume-mounted into the container at /root/.paddleocr.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

pip install -q "paddlepaddle==2.5.2" "paddleocr==2.7.3"

echo "Downloading PaddleOCR models (det/rec/cls, lang=en)..."
# PaddleOCR downloads to ~/.paddleocr — redirect HOME so files land in MODELS_DIR
FAKE_HOME=$(mktemp -d)
ln -s "$MODELS_DIR" "$FAKE_HOME/.paddleocr"

HOME="$FAKE_HOME" python3 -c "
from paddleocr import PaddleOCR
PaddleOCR(lang='en', use_angle_cls=True)
"

rm -rf "$FAKE_HOME"

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"
