#!/bin/bash
# EasyOCR downloads its detector/recognizer model assets through its Python
# model cache on first use. This script is kept as a compatibility entry point
# for provisioning flows that still call ocr/download_models.sh.
set -e

echo "EasyOCR model assets are managed by EasyOCR; no Paddle models to download."
