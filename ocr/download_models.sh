#!/bin/bash
# RapidOCR downloads ONNX model assets through the Python package cache on first
# use. This script is kept as a no-op compatibility entry point for existing
# provisioning flows that still call ocr/download_models.sh.
set -e

echo "RapidOCR uses ONNX assets managed by rapidocr-onnxruntime; no Paddle models to download."
