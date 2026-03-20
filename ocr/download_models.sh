#!/bin/bash
# Download PaddleOCR model weights (det/rec/cls).
# Run once on the RunPod volume before first container start.
# Files are volume-mounted into the container at /root/.paddleocr.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

download_and_extract() {
    local url="$1"
    local dest_dir="$2"
    local extracted_dir="$3"
    local filename
    filename=$(basename "$url")
    if [ -d "$extracted_dir" ]; then
        echo "Already extracted: $extracted_dir — skipping."
        return 0
    fi
    mkdir -p "$dest_dir"
    echo "Connecting to $(echo "$url" | cut -d/ -f3)..."
    wget -c --connect-timeout=30 --progress=bar:force -P "$dest_dir" "$url"
    tar --no-same-owner -xf "$dest_dir/$filename" -C "$dest_dir"
    rm "$dest_dir/$filename"
}

echo "Downloading PaddleOCR det model (en_PP-OCRv3, 4MB)..."
download_and_extract \
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar" \
    "$MODELS_DIR/whl/det/en" \
    "$MODELS_DIR/whl/det/en/en_PP-OCRv3_det_infer"

echo "Downloading PaddleOCR rec model (en_PP-OCRv4, 10MB)..."
download_and_extract \
    "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar" \
    "$MODELS_DIR/whl/rec/en" \
    "$MODELS_DIR/whl/rec/en/en_PP-OCRv4_rec_infer"

echo "Downloading PaddleOCR cls model (2MB)..."
download_and_extract \
    "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar" \
    "$MODELS_DIR/whl/cls" \
    "$MODELS_DIR/whl/cls/ch_ppocr_mobile_v2.0_cls_infer"

echo "Done. Files in $MODELS_DIR:"
ls -lhR "$MODELS_DIR/whl"
