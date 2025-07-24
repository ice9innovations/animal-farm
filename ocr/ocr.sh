#!/bin/bash

# PaddleOCR Service Startup Script
# Drop-in replacement for Tesseract OCR service

cd "$(dirname "$0")"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated PaddleOCR virtual environment"
else
    echo "Warning: Virtual environment not found. Run ./install_paddleocr.sh first"
    exit 1
fi

# Start the PaddleOCR service
echo "Starting PaddleOCR service on port ${PORT:-7775}..."
python REST.py