#!/bin/bash

# YOLOv8 REST API service
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$CURRENT_DIR"

# Check if virtual environment exists
if [ -f "yolo-venv/bin/activate" ]; then
    source yolo-venv/bin/activate
fi

# Start the service
python REST.py
