#!/bin/bash

# YOLOv8 REST API service
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$CURRENT_DIR"

# Start the service
yolo-venv/bin/python REST.py
