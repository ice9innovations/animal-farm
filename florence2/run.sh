#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "$SCRIPT_DIR/venv/bin/activate"
source "$SCRIPT_DIR/.env"

cd "$SCRIPT_DIR"
python REST.py
