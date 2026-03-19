#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LAVIS_DIR="$(realpath "$SCRIPT_DIR/../LAVIS")"

source "$SCRIPT_DIR/venv/bin/activate"
source "$SCRIPT_DIR/.env"

export PYTHONPATH="$LAVIS_DIR"

cd "$SCRIPT_DIR"
python REST.py
