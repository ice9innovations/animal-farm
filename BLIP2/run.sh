#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LAVIS_DIR="$SCRIPT_DIR/LAVIS"

source "$SCRIPT_DIR/.env"

export PYTHONPATH="$LAVIS_DIR"

cd "$SCRIPT_DIR"
"$SCRIPT_DIR/venv/bin/python" REST.py
