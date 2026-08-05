#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/venv/bin/activate"
set -a
source "$SCRIPT_DIR/.env"
set +a
exec python3 REST.py
