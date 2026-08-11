#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

set -a
source "$SCRIPT_DIR/.env"
set +a

cd "$SCRIPT_DIR"
"$SCRIPT_DIR/venv/bin/python" REST.py
