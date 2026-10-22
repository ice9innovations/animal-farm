#!/bin/bash
set -e
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if [ -f "$SCRIPT_DIR/.env" ]; then set -a; source "$SCRIPT_DIR/.env"; set +a; fi
cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/venv/bin/python" REST.py
