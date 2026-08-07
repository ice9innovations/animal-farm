#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "$SCRIPT_DIR/.env"

cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/venv/bin/python" REST.py
