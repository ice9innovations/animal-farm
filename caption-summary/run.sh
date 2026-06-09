#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV_DIR="${CAPTION_SUMMARY_VENV:-$SCRIPT_DIR/caption_summary_venv}"

cd "$SCRIPT_DIR"
source "$VENV_DIR/bin/activate"
source "$SCRIPT_DIR/.env"
exec python3 REST.py
