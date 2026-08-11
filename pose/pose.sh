#!/bin/bash
cd "$(dirname "$0")"
source pose_venv/bin/activate
set -a
source .env
set +a
python REST.py
