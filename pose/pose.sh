#!/bin/bash
cd "$(dirname "$0")"
source pose_venv/bin/activate
source .env
python3 REST.py