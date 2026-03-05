#!/bin/bash
cd "$(dirname "$0")"
source rembg_venv/bin/activate
source .env
python3 REST.py
