#!/bin/bash
cd "$(dirname "$0")"
source qr_venv/bin/activate
source .env
python3 REST.py
