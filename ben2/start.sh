#!/bin/bash
cd "$(dirname "$0")"
source ben2_venv/bin/activate
source .env
python3 REST.py
