#!/bin/bash
cd "$(dirname "$0")"
source rtdetr-venv/bin/activate
source .env
python3 REST.py
