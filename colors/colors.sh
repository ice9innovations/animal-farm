#!/bin/bash
cd "$(dirname "$0")"
source colors_venv/bin/activate
source .env
python3 REST.py
