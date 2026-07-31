#!/bin/bash
cd "$(dirname "$0")"
if [ -d venv ]; then
    source venv/bin/activate
else
    source colors_venv/bin/activate
fi
source .env
python3 REST.py
