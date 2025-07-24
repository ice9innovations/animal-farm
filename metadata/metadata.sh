#!/bin/bash
cd "$(dirname "$0")"
source metadata_venv/bin/activate
source .env
python REST.py
