#!/bin/bash
# Starts the caption-summary Flask REST API.
cd "$(dirname "$0")"
source /home/sd/animal-farm/caption-summary/caption_summary_venv/bin/activate
source .env
python3 REST.py
