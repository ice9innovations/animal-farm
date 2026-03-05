#!/bin/bash
# Starts the Gemini API Flask REST service.
cd "$(dirname "$0")"
source gemini_venv/bin/activate
source .env
python3 REST.py
