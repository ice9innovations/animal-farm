#!/bin/bash
# Starts the Claude API Flask REST service.
cd "$(dirname "$0")"
source claude_venv/bin/activate
source .env
python3 REST.py
