#!/bin/bash
# Starts the GPT Nano Flask REST service.
cd "$(dirname "$0")"
source gpt_nano_venv/bin/activate
source .env
python3 REST.py
