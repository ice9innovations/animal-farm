#!/bin/bash
# Starts the Flask REST API. llama-server must already be running.
cd "$(dirname "$0")"
source llamacpp_venv/bin/activate
source .env
python3 REST.py
