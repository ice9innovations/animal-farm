#!/bin/bash

# Ollama LLM API Startup Script
cd "$(dirname "$0")"

# Load environment variables if .env exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Start the Ollama LLM REST API service
source /home/sd/ollama-api/ollama_venv/bin/activate
python REST.py

