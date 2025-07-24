#!/bin/bash

# Ollama LLM API Startup Script
cd "$(dirname "$0")"

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
fi

# Start the Ollama LLM REST API service
source /home/sd/ollama-api/ollama_venv/bin/activate
python REST.py

