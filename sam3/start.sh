#!/bin/bash
set -e
cd "$(dirname "$0")"
exec /home/sd/sam3/sam3_venv/bin/python REST.py
