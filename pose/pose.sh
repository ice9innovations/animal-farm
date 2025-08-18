#!/bin/bash
# Pose Estimation Service Startup Script

cd "$(dirname "$0")"

echo "🧍 Starting Pose Estimation Service..."

# Check if virtual environment exists
if [ ! -d "pose_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv pose_venv
fi

# Activate virtual environment
source pose_venv/bin/activate

# Install dependencies if requirements.txt is newer than last install
if [ requirements.txt -nt pose_venv/installed.timestamp ] || [ ! -f pose_venv/installed.timestamp ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch pose_venv/installed.timestamp
fi

# Check if .env file exists, create from sample if not
if [ ! -f .env ]; then
    if [ -f .env.sample ]; then
        echo "Creating .env from .env.sample..."
        cp .env.sample .env
        echo "⚠️  Please edit .env file with your configuration"
    else
        echo "❌ Error: No .env.sample file found"
        exit 1
    fi
fi

# Get port from .env file
PORT=$(grep '^PORT=' .env 2>/dev/null | cut -d= -f2 | tr -d ' ')
if [ -z "$PORT" ]; then
    PORT=7783
fi

echo "🚀 Starting Pose Estimation Service on port $PORT"
echo "📊 Enhanced pose classification with 10+ pose types"
echo "🎯 MediaPipe Pose with body segmentation and joint analysis"

python3 REST.py