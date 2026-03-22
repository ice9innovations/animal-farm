# Gemini API Vision Service

**Port**: 7767
**Framework**: Google GenAI API (gemini-2.5-flash-lite default)
**Purpose**: Image captioning via the Google Gemini vision API with emoji mapping
**Status**: Active

## Overview

Gemini API wraps Google's Gemini vision models as a standard Animal Farm VLM service. It accepts images by URL, file path, or upload, sends them to the Gemini API, and returns captions in the unified platform response format with emoji mappings.

## Features

- **Cloud Vision**: Google's Gemini models for fast, high-quality captions
- **Emoji Integration**: Automatic word-to-emoji mapping with multi-word expression support
- **Unified Input Handling**: Single endpoint for URL, file path, and file upload
- **Configurable Model**: Swap Gemini model via `.env` without code changes
- **Token Reporting**: Input and output token counts included in response metadata
- **No GPU Required**: Pure API client, no local model weights

## Installation

### Prerequisites

- Python 3.11+
- Google AI Studio API key (https://aistudio.google.com/apikey)

### Setup

```bash
cd /home/sd/animal-farm/gemini-api

python3 -m venv gemini_venv
source gemini_venv/bin/activate

pip install -r requirements.txt
```

## Configuration

### Environment Variables (.env)

> **Security note**: This file contains a live API key. Do not commit it to version control.
> Use `--env-file` when running with Docker (see Docker section).

```bash
PORT=7767
PRIVATE=False

# Google AI Studio API key
GEMINI_API_KEY=AIza...

# Model to use — see https://ai.google.dev/gemini-api/docs/models
MODEL=gemini-2.5-flash-lite

# Vision prompt sent with every image
PROMPT="Briefly describe what you see in this image in a single short sentence."

# Max tokens in the response
MAX_TOKENS=75

# Set to false to use local emoji_mappings.json cache only
AUTO_UPDATE=true
```

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Service port |
| `PRIVATE` | Yes | `true` = localhost only |
| `GEMINI_API_KEY` | Yes | Google AI Studio API key |
| `MODEL` | Yes | Gemini model ID |
| `PROMPT` | Yes | Vision prompt sent with each image |
| `MAX_TOKENS` | Yes | Max output tokens |
| `AUTO_UPDATE` | No | Refresh emoji/MWE config from GitHub on startup (default: true) |

## API Endpoints

### Health Check

```bash
GET /health
```

```json
{
  "status": "healthy",
  "service": "Gemini API Vision",
  "model": "gemini-2.5-flash-lite"
}
```

### Analyze Image

```bash
GET  /analyze?url=<image_url>
GET  /analyze?file=<file_path>
POST /analyze   (multipart file upload)
```

An optional `prompt` parameter overrides the default vision prompt for a single request.

**Response:**
```json
{
  "service": "gemini-api",
  "status": "success",
  "predictions": [
    {
      "text": "A golden retriever sits on a red couch indoors.",
      "emoji_mappings": [
        {"word": "retriever", "emoji": "🐕"},
        {"word": "couch", "emoji": "🛋️"}
      ]
    }
  ],
  "metadata": {
    "processing_time": 0.843,
    "model_info": {
      "framework": "google-genai",
      "model": "gemini-2.5-flash-lite",
      "input_tokens": 298,
      "output_tokens": 16
    }
  }
}
```

Images are resized to fit within 1568px on the longest edge before encoding.

## Service Management

### Manual Startup

```bash
cd /home/sd/animal-farm/gemini-api
./rest.sh
```

### Systemd Service

```bash
sudo cp services/gemini-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gemini-api
sudo systemctl start gemini-api
systemctl status gemini-api
```

## Docker

The `.env` file contains a live API key and is **not** copied into the image. Pass all configuration via `--env-file` at runtime.

```bash
# Build
docker build -t gemini-api /home/sd/animal-farm/gemini-api/

# Run
docker run -d \
  --name gemini-api \
  --env-file /home/sd/animal-farm/gemini-api/.env \
  -p 7767:7767 \
  gemini-api

# Test
curl -s http://localhost:7767/health | python3 -m json.tool

curl -s "http://localhost:7767/analyze?url=https://example.com/image.jpg" \
  | python3 -m json.tool

# Delete
docker stop gemini-api && docker rm gemini-api && docker rmi gemini-api
```

## Troubleshooting

**Problem**: `GEMINI_API_KEY environment variable is required`
The `.env` file was not passed. Use `--env-file` as shown above.

**Problem**: `API_KEY_INVALID` from Google
The key is invalid or the Gemini API is not enabled for the project. Check https://aistudio.google.com/apikey.

**Problem**: `RESOURCE_EXHAUSTED` from Google
Rate limit hit. The free tier has per-minute limits — wait and retry, or upgrade the API quota.

---

**Documentation Version**: 1.0
**Last Updated**: 2026-03-13
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
