# GPT Nano Vision Service

**Port**: 7800
**Framework**: OpenAI API (gpt-4.1-nano default)
**Purpose**: Image captioning via the OpenAI vision API with emoji mapping
**Status**: Active

## Overview

GPT Nano wraps OpenAI's vision models as a standard Animal Farm VLM service. It accepts images by URL, file path, or upload, sends them to the OpenAI API, and returns captions in the unified platform response format with emoji mappings.

## Features

- **Cloud Vision**: OpenAI's latest nano-class models for fast, cost-effective captioning
- **Emoji Integration**: Automatic word-to-emoji mapping with multi-word expression support
- **Unified Input Handling**: Single endpoint for URL, file path, and file upload
- **Configurable Model**: Swap OpenAI model via `.env` without code changes
- **Token Reporting**: Input and output token counts included in response metadata
- **No GPU Required**: Pure API client, no local model weights

## Installation

### Prerequisites

- Python 3.11+
- OpenAI API key (https://platform.openai.com/api-keys)

### Setup

```bash
cd /home/sd/animal-farm/gpt-nano

python3 -m venv gpt_nano_venv
source gpt_nano_venv/bin/activate

pip install -r requirements.txt
```

## Configuration

### Environment Variables (.env)

> **Security note**: This file contains a live API key. Do not commit it to version control.
> Use `--env-file` when running with Docker (see Docker section).

```bash
PORT=7800
PRIVATE=False

# OpenAI API key
OPENAI_API_KEY=sk-proj-...

# Model to use
MODEL=gpt-4.1-nano

# Vision prompt sent with every image
PROMPT="Briefly describe this image in a single short sentence including the number and type of people if any are present and what they are doing and any text included in the image, also mentioning nudity and explicit content if applicable."

# Max tokens in the response
MAX_TOKENS=2000

# Set to false to use local emoji_mappings.json cache only
AUTO_UPDATE=true
```

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Service port |
| `PRIVATE` | Yes | `true` = localhost only |
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `MODEL` | Yes | OpenAI model ID |
| `PROMPT` | Yes | Vision prompt sent with each image |
| `MAX_TOKENS` | Yes | Max completion tokens |
| `AUTO_UPDATE` | No | Refresh emoji/MWE config from GitHub on startup (default: true) |

## API Endpoints

### Health Check

```bash
GET /health
```

```json
{
  "status": "healthy",
  "service": "GPT Nano Vision",
  "model": "gpt-4.1-nano"
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
  "service": "gpt-nano",
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
    "processing_time": 1.102,
    "model_info": {
      "framework": "openai",
      "model": "gpt-4.1-nano",
      "input_tokens": 305,
      "output_tokens": 17
    }
  }
}
```

Images are resized to fit within 1568px on the longest edge before base64 encoding.

## Service Management

### Manual Startup

```bash
cd /home/sd/animal-farm/gpt-nano
./rest.sh
```

### Systemd Service

```bash
sudo cp services/gpt-nano.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gpt-nano
sudo systemctl start gpt-nano
systemctl status gpt-nano
```

## Docker

The `.env` file contains a live API key and is **not** copied into the image. Pass all configuration via `--env-file` at runtime.

```bash
# Build
docker build -t gpt-nano /home/sd/animal-farm/gpt-nano/

# Run
docker run -d \
  --name gpt-nano \
  --env-file /home/sd/animal-farm/gpt-nano/.env \
  -p 7800:7800 \
  gpt-nano

# Test
curl -s http://localhost:7800/health | python3 -m json.tool

curl -s "http://localhost:7800/analyze?url=https://example.com/image.jpg" \
  | python3 -m json.tool

# Delete
docker stop gpt-nano && docker rm gpt-nano && docker rmi gpt-nano
```

## Troubleshooting

**Problem**: `OPENAI_API_KEY environment variable is required`
The `.env` file was not passed. Use `--env-file` as shown above.

**Problem**: `AuthenticationError` from OpenAI
The API key is invalid or revoked. Generate a new key at https://platform.openai.com/api-keys.

**Problem**: `RateLimitError` from OpenAI
Rate limit hit. Wait and retry, or increase the usage tier for the API key.

**Problem**: `model not found`
The model name in `.env` does not exist or is not accessible on the API key's organization. Check available models at https://platform.openai.com/docs/models.

---

**Documentation Version**: 1.0
**Last Updated**: 2026-03-13
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
