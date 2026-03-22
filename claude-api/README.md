# Claude API Vision Service

**Port**: 7797
**Framework**: Anthropic API (claude-haiku-4-5-20251001 default)
**Purpose**: Image captioning via the Anthropic Claude vision API with emoji mapping
**Status**: Active

## Overview

Claude API wraps Anthropic's vision models as a standard Animal Farm VLM service. It accepts images by URL, file path, or upload, sends them to the Claude API, and returns captions in the unified platform response format with emoji mappings.

## Features

- **Cloud Vision**: Anthropic's latest models for high-quality captions
- **Emoji Integration**: Automatic word-to-emoji mapping with multi-word expression support
- **Unified Input Handling**: Single endpoint for URL, file path, and file upload
- **Configurable Model**: Swap Claude model via `.env` without code changes
- **Token Reporting**: Input and output token counts included in response metadata
- **No GPU Required**: Pure API client, no local model weights

## Installation

### Prerequisites

- Python 3.11+
- Anthropic API key (https://console.anthropic.com/)

### Setup

```bash
cd /home/sd/animal-farm/claude-api

python3 -m venv claude_venv
source claude_venv/bin/activate

pip install -r requirements.txt
```

## Configuration

### Environment Variables (.env)

> **Security note**: This file contains a live API key. Do not commit it to version control.
> Use `--env-file` when running with Docker (see Docker section).

```bash
PORT=7797
PRIVATE=False

# Anthropic API key
ANTHROPIC_API_KEY=sk-ant-...

# Model to use — see https://docs.anthropic.com/en/docs/about-claude/models
MODEL=claude-haiku-4-5-20251001

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
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `MODEL` | Yes | Claude model ID |
| `PROMPT` | Yes | Vision prompt sent with each image |
| `MAX_TOKENS` | Yes | Max tokens in response |
| `AUTO_UPDATE` | No | Refresh emoji/MWE config from GitHub on startup (default: true) |

## API Endpoints

### Health Check

```bash
GET /health
```

```json
{
  "status": "healthy",
  "service": "Claude API Vision",
  "model": "claude-haiku-4-5-20251001"
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
  "service": "claude-api",
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
    "processing_time": 1.243,
    "model_info": {
      "framework": "anthropic",
      "model": "claude-haiku-4-5-20251001",
      "input_tokens": 312,
      "output_tokens": 18
    }
  }
}
```

Images are resized to fit within 1568px on the longest edge before encoding — this matches Claude's server-side resize threshold and avoids unnecessary latency.

## Service Management

### Manual Startup

```bash
cd /home/sd/animal-farm/claude-api
./rest.sh
```

### Systemd Service

```bash
sudo cp services/claude-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable claude-api
sudo systemctl start claude-api
systemctl status claude-api
```

## Docker

The `.env` file contains a live API key and is **not** copied into the image. Pass all configuration via `--env-file` at runtime.

```bash
# Build
docker build -t claude-api /home/sd/animal-farm/claude-api/

# Run
docker run -d \
  --name claude-api \
  --env-file /home/sd/animal-farm/claude-api/.env \
  -p 7797:7797 \
  claude-api

# Test
curl -s http://localhost:7797/health | python3 -m json.tool

curl -s "http://localhost:7797/analyze?url=https://example.com/image.jpg" \
  | python3 -m json.tool

# Delete
docker stop claude-api && docker rm claude-api && docker rmi claude-api
```

## Troubleshooting

**Problem**: `ANTHROPIC_API_KEY environment variable is required`
The `.env` file was not passed to the container. Use `--env-file` as shown above.

**Problem**: `authentication_error` from Anthropic
The API key is invalid or expired. Generate a new key at https://console.anthropic.com/.

**Problem**: `overloaded_error` from Anthropic
Anthropic API is under load. The request will need to be retried.

**Problem**: Images resize unexpectedly
Claude resizes images server-side above 1568px. The service pre-resizes to this threshold to save upload time and token cost.

---

**Documentation Version**: 1.0
**Last Updated**: 2026-03-13
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
