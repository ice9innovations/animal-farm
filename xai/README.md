# xAI Grok Vision Service

**Port**: 7805
**Framework**: xAI API (OpenAI-compatible)
**Purpose**: Image captioning via the xAI Grok vision API with emoji mapping
**Status**: Active

## Overview

xAI wraps Grok's vision models as a standard Animal Farm VLM service. It accepts images by URL, file path, or upload, sends them to the xAI API, and returns captions in the unified platform response format with emoji mappings. The xAI API is OpenAI-compatible — the implementation is structurally identical to `gpt-nano`.

## Features

- **Cloud Vision**: xAI Grok vision models for image captioning
- **Emoji Integration**: Automatic word-to-emoji mapping with multi-word expression support
- **Unified Input Handling**: Single endpoint for URL, file path, and file upload
- **Configurable Model**: Swap Grok model via `.env` without code changes
- **Token Reporting**: Input and output token counts included in response metadata
- **No GPU Required**: Pure API client, no local model weights

## Installation

### Prerequisites

- Python 3.11+
- xAI API key (https://console.x.ai/)

### RunPod

```bash
cd /workspace/animal-farm/xai
bash install.sh
```

### Manual

```bash
cd /home/sd/animal-farm/xai
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

### Environment Variables (.env)

> **Security note**: This file contains a live API key. Do not commit it to version control.

```bash
PORT=7805
PRIVATE=true

# xAI API key
XAI_API_KEY=xai-...

# Model to use — see https://docs.x.ai/docs/models
MODEL=grok-4-1-fast-non-reasoning

# Vision prompt sent with every image
PROMPT="Briefly describe this image in a single short sentence including the number and type of people if any are present and what they are doing and any text included in the image, also mentioning nudity and explicit content if applicable."

# Max tokens in the response
MAX_TOKENS=150

# Set to false to use local emoji_mappings.json cache only
AUTO_UPDATE=true
```

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Service port |
| `PRIVATE` | Yes | `true` = localhost only |
| `XAI_API_KEY` | Yes | xAI API key |
| `MODEL` | Yes | Grok model ID |
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
  "service": "xAI Grok Vision",
  "model": "grok-4-1-fast-non-reasoning"
}
```

### Analyze Image

```bash
GET  /v3/analyze?url=<image_url>
GET  /v3/analyze?file=<file_path>
POST /v3/analyze   (multipart file upload)
```

Also available at `/analyze` (no version prefix).

An optional `prompt` parameter overrides the default vision prompt for a single request.

**Response:**
```json
{
  "service": "xai",
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
      "framework": "xai",
      "model": "grok-4-1-fast-non-reasoning",
      "input_tokens": 312,
      "output_tokens": 14
    }
  }
}
```

Images are resized to fit within 1568px on the longest edge before base64 encoding.

## Service Management

### RunPod (farm.sh)

```bash
./farm.sh start xai
./farm.sh stop xai
./farm.sh restart xai
```

### Manual Startup

```bash
cd /home/sd/animal-farm/xai
./run.sh
```

### Systemd

```bash
sudo cp xai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable xai
sudo systemctl start xai
systemctl status xai
```

## Troubleshooting

**Problem**: `XAI_API_KEY environment variable is required`
The `.env` file is missing or was not sourced. Copy `.env.sample` to `.env` and fill in the key.

**Problem**: `AuthenticationError` from xAI
The API key is invalid or revoked. Generate a new key at https://console.x.ai/.

**Problem**: `model not found`
The model name in `.env` does not exist. Check available models at https://docs.x.ai/docs/models.

---

**Documentation Version**: 1.0
**Last Updated**: 2026-03-20
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
