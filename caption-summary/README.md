# Caption Summary Service

**Port**: 7799
**Framework**: Flask + llama.cpp or Anthropic API
**Purpose**: Synthesizes multiple VLM captions and noun/verb consensus data into a single descriptive sentence
**Status**: Active

## Overview

Caption Summary accepts captions from multiple VLM services (BLIP2, moondream, ollama, etc.) along with noun and verb consensus data from the pipeline, and synthesizes them into one coherent sentence. It supports two backends: a local llama.cpp server (default, via qwen-cpp) or the Anthropic Claude API.

## Features

- **Dual Backend**: Runs against a local llama.cpp server or the Anthropic API
- **Consensus-Aware**: Incorporates noun and verb consensus vote counts into the prompt for better synthesis
- **Fast**: Local backend produces results in under 0.1s
- **No GPU Required**: Flask wrapper only — inference runs in the backend server

## Installation

### Prerequisites

- Python 3.11+
- A running llama.cpp server (qwen-cpp, port 11436) **or** an Anthropic API key

### Setup

```bash
cd /home/sd/animal-farm/caption-summary

python3 -m venv caption_summary_venv
source caption_summary_venv/bin/activate

pip install -r requirements.txt
```

## Configuration

### Environment Variables (.env)

```bash
PORT=7799
PRIVATE=false

# "llamacpp" uses the local llama.cpp server (e.g. qwen-cpp)
# "claude" uses the Anthropic API
SYNTHESIS_BACKEND=llamacpp

# Required for SYNTHESIS_BACKEND=llamacpp
LLAMA_SERVER_HOST=http://127.0.0.1:11436

# Model name for logging (llamacpp) or actual API model ID (claude)
SYNTHESIS_MODEL=Qwen3-VL-2B
# SYNTHESIS_MODEL=claude-haiku-4-5-20251001

# Required for SYNTHESIS_BACKEND=claude
# ANTHROPIC_API_KEY=sk-ant-...

# Max tokens in synthesized output
# MAX_TOKENS=75
```

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Service port |
| `PRIVATE` | Yes | `true` = localhost only |
| `SYNTHESIS_BACKEND` | No | `llamacpp` (default) or `claude` |
| `LLAMA_SERVER_HOST` | When backend=llamacpp | Base URL of llama.cpp server |
| `SYNTHESIS_MODEL` | No | Model name for logging or Claude model ID |
| `ANTHROPIC_API_KEY` | When backend=claude | Anthropic API key |
| `MAX_TOKENS` | No | Max output tokens (default: 75) |

## API Endpoints

### Health Check

```bash
GET /health
```

```json
{
  "status": "healthy",
  "backend": "llamacpp",
  "backend_status": "ok",
  "backend_detail": "ok",
  "model": "Qwen3-VL-2B"
}
```

### Synthesize Caption

```bash
POST /summarize
Content-Type: application/json
```

**Request body:**
```json
{
  "captions": {
    "blip2": "a golden retriever sits on a couch",
    "moondream": "a dog resting indoors",
    "qwen": "a large dog on a fabric sofa"
  },
  "nouns": [
    {"canonical": "dog", "category": "animal", "vote_count": 3},
    {"canonical": "couch", "category": "furniture", "vote_count": 2}
  ],
  "verbs": [
    {"canonical": "sit", "vote_count": 2}
  ]
}
```

`nouns` and `verbs` are optional. Only `captions` is required.

**Response:**
```json
{
  "status": "success",
  "summary": "A golden retriever sits comfortably on a fabric sofa indoors.",
  "model": "Qwen3-VL-2B",
  "processing_time": 0.077
}
```

## Service Management

### Manual Startup

```bash
cd /home/sd/animal-farm/caption-summary
./rest.sh
```

### Systemd Service

```bash
sudo cp services/caption-summary.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable caption-summary
sudo systemctl start caption-summary
systemctl status caption-summary
```

## Docker

### Prerequisites

The llama.cpp backend (`qwen-cpp-server.service`) must be running on the host before starting this container. The container uses `--network=host` so it can reach `127.0.0.1:11436`.

```bash
# Build
docker build -t caption-summary /home/sd/animal-farm/caption-summary/

# Run (host networking required to reach local llama-server)
docker run -d \
  --name caption-summary \
  --network=host \
  --env-file /home/sd/animal-farm/caption-summary/.env \
  caption-summary

# Test
curl -s http://localhost:7799/health | python3 -m json.tool

curl -s -X POST http://localhost:7799/summarize \
  -H "Content-Type: application/json" \
  -d '{"captions": {"blip": "a dog in a park", "clip": "golden retriever outdoors"}}' \
  | python3 -m json.tool

# Delete
docker stop caption-summary && docker rm caption-summary && docker rmi caption-summary
```

> To use the Claude backend instead, set `SYNTHESIS_BACKEND=claude` and `ANTHROPIC_API_KEY=...` in `.env`. No `--network=host` required for that configuration.

## Troubleshooting

**Problem**: `LLAMA_SERVER_HOST is required when SYNTHESIS_BACKEND=llamacpp`
Ensure the `.env` has `LLAMA_SERVER_HOST` set and `qwen-cpp-server` is running.

**Problem**: Summarize request hangs
The llama.cpp server is unreachable. Check `systemctl status qwen-cpp-server` and verify `LLAMA_SERVER_HOST` matches.

**Problem**: Empty summary returned
The LLM returned an empty response. Check the backend server logs for errors.

---

**Documentation Version**: 1.0
**Last Updated**: 2026-03-13
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
