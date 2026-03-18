# qwen-cpp Vision Service

**Port**: 7796 (Flask REST wrapper)
**llama-server port**: 11436
**Framework**: llama.cpp v8100 + Flask (Qwen3-VL model)
**Purpose**: Local vision-language model inference using Qwen3-VL via the llama.cpp server
**Status**: Active

## Overview

qwen-cpp is a two-process service. `llama-server` (the compiled C++ binary) loads Qwen3-VL and handles inference. The Flask REST wrapper (`REST.py`) provides the Animal Farm unified API on top of it. Both must be running for the service to function.

The service uses the `/v1/chat/completions` OAI-compatible endpoint with base64-encoded image payloads — required for llama.cpp v8100+ which uses the mtmd multimodal library.

## Architecture

```
Client → Flask REST (port 7796) → llama-server (port 11436) → GPU
```

## Features

- **Local Inference**: No cloud API calls — runs entirely on-device
- **Qwen3-VL**: Efficient vision-language model with strong multilingual understanding
- **Emoji Integration**: Automatic word-to-emoji mapping
- **Configurable Temperature**: Per-request temperature override via query param
- **GPU Accelerated**: llama-server runs with CUDA, SM 8.6

## Installation

### Prerequisites

- Python 3.11+
- llama.cpp binary at `/home/sd/llama.cpp/build/bin/llama-server`
- Model weights in `models/` (see Model Setup below)
- CUDA 12.2+ for GPU acceleration

### Model Setup

GGUF model files are stored locally in `models/`:

```
models/Qwen3VL-2B-Instruct-Q4_K_M.gguf
models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf
```

A 4B variant is also available — swap the paths in `.env`:

```
models/Qwen3VL-4B-Instruct-Q4_K_M.gguf
models/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
```

### Setup

```bash
cd /home/sd/animal-farm/qwen-cpp

python3 -m venv qwen_venv
source qwen_venv/bin/activate

pip install -r requirements.txt
```

## Configuration

### Environment Variables (.env)

```bash
PORT=7796
PRIVATE=False

LLAMA_SERVER_HOST=http://127.0.0.1:11436
LLAMA_SERVER_PORT=11436
N_GPU_LAYERS=99

MODEL_PATH=/home/sd/animal-farm/qwen-cpp/models/Qwen3VL-2B-Instruct-Q4_K_M.gguf
MMPROJ_PATH=/home/sd/animal-farm/qwen-cpp/models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf
MODEL_NAME=qwen3-vl-2b

PROMPT="Briefly describe what you see in this image in a single short sentence."
TEMPERATURE=0.3
AUTO_UPDATE=true
```

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Flask service port |
| `PRIVATE` | Yes | `true` = localhost only |
| `LLAMA_SERVER_HOST` | Yes | Base URL of llama-server |
| `LLAMA_SERVER_PORT` | Yes | llama-server listening port |
| `N_GPU_LAYERS` | Yes | Number of layers to offload to GPU (99 = all) |
| `MODEL_PATH` | Yes | Path to Qwen3-VL GGUF weights |
| `MMPROJ_PATH` | Yes | Path to multimodal projector GGUF |
| `MODEL_NAME` | Yes | Model name for logging |
| `PROMPT` | Yes | Default vision prompt |
| `TEMPERATURE` | Yes | Sampling temperature |
| `AUTO_UPDATE` | No | Refresh emoji/MWE config from GitHub (default: true) |

## Service Management

### Start llama-server

```bash
cd /home/sd/animal-farm/qwen-cpp
./llama-server.sh
```

### Start Flask wrapper

```bash
./rest.sh
```

### Systemd Services

```bash
# llama-server
sudo cp services/qwen-cpp-server.service /etc/systemd/system/
sudo systemctl enable qwen-cpp-server
sudo systemctl start qwen-cpp-server

# Flask wrapper
sudo cp services/qwen-cpp-api.service /etc/systemd/system/
sudo systemctl enable qwen-cpp-api
sudo systemctl start qwen-cpp-api

systemctl status qwen-cpp-server qwen-cpp-api
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns `503` if `llama-server` is unreachable.

```json
{
  "status": "healthy",
  "service": "Qwen3-VL Vision API",
  "llama_server": {
    "host": "http://127.0.0.1:11436",
    "status": "ok"
  }
}
```

### Analyze Image

```bash
GET  /v3/analyze?url=<image_url>[&prompt=<text>][&temperature=<float>]
GET  /v3/analyze?file=<file_path>
POST /v3/analyze   (multipart file upload)
```

Also available at `/analyze` (no version prefix).

**Response:**
```json
{
  "service": "qwen",
  "status": "success",
  "predictions": [
    {
      "text": "A golden retriever sits on a red couch indoors.",
      "emoji_mappings": [
        {"word": "retriever", "emoji": "🐕"}
      ]
    }
  ],
  "metadata": {
    "processing_time": 0.416,
    "model_info": {
      "framework": "llama.cpp/qwen3-vl",
      "server": "http://127.0.0.1:11436"
    }
  }
}
```

## Docker

There are two Dockerfiles. Use `Dockerfile` (the default) when `llama-server` runs on the host. Use `Dockerfile.server` when you need a fully self-contained image that builds and runs `llama-server` itself — e.g. RunPod or any environment where you can't rely on a host-side binary.

> Consolidating these into a single Dockerfile is tracked in `issues/containerize-llama-server-binary.md`.

### Dockerfile — Flask wrapper only (default)

`llama-server` runs on the host as `qwen-cpp-server.service` and is reached via `--network=host`.

> **Note**: Stop `qwen-cpp-api.service` before starting the Docker container — both bind to port 7796 and only one can run at a time.

```bash
# Ensure qwen-cpp-server is running first
systemctl status qwen-cpp-server

# Build
docker build -t qwen-cpp /home/sd/animal-farm/qwen-cpp/

# Run (host networking so 127.0.0.1:11436 reaches llama-server)
docker run -d \
  --name qwen-cpp \
  --network=host \
  --env-file /home/sd/animal-farm/qwen-cpp/.env \
  qwen-cpp

# Test
curl -s http://localhost:7796/health | python3 -m json.tool

curl -s -X POST -F "file=@/path/to/image.jpg" \
  http://localhost:7796/v3/analyze | python3 -m json.tool

# Delete
docker stop qwen-cpp && docker rm qwen-cpp && docker rmi qwen-cpp
```

### Dockerfile.server — self-contained (llama-server + Flask)

Builds `llama-server` from source in a CUDA devel stage, then packages it with the Flask wrapper. No host binary required. Models are volume-mounted at runtime.

The `CUDA_ARCH` build arg defaults to `80;86;89;90` (A100, RTX 3090/4090, L40). Override for your specific GPU to reduce binary size and build time.

```bash
# Build (default multi-arch)
docker build -f Dockerfile.server -t qwen-cpp-server /home/sd/animal-farm/qwen-cpp/

# Build for a specific GPU (e.g. RTX 3090 = SM 8.6)
docker build -f Dockerfile.server --build-arg CUDA_ARCH=86 -t qwen-cpp-server /home/sd/animal-farm/qwen-cpp/

# Run (models volume-mounted, GPU passthrough required)
docker run -d \
  --name qwen-cpp-server \
  --gpus all \
  --env-file /home/sd/animal-farm/qwen-cpp/.env \
  -v /home/sd/animal-farm/qwen-cpp/models:/app/models:ro \
  -p 7796:7796 \
  qwen-cpp-server

# Delete
docker stop qwen-cpp-server && docker rm qwen-cpp-server && docker rmi qwen-cpp-server
```

## Troubleshooting

**Problem**: Service exits at startup with `llama-server not reachable`
Start `qwen-cpp-server.service` first. The Flask wrapper performs a connectivity check on boot.

**Problem**: Health returns `503`
`qwen-cpp-server` is down or `LLAMA_SERVER_HOST` is wrong.

**Problem**: `500 Internal Server Error` on analyze
Two processes are competing for the same llama-server. Stop either `qwen-cpp-api.service` or the Docker container, not both.

---

**Documentation Version**: 1.0
**Last Updated**: 2026-03-13
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
