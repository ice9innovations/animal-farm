# llama-cpp Vision Service

**Port**: 7782 (Flask REST wrapper)
**llama-server port**: 11435
**Framework**: llama.cpp v8100 + Flask
**Purpose**: Local vision-language model inference using llava-llama3 via the llama.cpp server
**Status**: Active

## Overview

llama-cpp is a two-process service. `llama-server` (the compiled C++ binary) loads the VLM model and handles inference. The Flask REST wrapper (`REST.py`) provides the Animal Farm unified API on top of it. Both must be running for the service to function.

The service uses the `/v1/chat/completions` OAI-compatible endpoint with base64-encoded image payloads — required for llama.cpp v8100+ which uses the mtmd multimodal library.

## Architecture

```
Client → Flask REST (port 7782) → llama-server (port 11435) → GPU
```

## Features

- **Local Inference**: No cloud API calls — runs entirely on-device
- **VLM Support**: llava-llama3 (default) or llama3.2-vision
- **Emoji Integration**: Automatic word-to-emoji mapping
- **Configurable Temperature**: Per-request temperature override via query param
- **GPU Accelerated**: llama-server runs with CUDA, SM 8.6

## Installation

### Prerequisites

- Python 3.11+
- llama.cpp binary at `/home/sd/llama.cpp/build/bin/llama-server`
- Model weights (see Model Setup below)
- CUDA 12.2+ for GPU acceleration

### Build llama-server

See `docs/SUDO.md` for the confirmed working build command (requires gcc-11 and real nvcc at `/usr/local/cuda/bin/nvcc`). The binary at `/home/sd/llama.cpp/build/bin/llama-server` is v8100, built for CUDA SM 8.6.

### Model Setup

The default model is llava-llama3, symlinked from the Ollama blob store:

```
models/llava-llama3-8b-q4_k_m.gguf    → ollama-api/models/
models/llava-llama3-mmproj-clip.gguf  → ollama-api/models/
```

Alternatively, switch to llama3.2-vision by updating `MODEL_PATH`, `MMPROJ_PATH`, and `MODEL_NAME` in `.env`.

### Setup

```bash
cd /home/sd/animal-farm/llama-cpp

python3 -m venv llamacpp_venv
source llamacpp_venv/bin/activate

pip install -r requirements.txt
```

## Configuration

### Environment Variables (.env)

```bash
PORT=7782
PRIVATE=False

LLAMA_SERVER_HOST=http://127.0.0.1:11435
LLAMA_SERVER_PORT=11435
N_GPU_LAYERS=99

MODEL_PATH=/home/sd/animal-farm/ollama-api/models/llava-llama3-8b-q4_k_m.gguf
MMPROJ_PATH=/home/sd/animal-farm/ollama-api/models/llava-llama3-mmproj-clip.gguf
MODEL_NAME=llava-llama3

PROMPT="Briefly describe this image in a single short sentence."
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
| `MODEL_PATH` | Yes | Path to GGUF model weights |
| `MMPROJ_PATH` | Yes | Path to CLIP multimodal projector |
| `MODEL_NAME` | Yes | Model name for logging |
| `PROMPT` | Yes | Default vision prompt |
| `TEMPERATURE` | Yes | Sampling temperature |
| `AUTO_UPDATE` | No | Refresh emoji/MWE config from GitHub (default: true) |

## Service Management

### Start llama-server

```bash
cd /home/sd/animal-farm/llama-cpp
./llama-server.sh
```

### Start Flask wrapper

```bash
./rest.sh
```

### Systemd Services

```bash
# llama-server
sudo cp services/llama-cpp-server.service /etc/systemd/system/
sudo systemctl enable llama-cpp-server
sudo systemctl start llama-cpp-server

# Flask wrapper
sudo cp services/llama-cpp-api.service /etc/systemd/system/
sudo systemctl enable llama-cpp-api
sudo systemctl start llama-cpp-api

systemctl status llama-cpp-server llama-cpp-api
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns `503` with reason if `llama-server` is unreachable.

```json
{
  "status": "healthy",
  "service": "llama.cpp Vision API",
  "llama_server": {
    "host": "http://127.0.0.1:11435",
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
  "service": "llama.cpp",
  "status": "success",
  "predictions": [
    {
      "text": "A golden retriever sits on a red couch.",
      "emoji_mappings": [
        {"word": "retriever", "emoji": "🐕"}
      ]
    }
  ],
  "metadata": {
    "processing_time": 2.341,
    "model_info": {
      "framework": "llama.cpp",
      "server": "http://127.0.0.1:11435"
    }
  }
}
```

## Docker

There are two Dockerfiles. Use `Dockerfile` (the default) when `llama-server` runs on the host. Use `Dockerfile.server` when you need a fully self-contained image that builds and runs `llama-server` itself — e.g. RunPod or any environment where you can't rely on a host-side binary.

> Consolidating these into a single Dockerfile is tracked in `issues/containerize-llama-server-binary.md`.

### Dockerfile — Flask wrapper only (default)

`llama-server` runs on the host as `llama-cpp-server.service` and is reached via `--network=host`.

```bash
# Ensure llama-server is running first
systemctl status llama-cpp-server

# Build
docker build -t llama-cpp /home/sd/animal-farm/llama-cpp/

# Run (host networking so 127.0.0.1:11435 reaches llama-server)
docker run -d \
  --name llama-cpp \
  --network=host \
  --env-file /home/sd/animal-farm/llama-cpp/.env \
  llama-cpp

# Test
curl -s http://localhost:7782/health | python3 -m json.tool

curl -s -X POST -F "file=@/path/to/image.jpg" \
  http://localhost:7782/v3/analyze | python3 -m json.tool

# Delete
docker stop llama-cpp && docker rm llama-cpp && docker rmi llama-cpp
```

### Dockerfile.server — self-contained (llama-server + Flask)

Builds `llama-server` from source in a CUDA devel stage, then packages it with the Flask wrapper. No host binary required. Models are volume-mounted at runtime.

The `CUDA_ARCH` build arg defaults to `80;86;89;90` (A100, RTX 3090/4090, L40). Override for your specific GPU to reduce binary size and build time.

```bash
# Build (default multi-arch)
docker build -f Dockerfile.server -t llama-cpp-server /home/sd/animal-farm/llama-cpp/

# Build for a specific GPU (e.g. RTX 3090 = SM 8.6)
docker build -f Dockerfile.server --build-arg CUDA_ARCH=86 -t llama-cpp-server /home/sd/animal-farm/llama-cpp/

# Run (models volume-mounted, GPU passthrough required)
docker run -d \
  --name llama-cpp-server \
  --gpus all \
  --env-file /home/sd/animal-farm/llama-cpp/.env \
  -v /home/sd/animal-farm/llama-cpp/models:/app/models:ro \
  -p 7782:7782 \
  llama-cpp-server

# Delete
docker stop llama-cpp-server && docker rm llama-cpp-server && docker rmi llama-cpp-server
```

## Troubleshooting

**Problem**: Service exits at startup with `llama-server not reachable`
Start `llama-cpp-server.service` first. The Flask wrapper performs a connectivity check on boot.

**Problem**: Health returns `503`
llama-server is down or the `LLAMA_SERVER_HOST` is wrong.

**Problem**: Responses ignore the image (hallucination)
Ensure llama.cpp is v8100+. The legacy `/completion + image_data` API is broken in v8100 — the service uses `/v1/chat/completions` with `image_url` format instead.

---

**Documentation Version**: 1.0
**Last Updated**: 2026-03-13
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
