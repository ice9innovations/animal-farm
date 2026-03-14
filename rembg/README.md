# rembg Background Removal Service

**Port**: 7768
**Framework**: rembg 2.0.61 + ONNX Runtime
**Purpose**: AI-powered background removal returning RGBA images and alpha masks
**Status**: Active

## Overview

rembg removes image backgrounds using the rembg library with ONNX Runtime inference. Unlike BEN2 which uses a PyTorch model, rembg supports a wide range of pre-trained ONNX models via a simple session name. The default model is `birefnet-general`, which provides strong general-purpose results.

The service returns both an RGBA image (original pixels + alpha mask composited) and a standalone grayscale mask, both base64-encoded PNG.

## Features

- **Multiple Models**: Switch between rembg-supported models via `MODEL_NAME` in `.env`
- **Soft Alpha Mattes**: Float alpha channel for clean edges on hair, fur, and complex subjects
- **RGBA + Mask Output**: Both composited RGBA and isolated alpha mask returned
- **GPU or CPU**: Default Dockerfile uses `onnxruntime-gpu`; a CPU variant (`Dockerfile.cpu`) is available
- **Unified Input Handling**: Single endpoint for URL, file path, and file upload
- **Security**: File validation, 16MB size limit, type checking

## Available Models

rembg supports several session models. Set `MODEL_NAME` in `.env` to switch:

| Model | Description |
|-------|-------------|
| `birefnet-general` | General-purpose, strong on most subjects (default) |
| `birefnet-portrait` | Optimized for human portraits |
| `u2net` | Original U²-Net, good all-around |
| `u2net_human_seg` | U²-Net trained for human segmentation |
| `isnet-general-use` | IS-Net, high quality general use |
| `silueta` | Compact, fast model |

Models are downloaded automatically on first use to `~/.u2net/`.

## Installation

### Prerequisites

- Python 3.11+
- GPU: ONNX Runtime GPU (`onnxruntime-gpu`) — recommended
- CPU: ONNX Runtime (`onnxruntime`) — use `Dockerfile.cpu` for Docker

### Setup

```bash
cd /home/sd/animal-farm/rembg

python3 -m venv rembg_venv
source rembg_venv/bin/activate

pip install -r requirements.txt
```

## Configuration

### Environment Variables (.env)

```bash
PORT=7768
PRIVATE=false

# rembg session model name
MODEL_NAME=birefnet-general
```

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Service port |
| `PRIVATE` | Yes | `true` = localhost only |
| `MODEL_NAME` | Yes | rembg session model name |

## API Endpoints

### Health Check

```bash
GET /health
```

```json
{
  "status": "healthy",
  "session_status": "loaded",
  "model": "birefnet-general",
  "device": "cpu"
}
```

### Remove Background

```bash
GET  /v3/analyze?url=<image_url>
GET  /v3/analyze?file=<file_path>
POST /v3/analyze   (multipart file upload)
```

Also available at `/analyze` (no version prefix).

**Response:**
```json
{
  "service": "rembg",
  "status": "success",
  "mask": "<base64-encoded grayscale PNG>",
  "rgba": "<base64-encoded RGBA PNG>",
  "metadata": {
    "processing_time": 0.612,
    "width": 1920,
    "height": 1080,
    "model_info": {
      "model": "birefnet-general",
      "device": "cpu"
    }
  }
}
```

- **`rgba`**: Original pixels composited with the soft alpha mask — ready to place on any background
- **`mask`**: Grayscale PNG of the alpha channel alone — use for compositing, inpainting, or segmentation

## Service Management

### Manual Startup

```bash
cd /home/sd/animal-farm/rembg
./start.sh
```

### Systemd Service

```bash
sudo cp services/rembg-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rembg-api
sudo systemctl start rembg-api
systemctl status rembg-api
```

## Docker

Two Dockerfiles are provided:

| File | Runtime | Use case |
|------|---------|----------|
| `Dockerfile` | `onnxruntime-gpu` | Default — GPU acceleration |
| `Dockerfile.cpu` | `onnxruntime` | CPU-only environments |

rembg downloads its model weights to `~/.u2net/` on first run. Mount a host directory to cache them across container restarts.

### GPU (default)

```bash
# Build
docker build -t rembg-gpu /home/sd/animal-farm/rembg/

# Run
docker run -d \
  --name rembg-gpu \
  --gpus all \
  --env-file /home/sd/animal-farm/rembg/.env \
  -v /home/sd/.u2net:/root/.u2net \
  -p 7768:7768 \
  rembg-gpu

# Test (model downloads on first run)
curl -s http://localhost:7768/health | python3 -m json.tool
curl -s -X POST -F "file=@/path/to/image.jpg" http://localhost:7768/v3/analyze | python3 -m json.tool

# Delete
docker stop rembg-gpu && docker rm rembg-gpu && docker rmi rembg-gpu
```

### CPU

```bash
# Build
docker build -t rembg-cpu -f /home/sd/animal-farm/rembg/Dockerfile.cpu /home/sd/animal-farm/rembg/

# Run (no --gpus flag)
docker run -d \
  --name rembg-cpu \
  --env-file /home/sd/animal-farm/rembg/.env \
  -v /home/sd/.u2net:/root/.u2net \
  -p 7768:7768 \
  rembg-cpu
```

## Troubleshooting

**Problem**: Service slow on first request after startup
The model is being downloaded to `~/.u2net/`. Mount the volume as shown above to cache it.

**Problem**: `MODEL_NAME environment variable is required`
The `.env` file was not loaded. Verify it exists and contains `MODEL_NAME`.

**Problem**: GPU not used despite `onnxruntime-gpu` installed
`onnxruntime-gpu` selects `CUDAExecutionProvider` automatically when a GPU is available. Verify `nvidia-smi` shows the GPU and `--gpus all` was passed to `docker run`.

---

**Documentation Version**: 1.0
**Last Updated**: 2026-03-13
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
