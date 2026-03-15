# rembg Background Removal Service

**Port**: 7768
**Purpose**: Background removal returning alpha masks
**Status**: Active

## Overview

The rembg service provides background removal with two selectable backends, chosen at startup via the `BACKEND` environment variable:

| Backend | Runtime | Device | Description |
|---------|---------|--------|-------------|
| `rembg` | ONNX Runtime | CPU or GPU | rembg 2.0.61 with configurable ONNX session models |
| `ben2` | PyTorch | GPU | BEN2 (Background Extraction Network 2) with soft float alpha mattes |

Both backends return an identical API response — a grayscale alpha mask as a base64-encoded PNG.

## Configuration

### Environment Variables (.env)

```bash
PORT=7768
PRIVATE=false

# Backend selector
BACKEND=rembg

# rembg backend
MODEL_NAME=birefnet-general

# ben2 backend
BEN2_MODEL_PATH=/home/sd/bkg/BEN2_Base.pth
BEN2_CODE_DIR=/home/sd/ComfyUI/models/RMBG/BEN2
# REFINE_FOREGROUND=false
```

| Variable | Required | When |
|----------|----------|------|
| `PORT` | Yes | Always |
| `PRIVATE` | Yes | Always |
| `BACKEND` | Yes | Always — `rembg` or `ben2` |
| `MODEL_NAME` | Yes | `BACKEND=rembg` |
| `BEN2_MODEL_PATH` | Yes | `BACKEND=ben2` |
| `BEN2_CODE_DIR` | Yes | `BACKEND=ben2` |
| `REFINE_FOREGROUND` | No | `BACKEND=ben2` — enables refined foreground estimation (slower, better edges) |

### rembg Models

Set `MODEL_NAME` in `.env` to switch models. Models download automatically to `~/.u2net/` on first use.

| Model | Description |
|-------|-------------|
| `birefnet-general` | General-purpose (default) |
| `birefnet-portrait` | Optimized for human portraits |
| `u2net` | Original U²-Net |
| `u2net_human_seg` | U²-Net for human segmentation |
| `isnet-general-use` | IS-Net, high quality general use |
| `silueta` | Compact, fast |

## Installation

### rembg backend

```bash
cd /home/sd/animal-farm/rembg
python3 -m venv rembg_venv
source rembg_venv/bin/activate
pip install -r requirements.txt
```

### ben2 backend

```bash
cd /home/sd/animal-farm/rembg
python3 -m venv rembg_venv
source rembg_venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## API Endpoints

### Health Check

```bash
GET /health
```

```json
{
  "status": "healthy",
  "backend": "rembg",
  "backend_status": "loaded",
  "device": "cpu",
  "model": "birefnet-general"
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

Three Dockerfiles are provided:

| File | Backend | Runtime |
|------|---------|---------|
| `Dockerfile.gpu` | `rembg` | `onnxruntime-gpu` |
| `Dockerfile.cpu` | `rembg` | `onnxruntime` (CPU only) |
| `Dockerfile.ben2` | `ben2` | PyTorch + CUDA |

### rembg GPU (default)

```bash
docker build -t rembg-gpu /home/sd/animal-farm/rembg/

docker run -d \
  --name rembg-gpu \
  --gpus all \
  --env-file /home/sd/animal-farm/rembg/.env \
  -v /home/sd/.u2net:/root/.u2net \
  -p 7768:7768 \
  rembg-gpu
```

### rembg CPU

```bash
docker build -t rembg-cpu -f /home/sd/animal-farm/rembg/Dockerfile.cpu /home/sd/animal-farm/rembg/

docker run -d \
  --name rembg-cpu \
  --env-file /home/sd/animal-farm/rembg/.env \
  -v /home/sd/.u2net:/root/.u2net \
  -p 7768:7768 \
  rembg-cpu
```

### ben2

```bash
docker build -t rembg-ben2 -f /home/sd/animal-farm/rembg/Dockerfile.ben2 /home/sd/animal-farm/rembg/

docker run -d \
  --name rembg-ben2 \
  --gpus all \
  --env-file /home/sd/animal-farm/rembg/.env \
  -v /home/sd/bkg:/models \
  -p 7768:7768 \
  rembg-ben2
```

Model weights are large — mount them via volume rather than baking into the image.

## Troubleshooting

**Problem**: `BACKEND environment variable is required`
The `.env` file was not loaded or `BACKEND` is missing.

**Problem**: `BEN2_MODEL_PATH does not exist`
The weights file path in `.env` is wrong or the file has not been downloaded.

**Problem**: Service slow on first request (rembg backend)
The model is being downloaded to `~/.u2net/`. Mount the volume as shown above to cache it across restarts.

---

**Last Updated**: 2026-03-15
