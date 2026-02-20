# Moondream Image Captioning Service

**Port**: 7795
**Framework**: Moondream2 (vikhyatk/moondream2) via HuggingFace Transformers
**Purpose**: Lightweight vision-language model image captioning with emoji mapping
**Status**: Active

## Overview

Moondream provides image captioning using vikhyatk/moondream2, a small and efficient vision-language model. The model runs entirely locally on GPU via HuggingFace Transformers — no cloud API calls are made during inference.

## Features

- **V3 API**: Unified endpoint matching the Animal Farm platform standard
- **Unified Input Handling**: Single endpoint for URL, file path, and file upload
- **Emoji Integration**: Automatic word-to-emoji mapping using local dictionary
- **Local Inference**: Model runs on-device, no external API required after initial download
- **GPU Acceleration**: CUDA support via `device_map`

## Installation

### 1. Get the moondream local inference source

The local inference library lives in the GitHub repo, not the PyPI package:

```bash
git clone https://github.com/vikhyat/moondream.git /tmp/moondream_src
cp -r /tmp/moondream_src/moondream /home/sd/animal-farm/moondream/moondream
rm -rf /tmp/moondream_src
```

### 2. Environment Setup

```bash
cd /home/sd/animal-farm/moondream

python3 -m venv moondream_venv
source moondream_venv/bin/activate

# Install PyTorch with CUDA support - use the index matching your CUDA version
# This machine uses cu128:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
cp .env.sample .env
# Edit .env with your settings
```

### 4. Model Download

The model downloads automatically from HuggingFace on first startup (~1.8GB). Cached to `~/.cache/huggingface/`.

## Configuration

### Environment Variables (.env)

```bash
PORT=7795
PRIVATE=False
MODEL_REVISION=2025-06-21
```

### Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | `False`=public, `True`=localhost-only |
| `MODEL_REVISION` | Yes | - | HuggingFace model revision to pin |
| `MODEL_ID` | No | `vikhyatk/moondream2` | HuggingFace model ID |
| `CAPTION_LENGTH` | No | `normal` | `short` or `normal` |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "device": "cuda"
}
```

### Analyze Image

#### From URL
```bash
GET /analyze?url=<image_url>
GET /v3/analyze?url=<image_url>
```

#### From File Path
```bash
GET /analyze?file=<file_path>
GET /v3/analyze?file=<file_path>
```

#### POST File Upload
```bash
POST /analyze
Content-Type: multipart/form-data
```

**Input Validation:**
- Exactly one of `url` or `file` must be provided
- Returns 400 if neither or both are provided

**Response Format:**
```json
{
  "service": "moondream",
  "status": "success",
  "predictions": [
    {
      "text": "a dog sitting on a wooden floor",
      "emoji_mappings": [
        {"word": "dog", "emoji": "🐶"},
        {"word": "wooden", "emoji": "🌲"},
        {"word": "floor", "emoji": "🏠"}
      ]
    }
  ],
  "metadata": {
    "processing_time": 0.821,
    "model_info": {"framework": "Moondream2 (vikhyatk/moondream2)"}
  }
}
```

**Error Response:**
```json
{
  "service": "moondream",
  "status": "error",
  "predictions": [],
  "error": {"message": "File not found: /path/to/image.jpg"},
  "metadata": {"processing_time": 0.001}
}
```

## Service Management

### Manual Startup

```bash
cd /home/sd/animal-farm/moondream
./start.sh
```

### Systemd Service

```bash
sudo cp services/moondream-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start moondream-api
sudo systemctl enable moondream-api
sudo journalctl -u moondream-api -f
```

## Supported Formats

- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: 8MB

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 4GB VRAM | 6GB+ VRAM |
| RAM | 8GB | 16GB |
| Storage | 3GB | 5GB (model cache) |

## Troubleshooting

**Problem**: `trust_remote_code` error
**Solution**: Ensure `transformers` is up to date: `pip install -U transformers`

**Problem**: CUDA out of memory
**Solution**: Moondream2 requires ~2GB VRAM. Ensure no other large models are loaded.

**Problem**: Slow first startup
**Solution**: Model downloads ~1.8GB on first run. Subsequent starts use the local cache.
