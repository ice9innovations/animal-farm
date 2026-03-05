# CLIP Caption Scoring Service

**Port**: 7776
**Framework**: OpenAI CLIP (ViT-L/14)
**Purpose**: Cosine similarity scoring between an image and a caption string
**Status**: ✅ Active

## Overview

clip-score computes how well a caption describes an image using CLIP's cosine similarity between image and text embeddings. It is a focused extraction of the scoring logic from the CLIP classification service — no label files, no embedding cache, no classification overhead.

Typical use: pass a VLM-generated caption (e.g. from claude-api, llama-cpp, or moondream) and the source image to get an objective quality score between 0 and 1.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)
- CLIP installed at `/home/sd/CLIP/clip` (symlinked into service directory)

### Setup

```bash
cd /home/sd/animal-farm/clip-score

# Symlink the CLIP package (already done)
ln -s /home/sd/CLIP/clip ./clip

# Create .env from sample
cp .env.sample .env

# Use the shared CLIP venv (already contains torch, clip-by-openai, flask)
# rest.sh points to /home/sd/CLIP/clip_venv automatically
```

## Configuration

### Environment Variables (.env)

```bash
PORT=7776
PRIVATE=False
CLIP_MODEL=ViT-L/14
```

### Configuration Details

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Service listening port |
| `PRIVATE` | Yes | `False` = all interfaces, `True` = localhost only |
| `CLIP_MODEL` | Yes | CLIP model variant — must match the CLIP classification service for comparable scores |

### Model Options

| Model | VRAM | Notes |
|-------|------|-------|
| `ViT-B/32` | ~4GB | Faster, less accurate |
| `ViT-L/14` | ~8GB (4GB FP16) | Production default |
| `ViT-L/14@336px` | ~10GB | Higher resolution input |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "clip-score",
  "model": "ViT-L/14",
  "device": "cuda"
}
```

### Score Caption

```bash
GET /v3/score?caption=<text>&url=<image_url>
GET /v3/score?caption=<text>&file=<file_path>
POST /v3/score  (multipart: caption + file)
```

Also available at `/score` (no version prefix).

#### URL Example
```bash
curl "http://localhost:7776/v3/score?caption=a+dog+in+the+park&url=http://example.com/image.jpg"
```

#### File Path Example
```bash
curl "http://localhost:7776/v3/score?caption=a+dog+in+the+park&file=/path/to/image.jpg"
```

#### Multipart Upload Example
```bash
curl -X POST \
  -F "caption=a dog in the park" \
  -F "file=@/path/to/image.jpg" \
  http://localhost:7776/v3/score
```

**Input Validation:**
- `caption` is required and must be non-empty
- Exactly one image source must be provided (`url`, `file`, or multipart upload)
- Cannot provide both `url` and `file`

**Response Format:**
```json
{
  "service": "clip-score",
  "status": "success",
  "similarity_score": 0.2814,
  "caption": "a dog in the park",
  "metadata": {
    "processing_time": 0.183,
    "model_info": {
      "framework": "openai-clip",
      "model": "ViT-L/14",
      "device": "cuda"
    }
  }
}
```

**Score interpretation**: Scores typically range from 0.15 (poor match) to 0.35 (strong match). Scores above 0.28 indicate a good caption.

**Error Response:**
```json
{
  "service": "clip-score",
  "status": "error",
  "similarity_score": null,
  "error": {"message": "Must provide non-empty 'caption' parameter"},
  "metadata": {"processing_time": 0.001}
}
```

## Service Management

### Manual Startup

```bash
cd /home/sd/animal-farm/clip-score
./rest.sh
```

### Systemd Service

```bash
# Install
sudo cp services/clip-score.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start / stop
sudo systemctl start clip-score
sudo systemctl stop clip-score

# Enable auto-start on boot
sudo systemctl enable clip-score

# Check status
sudo systemctl status clip-score

# View logs
sudo journalctl -u clip-score -f
```

## Integration Examples

### Python

```python
import requests

# Score a caption against a URL
response = requests.get(
    "http://localhost:7776/v3/score",
    params={
        "caption": "a man walking a dog",
        "url": "https://example.com/image.jpg"
    }
)
result = response.json()
print(result["similarity_score"])  # e.g. 0.2814

# Score a caption via multipart upload
with open("/path/to/image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:7776/v3/score",
        data={"caption": "a man walking a dog"},
        files={"file": f}
    )
result = response.json()
print(result["similarity_score"])
```

### JavaScript

```javascript
// URL-based scoring
const response = await fetch(
  'http://localhost:7776/v3/score?caption=a+man+walking+a+dog&url=https://example.com/image.jpg'
);
const result = await response.json();
console.log(result.similarity_score);

// Multipart upload
const formData = new FormData();
formData.append('caption', 'a man walking a dog');
formData.append('file', fileInput.files[0]);
const response = await fetch('http://localhost:7776/v3/score', {
  method: 'POST',
  body: formData
});
const result = await response.json();
console.log(result.similarity_score);
```

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'clip'`
The `clip` symlink is missing. Run:
```bash
ln -s /home/sd/CLIP/clip /home/sd/animal-farm/clip-score/clip
```

**Problem**: Service fails to start — port already in use
```bash
lsof -ti :7776 | xargs kill
```

**Problem**: CUDA out of memory
Switch to `ViT-B/32` in `.env` — it uses roughly half the VRAM.

**Problem**: Low scores on clearly correct captions
Ensure `CLIP_MODEL` matches the model used when scores were originally calibrated. Scores are not comparable across model variants.

---

**Last Updated**: 2026-02-20
**Service Version**: Production
