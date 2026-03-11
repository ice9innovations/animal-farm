# BEN2 Background Removal Service

**Port**: 7769
**Framework**: BEN2 (Background Extraction Network 2) — PramaLLC/BEN2
**Purpose**: AI-powered background removal with soft alpha matting
**Status**: ✅ Active

## Overview

BEN2 removes image backgrounds using the BEN2_Base neural network. Unlike binary mask approaches, BEN2 produces soft alpha mattes with smooth edges — useful for hair, fur, and semi-transparent subjects. The service returns both an RGBA image and a standalone grayscale mask, both base64-encoded PNG.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for URL, file path, or multipart upload
- **Soft Alpha Mattes**: Float alpha channel, not binary black/white masks
- **RGBA + Mask Output**: Both the composited RGBA image and isolated alpha mask returned
- **Optional Foreground Refinement**: Slower but better edge quality for complex subjects
- **Security**: File validation, size limits (16MB), type checking
- **Performance**: GPU acceleration with CUDA support

## Installation

### Prerequisites

- Python 3.8+
- CUDA 12.2+ (for GPU acceleration)
- 4GB+ VRAM recommended
- BEN2_Base.pth weights file (downloaded separately)
- BEN2.py model class file (from ComfyUI or PramaLLC/BEN2 repo)

### 1. Environment Setup

```bash
cd /home/sd/animal-farm/ben2

python3 -m venv ben2_venv
source ben2_venv/bin/activate

pip install -r requirements.txt

# Install PyTorch with CUDA support separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Model Setup

BEN2 requires the weights file and source code to be obtained from PramaLLC/BEN2:

```bash
# Weights file — place at path specified by BEN2_MODEL_PATH
# e.g. /home/sd/bkg/BEN2_Base.pth

# BEN2.py model class — place in directory specified by BEN2_CODE_DIR
# e.g. /home/sd/ComfyUI/models/RMBG/BEN2/BEN2.py
```

The service loads `BEN2.py` dynamically from `BEN2_CODE_DIR` at startup.

## Configuration

### Environment Variables (.env)

Copy `.env.sample` to `.env` and fill in the required values:

```bash
# Service Configuration
PORT=7769                                          # Service port
PRIVATE=false                                      # Access mode (false=public, true=localhost-only)

# BEN2 Model Configuration (required)
BEN2_MODEL_PATH=/home/sd/bkg/BEN2_Base.pth        # Absolute path to weights file
BEN2_CODE_DIR=/home/sd/ComfyUI/models/RMBG/BEN2   # Directory containing BEN2.py

# Optional
REFINE_FOREGROUND=false                            # Enable refined foreground estimation (slower, better edges)
```

### Configuration Details

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Service listening port |
| `PRIVATE` | Yes | Access control (false=public, true=localhost-only) |
| `BEN2_MODEL_PATH` | Yes | Absolute path to BEN2_Base.pth weights file |
| `BEN2_CODE_DIR` | Yes | Directory containing BEN2.py model class |
| `REFINE_FOREGROUND` | No | Enable foreground refinement pass (default: false) |

Both path variables are validated at startup — the service exits immediately if either path does not exist.

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
  "device": "cuda:0",
  "refine_foreground": false
}
```

### Remove Background (Unified Endpoint)

Both `/analyze` and `/v3/analyze` are active. Exactly one input method must be provided.

#### From URL
```bash
GET /v3/analyze?url=<image_url>
```

```bash
curl "http://localhost:7769/v3/analyze?url=https://example.com/photo.jpg"
```

#### From File Path
```bash
GET /v3/analyze?file=<file_path>
```

```bash
curl "http://localhost:7769/v3/analyze?file=/path/to/photo.jpg"
```

#### POST File Upload
```bash
POST /v3/analyze
Content-Type: multipart/form-data
```

```bash
curl -X POST -F "file=@/path/to/photo.jpg" http://localhost:7769/v3/analyze
```

**Input Validation:**
- Exactly one parameter required (`url` OR `file`, or a multipart upload)
- Cannot provide both `url` and `file` simultaneously

**Response Format:**
```json
{
  "service": "ben2",
  "status": "success",
  "mask": "<base64-encoded grayscale PNG>",
  "rgba": "<base64-encoded RGBA PNG>",
  "metadata": {
    "processing_time": 0.84,
    "width": 1920,
    "height": 1080,
    "model_info": {
      "model": "BEN2_Base",
      "refine_foreground": false
    }
  }
}
```

- **`mask`**: Grayscale PNG — the alpha channel only (soft edges, float precision)
- **`rgba`**: RGBA PNG — original pixels composited with the soft alpha mask

## Service Management

### Manual Startup

```bash
cd /home/sd/animal-farm/ben2
./start.sh
```

### Systemd Service

```bash
# Install service
sudo cp services/ben2-api.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start/stop service
sudo systemctl start ben2-api
sudo systemctl stop ben2-api

# Enable auto-start
sudo systemctl enable ben2-api

# Check status
systemctl status ben2-api

# View logs
sudo journalctl -u ben2-api -f
```

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: 16MB (larger than other services — background removal benefits from full resolution)
- **Input Methods**: URL, file path, multipart upload

### Output
- **`rgba`**: RGBA PNG with soft alpha composited — ready for placement on any background
- **`mask`**: Grayscale PNG of the alpha channel alone — use for compositing, inpainting, or segmentation pipelines

## Performance

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | GTX 1060 6GB | RTX 3080+ | CUDA required for reasonable speed |
| RAM | 8GB | 16GB+ | Large images are held in memory |
| Storage | 500MB | 1GB+ | Weights + venv |

### Refine Foreground

Set `REFINE_FOREGROUND=true` in `.env` for improved edge quality at the cost of slower inference. Useful for hair, fur, or semi-transparent subjects. Default is `false`.

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `BEN2_MODEL_PATH does not exist` | Weights file missing | Download BEN2_Base.pth and set correct path |
| `BEN2_CODE_DIR does not exist` | BEN2.py directory missing | Clone PramaLLC/BEN2 or set correct ComfyUI path |
| `Cannot import BEN2` | BEN2.py not found in code dir | Verify BEN2.py exists in `BEN2_CODE_DIR` |
| `File too large` | File exceeds 16MB | Resize image before submission |
| `CUDA out of memory` | Insufficient VRAM | Reduce image resolution |

### Error Response Format

```json
{
  "service": "ben2",
  "status": "error",
  "error": {"message": "File not found: /path/to/image.jpg"},
  "metadata": {"processing_time": 0.001}
}
```

## Integration Examples

### Python Integration

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# URL input
response = requests.get(
    "http://localhost:7769/v3/analyze",
    params={"url": "https://example.com/photo.jpg"}
)

# File path input
response = requests.get(
    "http://localhost:7769/v3/analyze",
    params={"file": "/path/to/photo.jpg"}
)

# POST file upload
with open('/path/to/photo.jpg', 'rb') as f:
    response = requests.post(
        "http://localhost:7769/v3/analyze",
        files={'file': f}
    )

result = response.json()
if result["status"] == "success":
    rgba_bytes = base64.b64decode(result["rgba"])
    rgba_image = Image.open(BytesIO(rgba_bytes))
    rgba_image.save("output.png")

    mask_bytes = base64.b64decode(result["mask"])
    mask_image = Image.open(BytesIO(mask_bytes))
    mask_image.save("mask.png")
```

### JavaScript Integration

```javascript
// URL input
const response = await fetch(
  'http://localhost:7769/v3/analyze?url=https://example.com/photo.jpg'
);

// POST file upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
const response = await fetch('http://localhost:7769/v3/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
if (result.status === 'success') {
  // result.rgba is a base64 PNG — use directly in an <img> tag
  document.getElementById('output').src = `data:image/png;base64,${result.rgba}`;
}
```

## Troubleshooting

### Service Won't Start

**Problem**: `BEN2_MODEL_PATH does not exist`
```bash
# Verify the weights file path
ls -lh /home/sd/bkg/BEN2_Base.pth

# Update BEN2_MODEL_PATH in .env to correct location
```

**Problem**: `Cannot import BEN2`
```bash
# Verify BEN2.py exists in the code directory
ls /home/sd/ComfyUI/models/RMBG/BEN2/BEN2.py

# Update BEN2_CODE_DIR in .env to the directory containing BEN2.py
```

**Problem**: Service fails to start, port in use
```bash
netstat -tlnp | grep 7769
```

### Runtime Issues

**Problem**: CUDA out of memory
- Reduce input image resolution before sending
- Disable `REFINE_FOREGROUND` if enabled

**Problem**: Slow inference on CPU
- CUDA is required for practical throughput — CPU inference is very slow
- Verify `nvidia-smi` shows the GPU is available and `nvcc --version` confirms CUDA

## Security Considerations

### Access Control
- Set `PRIVATE=true` to restrict to localhost only
- Use a reverse proxy with authentication for public-facing deployments

### File Security
- File type validation on all inputs
- 16MB size cap enforced at both Flask and application layers
- URL inputs validated for scheme and content-type before loading

---

**Documentation Version**: 1.0
**Last Updated**: 2026-03-11
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
