# SAM3 Segmentation Service

**Port**: 9779
**Framework**: Meta SAM3 (Segment Anything Model 3)
**Purpose**: Text-prompted image segmentation driven by noun consensus
**Status**: Active

## Overview

SAM3 is Meta's open-vocabulary segmentation model. Given a text noun and an image, it finds and segments all instances of that concept. In Animal Farm, it is triggered automatically by the noun consensus worker after all VLM services have reported, using the high-confidence nouns as prompts.

The pipeline:
```
VLMs (blip, moondream, ollama)
    → noun_consensus_worker (confidence > 0.5)
        → SAM3 (one prompt per noun)
            → sam3_results table
```

## Features

- **Open-vocabulary prompting**: Any noun, no fixed class list
- **Multi-instance detection**: Finds all instances of a concept in one pass
- **Per-instance masks**: RLE-encoded binary masks alongside bounding boxes
- **Automatic triggering**: Fired by the Windmill pipeline, no manual intervention needed
- **Presence token**: Distinguishes visually similar concepts (e.g. "player in white" vs "player in red")

## Installation

### Prerequisites

- Python 3.12+
- PyTorch 2.7+ (cu118 wheel is compatible with CUDA 12.2)
- NVIDIA GPU with 6GB+ VRAM (tested on RTX 3090 Ti)
- HuggingFace account with approved access to `facebook/sam3`

### 1. Clone SAM3

```bash
git clone https://github.com/facebookresearch/sam3.git /home/sd/sam3
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv /home/sd/sam3/sam3_venv
```

### 3. Install PyTorch (CUDA 12.2 compatible)

```bash
/home/sd/sam3/sam3_venv/bin/pip install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install SAM3 and Dependencies

```bash
cd /home/sd/sam3
/home/sd/sam3/sam3_venv/bin/pip install -e .
/home/sd/sam3/sam3_venv/bin/pip install flask flask-cors python-dotenv requests pillow \
    einops decord pycocotools psutil opencv-python scikit-image scikit-learn
/home/sd/sam3/sam3_venv/bin/pip install "numpy>=1.26,<2"
```

> **Note**: A patch is required to `sam3/model_builder.py` — move the top-level
> `from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor` import
> inside the `if enable_inst_interactivity:` block. Without this patch, the import
> unconditionally pulls in training data dependencies (pycocotools, decord, etc.)
> even for inference-only use.

### 5. Download Checkpoint

```bash
hf auth login   # requires approved access to facebook/sam3

/home/sd/sam3/sam3_venv/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt',
                local_dir='/home/sd/sam3/checkpoints')
"
```

The checkpoint is 3.3GB.

## Configuration

### Environment Variables (.env)

```bash
PORT=9779
PRIVATE=true
SAM3_PATH=/home/sd/sam3
SAM3_CHECKPOINT=/home/sd/sam3/checkpoints/sam3.pt
CONFIDENCE_THRESHOLD=0.5
```

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 9779 | Service listening port |
| `PRIVATE` | true | Bind to 127.0.0.1 only |
| `SAM3_PATH` | /home/sd/sam3 | Path to SAM3 source |
| `SAM3_CHECKPOINT` | — | Path to sam3.pt; omit to auto-download from HuggingFace cache |
| `CONFIDENCE_THRESHOLD` | 0.5 | Minimum detection confidence for returned instances |

## API Endpoints

### Health Check

```bash
GET /health
```

```json
{
  "status": "healthy",
  "model": "SAM3",
  "device": "cuda",
  "model_loaded": true,
  "confidence_threshold": 0.5
}
```

### Analyze Image

```bash
POST /analyze
```

Accepts an image via file upload, JSON `image_url`, JSON `image_b64`, or JSON `file` path.
Nouns are provided as a JSON array or comma-separated string.

**File upload:**
```bash
curl -X POST http://localhost:9779/analyze \
  -F "file=@image.jpg" \
  -F 'nouns=["cat","box"]'
```

**JSON with URL:**
```bash
curl -X POST http://localhost:9779/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_url": "http://example.com/image.jpg", "nouns": ["cat", "box"]}'
```

**Response:**
```json
{
  "service": "sam3",
  "status": "success",
  "nouns_queried": ["cat", "box"],
  "results": {
    "cat": {
      "instances": [
        {
          "score": 0.9648,
          "bbox": {"x": 177, "y": 119, "width": 555, "height": 478},
          "mask_rle": [163641, 31, ...],
          "mask_shape": [1024, 768]
        }
      ]
    },
    "box": {
      "instances": [
        {
          "score": 0.9375,
          "bbox": {"x": 0, "y": 211, "width": 241, "height": 445},
          "mask_rle": [...],
          "mask_shape": [1024, 768]
        }
      ]
    }
  },
  "image_dimensions": {"width": 768, "height": 1024},
  "metadata": {
    "processing_time": 2.083,
    "confidence_threshold": 0.5,
    "total_instances": 4
  }
}
```

The `mask_rle` field is a run-length encoded binary mask. Values alternate between
zero-pixel counts and one-pixel counts, flattened in row-major order.

## Service Management

### Manual Startup

```bash
/home/sd/animal-farm/sam3/start.sh
```

### Systemd Service

```bash
sudo cp services/sam3-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start sam3-api
sudo systemctl enable sam3-api
sudo journalctl -u sam3-api -f
```

## Windmill Integration

SAM3 is wired into the Windmill pipeline as a system service. It is triggered
automatically by the `noun_consensus_worker` when all configured VLM services
(blip, moondream, ollama) have reported for an image.

Only nouns with confidence > 0.5 (strict majority vote) are sent as prompts.
Results are stored in the `sam3_results` table and exposed via the Windmill API
at `/status/<image_id>` and `/results/<image_id>` under the `sam3` key.

**Windmill service_config.yaml entry:**
```yaml
system:
  sam3:
    queue_name: sam3
    category: infrastructure
    service_type: sam3
    host: localhost
    port: 9779
    endpoint: /analyze
```

## Performance

Tested on RTX 3090 Ti (24GB VRAM, CUDA 12.2):

| Nouns | Instances found | Time |
|-------|----------------|------|
| 1 | 1 | ~0.8s |
| 2 | 4 | ~2.1s |

Model size: 848M parameters (~3.3GB checkpoint). VRAM usage at inference is
approximately 6–8GB depending on image resolution.

## Supported Input Formats

- PNG, JPG, JPEG, WebP, BMP, GIF
- Max recommended size: 8MB
- Input methods: multipart file upload, URL, base64, local file path

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: No module named 'einops'` | Missing inference dep | `pip install einops decord pycocotools psutil` |
| `TypeError: unsupported operand type(s) for \|` | Python < 3.10 syntax | Use Python 3.12 venv |
| `PRECONDITION_FAILED` on queue declare | Queue arg mismatch | Restart sam3_worker after any queue changes |
| `sam3: null` in API results | Queried before SAM3 finished | Check `/results/<image_id>` after `sam3_complete: true` in status |
| Model not loading | Checkpoint path wrong | Verify `SAM3_CHECKPOINT` in `.env` points to `sam3.pt` |

## Docker

```bash
# Build (clones SAM3 from GitHub at build time)
docker build -t sam3 /home/sd/animal-farm/sam3/

# Run (mounts checkpoint dir; overrides host paths in .env)
docker run -d \
  --name sam3 \
  --gpus all \
  -v /home/sd/sam3/checkpoints:/app/sam3/checkpoints:ro \
  -e PORT=9779 \
  -e PRIVATE=true \
  -e SAM3_CHECKPOINT=/app/sam3/checkpoints/sam3.pt \
  -e CONFIDENCE_THRESHOLD=0.5 \
  -p 9779:9779 \
  sam3

# Test
curl -s http://localhost:9779/health | python3 -m json.tool

# Delete
docker stop sam3 && docker rm sam3 && docker rmi sam3
```

> `-e` flags are used instead of `--env-file` because the `.env` contains host-specific paths that differ inside the container.

---

**Last Updated**: 2026-02-20
**Checkpoint**: facebook/sam3 (gated — HuggingFace access required)
**Maintainer**: Animal Farm ML Team
