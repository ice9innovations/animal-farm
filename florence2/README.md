# Florence-2 Multi-Task Vision Service

**Port**: 7803
**Framework**: Microsoft Florence-2 via Hugging Face Transformers
**Purpose**: Multi-task vision analysis — captioning, object detection, OCR, phrase grounding, and segmentation from a single model
**Status**: Active

## Overview

Florence-2 is a prompt-based vision foundation model that handles ten different analysis tasks through a single unified model. Unlike single-purpose VLMs, Florence-2 switches behavior based on a task token, making it possible to get captions, bounding boxes, OCR text, or segmentation polygons from the same GPU-resident model without loading multiple specialized models.

## Task Reference

All tasks are invoked via the `task` query parameter. The default task is `DENSE_REGION_CAPTION`.

### Image-Only Tasks (no extra parameters needed)

| Task | Description | Output |
|------|-------------|--------|
| `CAPTION` | Brief single-sentence caption | `text` string |
| `DETAILED_CAPTION` | Richer description of the scene | `text` string |
| `MORE_DETAILED_CAPTION` | Exhaustive scene description | `text` string |
| `OD` | Object detection — bounding boxes with class labels | `label` + `bbox` per object |
| `DENSE_REGION_CAPTION` | Bounding boxes with descriptive captions per region (default) | `label` + `bbox` per region |
| `OCR` | Extract all text in the image as a single string | `text` string |
| `OCR_WITH_REGION` | Extract text blocks with their locations | `text` + `quad_box` per block |

### Tasks Requiring a `text` Parameter

These tasks use the image plus a text prompt to perform grounded analysis.

| Task | `text` input | Output |
|------|-------------|--------|
| `CAPTION_TO_PHRASE_GROUNDING` | A caption sentence (e.g. "a dog on a couch") | `label` + `bbox` for each phrase grounded in the image |
| `OPEN_VOCABULARY_DETECTION` | Space-separated class names (e.g. "cat dog person") | `label` + `bbox` for each detected class |
| `REFERRING_EXPRESSION_SEGMENTATION` | A description of one region (e.g. "the red car on the left") | `label` + `polygon` for the described region |

### Deferred Tasks

| Task | Reason deferred |
|------|----------------|
| `REGION_TO_SEGMENTATION` | Requires a bounding box input — not yet implemented |

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.x (for GPU acceleration)
- ~1.5GB disk space for `Florence-2-large` model weights (auto-downloaded on first run)

### Virtual Environment Setup

```bash
cd /home/sd/animal-farm/florence2

python3 -m venv florence2_venv
source florence2_venv/bin/activate

# Torch must be installed first with the CUDA index (cu128 bundles its own runtime)
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Remaining dependencies
pip install -r requirements.txt --no-index-url
# Or just:
pip install transformers==4.56.1 timm==1.0.25 einops==0.8.2 \
    flask==3.1.3 flask-cors==6.0.2 python-dotenv requests Pillow nltk
```

### Configuration

```bash
cp .env.sample .env
# Edit .env as needed — defaults are sensible for most setups
```

### First Run

Model weights are downloaded automatically from Hugging Face on first run (~1.5GB for Florence-2-large). Set `HF_HOME` or `TRANSFORMERS_CACHE` env var to control where they land.

```bash
source florence2_venv/bin/activate
python REST.py
```

### systemd Service

```bash
sudo cp services/florence2-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable florence2-api
sudo systemctl start florence2-api
sudo systemctl status florence2-api
```

## API Endpoints

### Health Check

```
GET /health
```

```json
{
  "status": "healthy",
  "model_status": "loaded",
  "model": "microsoft/Florence-2-large",
  "device": "cuda:0",
  "default_task": "DENSE_REGION_CAPTION",
  "valid_tasks": ["CAPTION", "CAPTION_TO_PHRASE_GROUNDING", "..."]
}
```

### Analyze Image (single task)

```
GET  /analyze?url=<image_url>[&task=<TASK>][&text=<text>]
POST /analyze        (multipart file upload)
```

Parameters:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `url` | One of `url`/`file`/POST | Image URL |
| `file` | One of `url`/`file`/POST | Local file path |
| `task` | No | Task name (default: `DENSE_REGION_CAPTION`) |
| `text` | For text-input tasks | Text prompt for grounded tasks |

### Analyze Image (batch — multiple tasks, image encoded once)

```
POST /analyze/batch
```

Image supplied the same way as the single-task endpoint (`?url=`, `?file=`, or multipart upload). Task list is JSON in the request body.

```json
{
  "tasks": [
    {"task": "MORE_DETAILED_CAPTION"},
    {"task": "OD"},
    {"task": "OCR_WITH_REGION"},
    {"task": "CAPTION_TO_PHRASE_GROUNDING", "text": "A pangolin walking across a dirt field."}
  ]
}
```

Response:

```json
{
  "service": "florence2",
  "status": "success",
  "results": {
    "MORE_DETAILED_CAPTION": {"predictions": [...], "processing_time": 1.41},
    "OD":                    {"predictions": [...], "processing_time": 0.04},
    "OCR_WITH_REGION":       {"predictions": [...], "processing_time": 0.10},
    "CAPTION_TO_PHRASE_GROUNDING": {"predictions": [...], "processing_time": 0.08}
  },
  "metadata": {
    "total_processing_time": 1.63,
    "task_count": 4,
    "succeeded": 4,
    "failed": 0,
    "model_info": {"model": "microsoft/Florence-2-base", "framework": "Florence-2"}
  }
}
```

- `status` is `"success"` when all tasks succeed, `"partial"` when some fail, `"error"` when all fail.
- Failed tasks return `{"error": "..."}` instead of `predictions`; other tasks are unaffected.
- Duplicate task names in the request are keyed as `TASK_2`, `TASK_3`, etc.

## Response Format

All responses use the standard Animal Farm envelope. The `predictions` array shape varies by task.

### Caption task (`CAPTION`, `DETAILED_CAPTION`, `MORE_DETAILED_CAPTION`, `OCR`)

```json
{
  "service": "florence2",
  "status": "success",
  "predictions": [
    {
      "task": "CAPTION",
      "text": "a dog sitting on a red couch",
      "emoji_mappings": [
        {"emoji": "🐕", "word": "dog"},
        {"emoji": "🛋️", "word": "couch"}
      ]
    }
  ],
  "metadata": {
    "processing_time": 0.412,
    "task": "CAPTION",
    "model_info": {"model": "microsoft/Florence-2-large", "framework": "Florence-2"}
  }
}
```

### Detection task (`OD`, `DENSE_REGION_CAPTION`, `CAPTION_TO_PHRASE_GROUNDING`, `OPEN_VOCABULARY_DETECTION`)

One prediction per detected region:

```json
{
  "service": "florence2",
  "status": "success",
  "predictions": [
    {
      "task": "DENSE_REGION_CAPTION",
      "label": "a golden retriever sitting on a red fabric couch",
      "bbox": [42.5, 130.2, 398.1, 510.7],
      "emoji_mappings": [
        {"emoji": "🐕", "word": "retriever"},
        {"emoji": "🛋️", "word": "couch"}
      ]
    },
    {
      "task": "DENSE_REGION_CAPTION",
      "label": "a wooden coffee table",
      "bbox": [210.0, 480.0, 560.0, 600.0],
      "emoji_mappings": [
        {"emoji": "🌲", "word": "wooden"}
      ]
    }
  ],
  "metadata": {
    "processing_time": 0.815,
    "task": "DENSE_REGION_CAPTION",
    "model_info": {"model": "microsoft/Florence-2-large", "framework": "Florence-2"}
  }
}
```

Bounding boxes are in `[x1, y1, x2, y2]` pixel coordinates (absolute, not normalized).

### OCR with regions (`OCR_WITH_REGION`)

```json
{
  "predictions": [
    {
      "task": "OCR_WITH_REGION",
      "text": "STOP",
      "quad_box": [102.0, 45.0, 210.0, 45.0, 210.0, 98.0, 102.0, 98.0]
    }
  ]
}
```

`quad_box` is `[x1,y1, x2,y2, x3,y3, x4,y4]` — four corner points clockwise from top-left.

### Referring segmentation (`REFERRING_EXPRESSION_SEGMENTATION`)

```json
{
  "predictions": [
    {
      "task": "REFERRING_EXPRESSION_SEGMENTATION",
      "label": "the red car on the left",
      "polygon": [[120.5, 200.3], [340.1, 200.3], [340.1, 410.7], [120.5, 410.7]]
    }
  ]
}
```

## Example Requests

```bash
# Default task (DENSE_REGION_CAPTION)
curl "http://localhost:7803/analyze?url=https://example.com/image.jpg"

# Plain caption
curl "http://localhost:7803/analyze?url=https://example.com/image.jpg&task=CAPTION"

# Object detection
curl "http://localhost:7803/analyze?url=https://example.com/image.jpg&task=OD"

# OCR
curl "http://localhost:7803/analyze?url=https://example.com/image.jpg&task=OCR"

# Open vocabulary detection — find specific classes
curl "http://localhost:7803/analyze?url=https://example.com/image.jpg&task=OPEN_VOCABULARY_DETECTION&text=cat+dog+person"

# Phrase grounding — locate parts of a caption
curl "http://localhost:7803/analyze?url=https://example.com/image.jpg&task=CAPTION_TO_PHRASE_GROUNDING&text=a+dog+on+a+couch"

# Referring expression segmentation — segment a described region
curl "http://localhost:7803/analyze?url=https://example.com/image.jpg&task=REFERRING_EXPRESSION_SEGMENTATION&text=the+red+car+on+the+left"

# File upload
curl -X POST -F "file=@/path/to/image.jpg" "http://localhost:7803/analyze?task=OD"

# Local file path
curl "http://localhost:7803/analyze?file=/path/to/image.jpg&task=DETAILED_CAPTION"
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7803` | Service port |
| `PRIVATE` | `False` | `True` = localhost only |
| `AUTO_UPDATE` | `True` | Download emoji/MWE configs from GitHub |
| `MODEL_NAME` | `microsoft/Florence-2-large` | Hugging Face model ID |
| `DEFAULT_TASK` | `DENSE_REGION_CAPTION` | Task used when no `task` param is given |
| `TIMEOUT` | `10.0` | GitHub download timeout (seconds) |

## Performance Notes

- First inference after startup is slower due to CUDA kernel compilation (typically 2-5s extra)
- Subsequent inferences: ~0.4-1.2s depending on task and image size on a mid-range GPU
- `Florence-2-base` (~230MB) is significantly faster but produces lower-quality outputs
- The model uses `float16` on GPU automatically; falls back to `float32` on CPU

## Docker

```bash
# Build
docker build -t florence2 /home/sd/animal-farm/florence2/

# Run (mounts HF cache to avoid re-downloading the ~1.5GB model on restart)
docker run -d \
  --name florence2 \
  --gpus all \
  --env-file /home/sd/animal-farm/florence2/.env \
  -v /home/sd/.cache/huggingface:/root/.cache/huggingface \
  -p 7803:7803 \
  florence2

# Test (model downloads on first run — allow a minute)
curl -s http://localhost:7803/health | python3 -m json.tool
curl -s "http://localhost:7803/analyze?url=https://example.com/image.jpg" | python3 -m json.tool

# Delete
docker stop florence2 && docker rm florence2 && docker rmi florence2
```
