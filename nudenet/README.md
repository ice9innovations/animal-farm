# NudeNet+ Content Moderation Service

**Port**: 7789
**Framework**: NudeNet+ (ONNX Runtime)
**Purpose**: Fine-grained content moderation with category-level detection
**Status**: ✅ Active

## Overview

NudeNet+ provides fine-grained content moderation using advanced body part detection. Unlike binary NSFW classifiers, this service detects specific categories of exposed/covered body parts with bounding box locations, enabling nuanced content moderation decisions.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for URL, file path, and file upload
- **Fine-Grained Detection**: 18 specific categories (not just binary NSFW/safe)
- **Bounding Boxes**: Precise location data for each detection
- **Emoji Integration**: Automatic category-to-emoji mapping
- **GPU Acceleration**: ONNX Runtime with CUDA support (~18ms inference)
- **Security**: File validation, size limits, automatic cleanup

## Detection Categories

NudeNet detects **18 distinct categories** organized by exposure level:

### Exposed Categories (Content Moderation)
- `FEMALE_GENITALIA_EXPOSED` 🔞
- `MALE_GENITALIA_EXPOSED` 🔞
- `FEMALE_BREAST_EXPOSED` 🔞
- `MALE_BREAST_EXPOSED` 🔞
- `BUTTOCKS_EXPOSED` 🔞
- `ANUS_EXPOSED` 🔞
- `BELLY_EXPOSED` 🔞
- `ARMPITS_EXPOSED` 🔞
- `FEET_EXPOSED` 🦶

### Covered Categories (Context Detection)
- `FEMALE_GENITALIA_COVERED` 👙
- `FEMALE_BREAST_COVERED` 👙
- `BUTTOCKS_COVERED` 👖
- `ANUS_COVERED` 👖
- `BELLY_COVERED` 👕
- `ARMPITS_COVERED` 👕
- `FEET_COVERED` 👟

### Face Detection
- `FACE_FEMALE` 👩
- `FACE_MALE` 👨

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.2+ (for GPU acceleration)
- 4GB+ RAM
- ONNX Runtime GPU support

### 1. Environment Setup

```bash
# Navigate to nudenet directory
cd /home/sd/animal-farm/nudenet

# Create virtual environment
python3 -m venv nudenet_venv

# Activate virtual environment
source nudenet_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Download

NudeNet automatically downloads required models on first run. No manual download needed.

The detector model (~60MB) will be downloaded to `~/.NudeNet/` on initialization.

## Configuration

### Environment Variables (.env)

Create a `.env` file in the nudenet directory:

```bash
# Service Configuration
PORT=7789                      # Service port (default: 7789)
PRIVATE=false                  # Access mode (false=public, true=localhost-only)

# Detection Configuration
DETECTION_THRESHOLD=60         # Minimum confidence threshold (0-100, default: 60)
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (false=public, true=localhost-only) |
| `DETECTION_THRESHOLD` | Yes | - | Minimum confidence for detections (0-100) |

### Threshold Guidelines

| Threshold | Use Case | Result |
|-----------|----------|--------|
| 40-50 | Maximum sensitivity | More detections, possible false positives |
| 60-70 | Balanced (recommended) | Good accuracy, minimal false positives |
| 80-90 | High precision | Only very confident detections |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "NudeNet Detection",
  "model": {
    "status": "loaded",
    "framework": "NudeNet+",
    "threshold": 60.0
  },
  "endpoints": [
    "GET /health - Health check",
    "GET,POST /analyze - Unified endpoint (URL/file/upload)",
    "GET /v3/analyze - V3 compatibility",
    "GET /v2/analyze - V2 compatibility (deprecated)",
    "GET /v2/analyze_file - V2 compatibility (deprecated)"
  ]
}
```

### Analyze Image (Unified Endpoint)

The unified `/analyze` endpoint accepts either URL or file path input:

#### Analyze Image from URL
```bash
GET /analyze?url=<image_url>
```

**Example:**
```bash
curl "http://localhost:7789/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7789/analyze?file=/path/to/image.jpg"
```

#### POST Request (File Upload)
```bash
POST /analyze
Content-Type: multipart/form-data
```

**Example:**
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7789/analyze
```

**Input Validation:**
- Exactly one parameter must be provided (`url` OR `file`)
- Cannot provide both parameters simultaneously
- Returns error if neither parameter is provided

**Response Format:**
```json
{
  "service": "nudenet",
  "status": "success",
  "predictions": [
    {
      "label": "FEMALE_BREAST_EXPOSED",
      "confidence": 0.953,
      "bbox": [245, 123, 456, 378],
      "emoji": "🔞"
    },
    {
      "label": "FACE_FEMALE",
      "confidence": 0.891,
      "bbox": [189, 45, 312, 198],
      "emoji": "👩"
    }
  ],
  "metadata": {
    "processing_time": 0.018,
    "model_info": {
      "framework": "NudeNet+"
    }
  }
}
```

### V2 Compatibility Endpoints

#### V2 URL Analysis
```bash
GET /v2/analyze?image_url=<url>
```

#### V2 File Analysis
```bash
GET /v2/analyze_file?file_path=<path>
```

#### V3 Compatibility
```bash
GET /v3/analyze?url=<url>
GET /v3/analyze?file=<path>
```

## Service Management

### Manual Startup

```bash
# Start service
cd /home/sd/animal-farm/nudenet
./nudenet.sh
```

### Systemd Service

```bash
# Install service (if available)
sudo cp services/nudenet-api.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start/stop service
sudo systemctl start nudenet-api
sudo systemctl stop nudenet-api

# Enable auto-start
sudo systemctl enable nudenet-api

# Check status
sudo systemctl status nudenet-api

# View logs
sudo journalctl -u nudenet-api -f
```

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: 8MB
- **Input Methods**: URL, file upload, local path

### Output Features
- **Category Detection**: 18 distinct body part categories
- **Bounding Boxes**: [x1, y1, x2, y2] format for each detection
- **Confidence Scores**: 0.0-1.0 range per detection
- **Emoji Mapping**: Visual category indicators
- **Filtering**: Threshold-based confidence filtering

## Performance

### Benchmarks

| Hardware | Processing Time | Notes |
|----------|----------------|-------|
| RTX 3080 | ~18ms | CUDA + TensorRT |
| RTX 2060 | ~35ms | CUDA only |
| GTX 1060 | ~75ms | CUDA only |
| CPU (8-core) | ~400ms | No GPU acceleration |

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|--------|
| GPU | GTX 1060 3GB | RTX 3060+ | CUDA 11.2+ required |
| RAM | 4GB | 8GB+ | Model + inference |
| Storage | 500MB | 1GB+ | Models + cache |

### Optimization

**GPU Acceleration:**
```bash
# Verify GPU is available
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

**CPU Fallback:**
If GPU is unavailable, service automatically falls back to CPU (slower inference).

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "NudeNet detector not initialized" | Model download failed | Check internet connection, retry |
| "File too large" | File > 8MB | Resize or compress image |
| "Invalid URL" | Malformed URL | Check URL format |
| "File not found" | Invalid path | Verify file exists |
| "Must provide either url or file parameter" | No input | Provide url or file parameter |
| "Cannot provide both url and file parameters" | Duplicate input | Use only one parameter |

### Error Response Format

```json
{
  "service": "nudenet",
  "status": "error",
  "predictions": [],
  "error": {"message": "File not found: /path/to/image.jpg"},
  "metadata": {"processing_time": 0.001}
}
```

## Integration Examples

### Python Integration

```python
import requests

# URL input
response = requests.get(
    "http://localhost:7789/analyze",
    params={"url": "https://example.com/image.jpg"}
)

# File input
response = requests.get(
    "http://localhost:7789/analyze",
    params={"file": "/path/to/image.jpg"}
)

# POST file upload
with open('/path/to/image.jpg', 'rb') as f:
    response = requests.post(
        "http://localhost:7789/analyze",
        files={'file': f}
    )

result = response.json()
if result["status"] == "success":
    for prediction in result["predictions"]:
        label = prediction["label"]
        confidence = prediction["confidence"]
        bbox = prediction["bbox"]
        emoji = prediction["emoji"]
        print(f"{emoji} {label}: {confidence:.2%} at {bbox}")
```

### JavaScript Integration

```javascript
// URL input
const response = await fetch(
  'http://localhost:7789/analyze?url=https://example.com/image.jpg'
);

// File input
const response = await fetch(
  'http://localhost:7789/analyze?file=/path/to/image.jpg'
);

// POST file upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
const response = await fetch('http://localhost:7789/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
if (result.status === 'success') {
  result.predictions.forEach(prediction => {
    console.log(`${prediction.emoji} ${prediction.label}:`,
                `${(prediction.confidence * 100).toFixed(1)}%`,
                `at [${prediction.bbox.join(', ')}]`);
  });
}
```

### Filtering Results

```python
# Filter for only explicit content (exposed categories)
explicit_categories = [
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED"
]

result = requests.get("http://localhost:7789/analyze?url=...").json()
explicit_detections = [
    p for p in result["predictions"]
    if p["label"] in explicit_categories
]

if explicit_detections:
    print("⚠️ Explicit content detected!")
    for det in explicit_detections:
        print(f"  - {det['label']}: {det['confidence']:.2%}")
```

## Troubleshooting

### Installation Issues

**Problem**: ONNX Runtime GPU not working
```bash
# Verify CUDA is available
nvidia-smi

# Check ONNX Runtime providers
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Reinstall GPU version
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu>=1.23.0
```

**Problem**: Model download fails
```bash
# Check internet connection
ping pypi.org

# Manually trigger model download
python -c "from nudenet import NudeDetector; NudeDetector()"

# Check model cache
ls -lh ~/.NudeNet/
```

### Runtime Issues

**Problem**: Service fails to start
```bash
# Check port availability
netstat -tlnp | grep 7789

# Check .env file exists
cat .env

# Test manual startup
source nudenet_venv/bin/activate
python REST.py
```

**Problem**: Slow inference times
```bash
# Verify GPU is being used
nvidia-smi  # Should show python process using GPU

# Check ONNX Runtime providers
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should show CUDAExecutionProvider first

# Lower detection threshold for faster processing
# Edit .env: DETECTION_THRESHOLD=70
```

### Performance Issues

**Problem**: High memory usage
- NudeNet creates temporary files during inference
- Ensure sufficient /tmp space available
- Consider lowering detection threshold to reduce output size

**Problem**: False positives
- Increase `DETECTION_THRESHOLD` in .env (try 70-80)
- Filter results by category (ignore covered/armpits/feet categories)
- Use confidence scores to filter low-quality detections

**Problem**: Missing detections
- Lower `DETECTION_THRESHOLD` in .env (try 40-50)
- Check image quality and resolution
- Verify body parts are visible in frame

## Security Considerations

### Access Control
- Set `PRIVATE=true` for localhost-only access
- Use reverse proxy with authentication for public access
- Validate all input URLs and file paths

### File Security
- Automatic cleanup of temporary files
- File type validation prevents executable uploads
- Size limits prevent DoS attacks
- Temporary files use secure random names

### Content Policy
- Define acceptable thresholds per use case
- Log detections for policy enforcement
- Consider legal requirements for content storage
- Implement rate limiting for public APIs

## Use Cases

### Content Moderation
```python
# Flag explicit content for review
explicit = ["FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"]
result = requests.get(f"http://localhost:7789/analyze?url={image_url}").json()
flagged = any(p["label"] in explicit for p in result["predictions"])
```

### Age-Appropriate Filtering
```python
# Block any exposed content
exposed_categories = [c for c in all_categories if "EXPOSED" in c]
has_exposed = any(p["label"] in exposed_categories for p in predictions)
```

### Context Analysis
```python
# Distinguish artistic nudity from explicit content
# Use covered categories + face detection for context
has_faces = any(p["label"].startswith("FACE_") for p in predictions)
has_covered = any("COVERED" in p["label"] for p in predictions)
likely_artistic = has_faces and has_covered
```

## Emoji Customization

Edit `/home/sd/animal-farm/nudenet/emoji_mappings.json` to customize category emojis:

```json
{
  "FEMALE_GENITALIA_EXPOSED": "🔞",
  "MALE_GENITALIA_EXPOSED": "🔞",
  "FACE_FEMALE": "👩",
  "FACE_MALE": "👨"
}
```

Changes take effect after service restart.

---

**Documentation Version**: 1.0
**Last Updated**: 2025-12-30
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
