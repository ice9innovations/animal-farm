# SpeciesNet Image Classification Service

**Framework**: Google SpeciesNet (Camera Trap AI)
**Purpose**: Wildlife species identification from camera trap images
**Status**: Active

## Overview

SpeciesNet runs a full detector + classifier + ensemble pipeline to identify wildlife species in images. It returns a primary prediction (species, genus, family, or broader category), top-5 classification scores, bounding box detections, and geofenced results when location data is provided.

## Features

- **Unified API**: Single endpoint for URL, local file path, or file upload
- **Geolocation support**: Optional country/region/GPS coordinates for geofenced predictions
- **Full pipeline**: Detector, classifier, and ensemble run together by default
- **CORS enabled**: Direct browser access supported

## Installation

### Prerequisites

- Python 3.10+
- SpeciesNet installed via pip (`pip install speciesnet`)
- Flask and flask-cors installed in the same environment

### 1. Environment Setup

```bash
cd /home/sd/speciesnet

# Activate virtual environment
source speciesnet_venv/bin/activate

# Install Flask dependencies (if not already done)
pip install flask flask-cors python-dotenv
```

### 2. Model Download

SpeciesNet downloads model weights automatically from Kaggle on first run. Weights are cached locally (~1.5GB). A Kaggle account and API token are required for the first download.

```bash
# Set up Kaggle credentials (one-time)
mkdir -p ~/.kaggle
# Place your kaggle.json token file at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Configuration

### Environment Variables (.env)

```bash
PORT=7778           # Service port
PRIVATE=False       # False = bind to 0.0.0.0, True = localhost only
MODEL=kaggle:google/speciesnet/pyTorch/v4.0.2a/1   # Optional, this is the default
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (False=public, True=localhost-only) |
| `MODEL` | No | `kaggle:google/speciesnet/pyTorch/v4.0.2a/1` | Model version to load |

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
  "model": "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"
}
```

### Analyze Image (Unified Endpoint)

```bash
GET  /analyze?url=<image_url>
GET  /analyze?file=<local_file_path>
POST /analyze  (multipart file upload)
```

#### Optional geo parameters (query string or form fields)

| Parameter | Type | Description |
|-----------|------|-------------|
| `country` | string | ISO 3166-1 alpha-3 code (e.g. `USA`, `AUS`, `KEN`) |
| `admin1_region` | string | ISO 3166-2 region (e.g. `CA` for California) |
| `latitude` | float | GPS latitude |
| `longitude` | float | GPS longitude |

**Examples:**

```bash
# Analyze from URL
curl "http://localhost:7778/analyze?url=https://example.com/camera_trap.jpg"

# Analyze from local file
curl "http://localhost:7778/analyze?file=/path/to/image.jpg"

# Analyze with geolocation (improves accuracy via geofencing)
curl "http://localhost:7778/analyze?file=/path/to/image.jpg&country=KEN&latitude=-1.2&longitude=36.8"

# Upload a file
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7778/analyze

# Upload with geo params
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  -F "country=USA" \
  -F "admin1_region=CA" \
  http://localhost:7778/analyze
```

**Successful Response:**
```json
{
  "service": "speciesnet",
  "status": "success",
  "predictions": [
    {
      "filepath": "/tmp/uploads/abc123.jpg",
      "classifications": {
        "classes": [
          "872e96f1-7a77-4ed2-824a-22fcbb32f598;mammalia;rodentia;caviidae;hydrochoerus;hydrochaeris;capybara",
          "6286f8b4-312f-4d04-a018-51a5d65be3f6;mammalia;rodentia;sciuridae;marmota;marmota;alpine marmot"
        ],
        "scores": [0.9662, 0.0069]
      },
      "detections": [
        {
          "category": "1",
          "label": "animal",
          "conf": 0.9191,
          "bbox": [0.16, 0.1585, 0.72, 0.6885]
        }
      ],
      "prediction": "872e96f1-7a77-4ed2-824a-22fcbb32f598;mammalia;rodentia;caviidae;hydrochoerus;hydrochaeris;capybara",
      "prediction_score": 0.9662,
      "prediction_source": "classifier",
      "model_version": "4.0.2a"
    }
  ],
  "metadata": {
    "processing_time": 1.243,
    "model_info": {
      "model": "kaggle:google/speciesnet/pyTorch/v4.0.2a/1",
      "framework": "SpeciesNet (Google Camera Trap AI)"
    }
  }
}
```

**Error Response:**
```json
{
  "service": "speciesnet",
  "status": "error",
  "predictions": [],
  "error": {"message": "File not found: /path/to/missing.jpg"},
  "metadata": {"processing_time": 0.001}
}
```

### Understanding the Prediction Field

The `prediction` field is a semicolon-delimited taxonomy string:

```
<uuid>;<class>;<order>;<family>;<genus>;<species>;<common_name>
```

Examples:
- `...;mammalia;rodentia;caviidae;hydrochoerus;hydrochaeris;capybara` → species-level ID
- `...;mammalia;carnivora;;;;<common>` → rolled up to order level
- `...;;;;;;blank` → empty frame, no animal detected
- `...;;;;;;animal` → animal detected but not identified to species

The `prediction_source` field indicates how the result was derived:
- `classifier` - classifier confidence was high enough
- `detector` - detection confidence drove the result
- `classifier+rollup_to_*` - classifier result rolled up the taxonomy tree due to low confidence
- `geofence_*` - geofencing filtered or modified the result

### Bounding Box Format

Bounding boxes use `[xmin, ymin, width, height]` as fractions of image dimensions (0.0–1.0).

## Service Management

### Manual Startup

```bash
cd /home/sd/speciesnet
source speciesnet_venv/bin/activate
python REST.py
```

### Systemd Service

```bash
sudo cp services/speciesnet-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start speciesnet-api
sudo systemctl enable speciesnet-api
sudo journalctl -u speciesnet-api -f
```

## Supported Image Formats

- **Input**: PNG, JPG/JPEG, TIFF/TIF, WebP
- **Max size**: 20MB
- **Input methods**: URL download, local file path, multipart upload

## Integration Examples

### Python

```python
import requests

# From URL
resp = requests.get(
    "http://localhost:7778/analyze",
    params={"url": "https://example.com/camera_trap.jpg", "country": "KEN"}
)

# From local file path
resp = requests.get(
    "http://localhost:7778/analyze",
    params={"file": "/data/images/trap001.jpg"}
)

# File upload
with open("/data/images/trap001.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:7778/analyze",
        files={"file": f},
        data={"country": "USA", "admin1_region": "CA"}
    )

result = resp.json()
if result["status"] == "success":
    pred = result["predictions"][0]
    print(f"Species: {pred['prediction'].split(';')[-1]}")
    print(f"Score:   {pred['prediction_score']:.3f}")
    print(f"Source:  {pred['prediction_source']}")
```

### JavaScript

```javascript
// From URL
const resp = await fetch(
  'http://localhost:7778/analyze?url=https://example.com/camera_trap.jpg'
);

// File upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('country', 'USA');
const resp = await fetch('http://localhost:7778/analyze', {
  method: 'POST',
  body: formData
});

const result = await resp.json();
if (result.status === 'success') {
  const pred = result.predictions[0];
  const commonName = pred.prediction.split(';').pop();
  console.log('Identified as:', commonName, 'with score', pred.prediction_score);
}
```

## Error Reference

| Error | Cause | Solution |
|-------|-------|----------|
| `File not found` | Bad local path | Check the file path |
| `File type not allowed` | Unsupported extension | Use JPG, PNG, TIFF, or WebP |
| `File too large` | File > 20MB | Resize or compress the image |
| `Failed to download image` | Bad URL or network error | Verify the URL is accessible |
| `Model not loaded` | Startup failure | Check logs for Kaggle auth errors |

---

**Documentation Version**: 1.0
**Last Updated**: 2026-02-18
**Upstream**: https://github.com/google/cameratrapai
