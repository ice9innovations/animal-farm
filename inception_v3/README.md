# Inception v3 Image Classification Service

**Port**: 7779  
**Framework**: Google Inception v3 with TensorFlow  
**Purpose**: ImageNet image classification with emoji mapping  
**Status**: ✅ Active

## Overview

Inception v3 provides state-of-the-art image classification using Google's Inception v3 model trained on ImageNet. The service analyzes images and identifies objects, scenes, and concepts from 1,000+ categories, automatically mapping classifications to relevant emojis for enhanced user experience.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **Emoji Integration**: Automatic classification-to-emoji mapping using local dictionary
- **High Accuracy**: 1,000+ ImageNet classes with confidence scoring
- **Security**: File validation, size limits, secure cleanup
- **Performance**: GPU acceleration with TensorFlow optimization

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- CUDA 11.2+ (for GPU acceleration)
- 4GB+ RAM (8GB+ recommended for optimal performance)

### 1. Environment Setup

```bash
# Navigate to Inception v3 directory
cd /home/sd/animal-farm/inception_v3

# Create virtual environment
python3 -m venv inception_venv

# Activate virtual environment
source inception_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Download

The Inception v3 model with ImageNet weights is automatically downloaded by TensorFlow on first run (approximately 92MB).

## Configuration

### Environment Variables (.env)

Create a `.env` file in the inception_v3 directory:

```bash
# Service Configuration
PORT=7779                    # Service port (default: 7779)
PRIVATE=False               # Access mode (False=public, True=localhost-only)

# API Configuration (Required for emoji mapping)
API_HOST=localhost          # Host for emoji API
API_PORT=8080              # Port for emoji API  
API_TIMEOUT=2.0            # Timeout for emoji API requests
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (False=public, True=localhost-only) |
| `API_HOST` | Yes | - | Host for emoji mapping API |
| `API_PORT` | Yes | - | Port for emoji mapping API |
| `API_TIMEOUT` | No | 2.0 | Timeout for emoji API requests |

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
  "confidence_threshold": 0.15,
  "framework": "TensorFlow",
  "model": "Inception v3"
}
```

### Analyze Image (Unified Endpoint)

The unified `/v3/analyze` endpoint accepts either URL or file path input:

#### Analyze Image from URL
```bash
GET /v3/analyze?url=<image_url>
```

**Example:**
```bash
curl "http://localhost:7779/v3/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /v3/analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7779/v3/analyze?file=/path/to/image.jpg"
```

**Input Validation:**
- Exactly one parameter must be provided (`url` OR `file`)
- Cannot provide both parameters simultaneously
- Returns error if neither parameter is provided

**Response Format:**
```json
{
  "metadata": {
    "model_info": {
      "framework": "TensorFlow"
    },
    "processing_time": 0.116
  },
  "predictions": [
    {
      "confidence": 0.815,
      "emoji": "👔",
      "label": "groom"
    },
    {
      "confidence": 0.743,
      "emoji": "👨",
      "label": "bridegroom"
    },
    {
      "confidence": 0.521,
      "emoji": "🤵",
      "label": "suit"
    }
  ],
  "service": "inception",
  "status": "success"
}
```

### V2 Compatibility Endpoints

Legacy V2 endpoints remain available for backward compatibility:

#### V2 URL Analysis
```bash
GET /v2/analyze?image_url=<image_url>
```

#### V2 File Analysis
```bash
GET /v2/analyze_file?file_path=<file_path>
```

Both V2 endpoints return the same format as V3 but with parameter translation.

## Service Management

### Manual Startup

```bash
# Start service
cd /home/sd/animal-farm/inception_v3
python REST.py
```

### Systemd Service

```bash
# Install service
sudo cp services/inception-api.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start/stop service
sudo systemctl start inception-api
sudo systemctl stop inception-api

# Enable auto-start
sudo systemctl enable inception-api

# Check status
sudo systemctl status inception-api

# View logs
sudo journalctl -u inception-api -f
```

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: 8MB
- **Input Methods**: URL, file upload, local path

### Classification Output
- **ImageNet Classes**: 1,000+ object and scene categories
- **Confidence Scores**: Probability values from 0.0 to 1.0
- **Emoji Mapping**: Visual representations for classifications
- **Threshold Filtering**: Only returns classifications above 0.15 confidence

## Performance Optimization

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|--------|
| GPU | GTX 1050 2GB | GTX 1660+ | CUDA 11.2+ required |
| RAM | 4GB | 8GB+ | Model + image processing |
| Storage | 1GB | 2GB+ | Model cache + temp files |

### Optimization Settings

```python
# Confidence threshold (in REST.py)
CONFIDENCE_THRESHOLD = 0.15  # Lower = more results, higher = fewer but more confident

# Image preprocessing
IMAGE_SIZE = 299  # Inception v3 input size (fixed)

# GPU memory growth
tf.config.experimental.set_memory_growth(gpu, True)  # Prevents full GPU allocation
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Model not loaded" | TensorFlow installation issue | Reinstall TensorFlow with GPU support |
| "File too large" | File > 8MB | Resize or compress image |
| "Invalid URL" | Malformed URL | Check URL format and accessibility |
| "File type not allowed" | Unsupported format | Use PNG, JPG, JPEG, GIF, BMP, or WebP |
| "File not found" | Invalid file path | Check file path and permissions |

### Error Response Format

```json
{
  "metadata": {
    "processing_time": 0.001
  },
  "predictions": [],
  "service": "inception",
  "status": "error",
  "error": {
    "message": "File not found: /path/to/image.jpg"
  }
}
```

## Integration Examples

### Python Integration

```python
import requests

# URL input
response = requests.get(
    "http://localhost:7779/v3/analyze",
    params={"url": "https://example.com/image.jpg"}
)

# File input
response = requests.get(
    "http://localhost:7779/v3/analyze",
    params={"file": "/path/to/image.jpg"}
)

result = response.json()
if result["status"] == "success":
    predictions = result["predictions"]
    for pred in predictions:
        print(f"{pred['emoji']} {pred['label']}: {pred['confidence']:.3f}")
```

### JavaScript Integration

```javascript
// URL input
const response = await fetch(
  'http://localhost:7779/v3/analyze?url=https://example.com/image.jpg'
);

// File input
const response = await fetch(
  'http://localhost:7779/v3/analyze?file=/path/to/image.jpg'
);

const result = await response.json();
if (result.status === 'success') {
  result.predictions.forEach(pred => {
    console.log(`${pred.emoji} ${pred.label}: ${pred.confidence.toFixed(3)}`);
  });
}
```

## Troubleshooting

### Installation Issues

**Problem**: TensorFlow GPU not working
```bash
# Check CUDA installation
nvidia-smi

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Reinstall TensorFlow with GPU support
pip install tensorflow[and-cuda]
```

**Problem**: Model download fails
```bash
# Check internet connection
curl -I https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5

# Clear TensorFlow cache
rm -rf ~/.keras/models/
```

### Runtime Issues

**Problem**: Service fails to start
```bash
# Check port availability
netstat -tlnp | grep 7779

# Check environment variables
cat .env

# Test dependencies
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Problem**: Emoji mapping failures
```bash
# Verify emoji API is running
curl http://localhost:8080/emoji_mappings.json

# Check API configuration in .env
grep API_ .env
```

### Performance Issues

**Problem**: Slow inference
- Enable GPU acceleration (ensure CUDA is properly installed)
- Reduce image size before processing
- Use confidence threshold to limit results

**Problem**: Memory errors
- Enable TensorFlow memory growth
- Process images sequentially rather than in batches
- Monitor system memory usage

## Security Considerations

### Access Control
- Set `PRIVATE=True` for localhost-only access
- Use reverse proxy with authentication for public access
- Validate all input URLs and file paths

### File Security
- Automatic cleanup of temporary files
- File type validation prevents executable uploads
- Size limits prevent DoS attacks
- Path traversal protection

### Data Privacy
- No classification data is stored permanently
- Temporary files are automatically cleaned
- Processing logs exclude sensitive information

---

**Documentation Version**: 1.0  
**Last Updated**: 2025-08-13  
**Service Version**: Production  
**Maintainer**: Animal Farm ML Team