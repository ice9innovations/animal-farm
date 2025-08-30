# Face Detection Service

**Port**: 7772  
**Framework**: MediaPipe  
**Purpose**: AI-powered face detection and facial analysis with fairness optimization  
**Status**: ✅ Active

## Overview

The Face service provides state-of-the-art face detection using Google's MediaPipe framework. The service analyzes images to detect human faces and extract facial keypoints with fairness optimizations across demographics.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **MediaPipe Framework**: Google's production-ready computer vision models
- **Fairness Optimization**: Tested across demographics to reduce bias
- **Focused Analysis**: Dedicated face detection with facial keypoints
- **High Performance**: Optimized model initialization and processing pipeline
- **Security**: File validation, size limits, secure cleanup

## Installation

### Prerequisites

- Python 3.8+
- OpenCV for image processing
- MediaPipe framework for computer vision
- 1GB+ RAM for model loading

### 1. Environment Setup

```bash
# Navigate to face directory
cd /home/sd/animal-farm/face

# Create virtual environment
python3 -m venv face_venv

# Activate virtual environment
source face_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dependency Installation

Install the required Python packages:

```bash
# Core dependencies
pip install flask flask-cors pillow opencv-python mediapipe python-dotenv requests numpy
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the face directory:

```bash
# Service Configuration
PORT=7772                    # Service port (default: 7772)
PRIVATE=false               # Access mode (false=public, true=localhost-only)

# API Configuration (Required for emoji mapping)
API_HOST=localhost          # Host for emoji API
API_PORT=8080              # Port for emoji API  
API_TIMEOUT=5              # Timeout for emoji API requests
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (false=public, true=localhost-only) |
| `API_HOST` | Yes | - | Host for emoji mapping API |
| `API_PORT` | Yes | - | Port for emoji mapping API |
| `API_TIMEOUT` | Yes | - | Timeout for emoji API requests |

### Model Configuration

The service uses MediaPipe models with optimized settings:

| Component | Configuration | Purpose |
|-----------|---------------|---------|
| Face Detection | Full-range model, confidence 0.2 | Fairness across demographics |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "face",
  "capabilities": ["face_detection"],
  "models": {
    "face_detection": {
      "status": "ready",
      "version": "0.10.14",
      "model": "MediaPipe Face Detection (Full Range)",
      "fairness": "Tested across demographics"
    }
  },
  "endpoints": [
    "GET /health - Health check",
    "GET /analyze?url=<image_url> - Analyze image from URL", 
    "GET /analyze?file=<file_path> - Analyze image from file",
    "POST /analyze - Analyze uploaded image file",
    "GET /v2/analyze?image_url=<image_url> - V2 compatibility (deprecated)",
    "GET /v2/analyze_file?file_path=<file_path> - V2 compatibility (deprecated)"
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
curl "http://localhost:7772/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7772/analyze?file=/path/to/image.jpg"
```

#### POST Request (File Upload)
```bash
POST /analyze
Content-Type: multipart/form-data
```

**Example:**
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7772/analyze
```

**Input Validation:**
- Exactly one parameter must be provided (`url` OR `file`)
- Cannot provide both parameters simultaneously
- Returns error if neither parameter is provided

**Response Format:**
```json
{
  "service": "face",
  "status": "success",
  "predictions": [
    {
      "label": "face",
      "emoji": "🙂",
      "confidence": 0.7625,
      "bbox": [385, 69, 79, 79],
      "keypoints": {
        "right_eye": [418, 95],
        "left_eye": [431, 93],
        "nose_tip": [409, 114],
        "mouth_center": [421, 129],
        "right_ear_tragion": [440, 100],
        "left_ear_tragion": [484, 93]
      }
    }
  ],
  "metadata": {
    "processing_time": 0.032,
    "model_info": {
      "framework": "MediaPipe"
    }
  }
}
```

### Legacy V2 Endpoints (Deprecated)

For backward compatibility, V2 endpoints are still supported but deprecated:

#### V2 URL Analysis
```bash
GET /v2/analyze?image_url=<image_url>
```

#### V2 File Analysis
```bash
GET /v2/analyze_file?file_path=<file_path>
```

## Service Management

### Manual Startup

```bash
# Start service
cd /home/sd/animal-farm/face
python3 REST.py
```

### Service Script

```bash
# Using startup script (if available)
./face.sh
```

### Systemd Service

```bash
# Start service
sudo systemctl start face-api

# Enable auto-start
sudo systemctl enable face-api

# Check status
sudo systemctl status face-api

# View logs
journalctl -u face-api -f
```

## Performance Optimization

### Hardware Requirements

| Configuration | RAM | CPU | Response Time |
|---------------|-----|-----|---------------|
| Minimum | 1GB | 2 cores | 0.2-0.4s |
| Recommended | 2GB+ | 4+ cores | 0.1-0.2s |
| High Volume | 4GB+ | 8+ cores | 0.05-0.15s |

### Performance Tuning

- **Model Loading**: Models initialized once at startup for optimal performance
- **File Size Limit**: 8MB maximum (configurable)
- **Concurrent Requests**: Flask threaded mode enabled
- **Memory Usage**: ~200MB base + image size during processing
- **GPU Support**: CPU-optimized, no GPU requirements

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Must provide either 'url' or 'file' parameter` | Missing input parameter | Provide exactly one parameter |
| `Cannot provide both 'url' and 'file' parameters` | Both parameters provided | Use only one parameter |
| `File not found: <path>` | Invalid file path | Check file exists and path is correct |
| `File type not allowed` | Unsupported image format | Use supported formats (PNG, JPG, JPEG, GIF, BMP, WEBP) |
| `Failed to download image` | Network/URL issue | Verify URL is accessible |
| `Image too large` | File > 8MB | Use smaller image or compress |

### Error Response Format

```json
{
  "service": "face",
  "status": "error",
  "predictions": [],
  "error": {"message": "Error description"},
  "metadata": {
    "processing_time": 0.003,
    "model_info": {
      "framework": "MediaPipe"
    }
  }
}
```

## Integration Examples

### Python Integration

```python
import requests

# Analyze image from URL
response = requests.get(
    'http://localhost:7772/analyze',
    params={'url': 'https://example.com/image.jpg'}
)

# POST file upload
with open('/path/to/image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:7772/analyze',
        files={'file': f}
    )
result = response.json()

# Process face detections
for prediction in result['predictions']:
    if prediction['label'] == 'face':
        bbox = prediction['bbox']
        confidence = prediction['confidence']
        print(f"Face detected: bbox={bbox}, confidence={confidence:.3f}")
        
        # Access facial keypoints
        keypoints = prediction['keypoints']
        print(f"Eyes: {keypoints['left_eye']}, {keypoints['right_eye']}")
```

### JavaScript Integration

```javascript
// Analyze image from URL
async function analyzeFaces(imageUrl) {
    const response = await fetch(`http://localhost:7772/analyze?url=${encodeURIComponent(imageUrl)}`);
    const result = await response.json();
    
    if (result.status === 'success') {
        result.predictions.forEach(prediction => {
            if (prediction.label === 'face') {
                console.log(`Face: confidence=${prediction.confidence}`);
                console.log(`Bounding box: [${prediction.bbox.join(', ')}]`);
            }
        });
        
        console.log(`Processing time: ${result.metadata.processing_time}s`);
    }
}

// Usage
analyzeFaces('https://example.com/image.jpg');
```

### cURL Examples

```bash
# Face detection analysis
curl "http://localhost:7772/analyze?url=https://example.com/image.jpg"

# File analysis
curl "http://localhost:7772/analyze?file=/path/to/image.jpg"

# POST file upload
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7772/analyze

# Health check
curl "http://localhost:7772/health"

# V2 compatibility (deprecated)
curl "http://localhost:7772/v2/analyze?image_url=https://example.com/image.jpg"
```

## Troubleshooting

### Installation Issues

**Problem**: MediaPipe installation fails
```bash
# Solution - install system dependencies
sudo apt-get update
sudo apt-get install python3-opencv
pip install mediapipe==0.10.14
```

**Problem**: OpenCV import errors
```bash
# Solution - reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python-headless
```

### Runtime Issues

**Problem**: Port already in use
```bash
# Check what's using the port
lsof -i :7772

# Change port in .env file
echo "PORT=7773" >> .env
```

**Problem**: Models fail to initialize
```bash
# Check MediaPipe version
python -c "import mediapipe as mp; print(mp.__version__)"

# Reinstall if needed
pip install --upgrade mediapipe
```

### Performance Issues

**Problem**: Slow face detection
- Reduce image size before analysis (< 1024px recommended)
- Ensure sufficient RAM available (2GB+ recommended)
- Check CPU load during analysis

**Problem**: Memory usage too high
- Restart service periodically for long-running processes  
- Monitor memory usage during batch processing
- Consider processing images in smaller batches

### Configuration Issues

**Problem**: Environment variable errors
```bash
# Check .env file exists and has correct format
cat .env

# Verify all required variables are set
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
required = ['PORT', 'PRIVATE', 'API_HOST', 'API_PORT', 'API_TIMEOUT']
missing = [k for k in required if not os.getenv(k)]
if missing: print(f'Missing: {missing}')
else: print('All variables set')
"
```

**Problem**: Emoji mappings not loading
- Check API_HOST and API_PORT point to running emoji API
- Verify network connectivity between services
- Check API_TIMEOUT setting (increase if needed)

## Security Considerations

### Access Control

- **Private Mode**: Set `PRIVATE=true` for localhost-only access
- **File Path Access**: Restricted in private mode for security
- **Input Validation**: All inputs validated before processing

### File Security

- **Size Limits**: 8MB maximum file size
- **Format Validation**: Only supported image formats accepted  
- **Temporary Files**: Automatically cleaned up after processing
- **Path Validation**: File paths validated to prevent directory traversal

### Network Security

- **Timeout Protection**: Download timeouts prevent hanging connections
- **CORS Configuration**: Configured for controlled browser access
- **Error Information**: Error messages don't expose system internals

### Model Security

- **Bias Reduction**: MediaPipe models tested for fairness across demographics
- **No Data Retention**: Images processed in memory, not stored
- **Privacy**: No external API calls except for emoji mapping

---

**Generated**: August 13, 2025  
**Framework Version**: MediaPipe 0.10.14  
**Service Version**: 3.0 (Modernized)