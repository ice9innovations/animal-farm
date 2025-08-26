# RT-DETR Object Detection Service

**Port**: 7780  
**Framework**: RT-DETR (Real-Time Detection Transformer)  
**Purpose**: Real-time object detection with transformer architecture and emoji mapping  
**Status**: ✅ Active

## Overview

RT-DETR provides state-of-the-art real-time object detection using the RT-DETR (Real-Time Detection Transformer) model. The service detects 80 COCO object classes in images with confidence scores, bounding boxes, and automatic emoji mapping for enhanced user experience.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **COCO Object Detection**: Detects 80 common object classes
- **Advanced IoU Filtering**: Removes overlapping detections for cleaner results
- **Emoji Integration**: Automatic word-to-emoji mapping using local dictionary
- **GPU Acceleration**: CUDA support for fast inference with fallback to CPU
- **Confidence Filtering**: Configurable detection thresholds
- **Security**: File validation, size limits, secure cleanup

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for GPU)
- 15GB+ disk space for models

### 1. Environment Setup

```bash
# Navigate to RT-DETR directory
cd /home/sd/animal-farm/rtdetr

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Installation

```bash
# RT-DETR model will be loaded automatically from rtdetr_pytorch/
# The default model rtdetr_r50vd_6x_coco_from_paddle.pth should be present
# If not, download it from the RT-DETR repository

# Verify model file exists
ls rtdetr_pytorch/rtdetr_r50vd_6x_coco_from_paddle.pth
```

### 3. Hardware Configuration

```bash
# Verify CUDA availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi  # If CUDA GPU available
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the rtdetr directory:

```bash
# Service Configuration
PORT=7780                           # Service port
PRIVATE=False                       # Access mode (False=public, True=localhost-only)

# API Configuration (Required for emoji mapping)
API_HOST=localhost                  # Host for emoji API
API_PORT=8080                      # Port for emoji API
API_TIMEOUT=2.0                    # Timeout for emoji API requests
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (False=public, True=localhost-only) |
| `API_HOST` | Yes | - | Host for emoji mapping API |
| `API_PORT` | Yes | - | Port for emoji mapping API |
| `API_TIMEOUT` | Yes | - | Timeout for emoji API requests |

## API Endpoints

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "RT-DETR R50",
  "device": "cuda:0",
  "model_loaded": true,
  "emoji_service": "local_file"
}
```

### V3 Unified Analysis (Recommended)
```bash
GET /analyze?url=<image_url>
GET /analyze?file=<file_path>
```

**Parameters:**
- `url` (string): Image URL to analyze
- `file` (string): Local file path to analyze

**Note:** Exactly one parameter (`url` or `file`) must be provided.

**Response:**
```json
{
  "service": "rtdetr",
  "status": "success",
  "predictions": [
    {
      "label": "person",
      "confidence": 0.971,
      "bbox": {
        "x": 247,
        "y": 6,
        "width": 391,
        "height": 446
      },
      "emoji": "🧑"
    },
    {
      "label": "teddy bear",
      "confidence": 0.942,
      "bbox": {
        "x": 260,
        "y": 158,
        "width": 115,
        "height": 119
      },
      "emoji": "🧸"
    },
    {
      "label": "person",
      "confidence": 0.892,
      "bbox": {
        "x": 13,
        "y": 24,
        "width": 211,
        "height": 428
      },
      "emoji": "🧑"
    }
  ],
  "metadata": {
    "processing_time": 0.127,
    "model_info": {
      "framework": "PyTorch"
    }
  }
}
```

### V2 Compatibility Routes
```bash
GET /v2/analyze?image_url=<url>       # Translates to V3 url parameter
GET /v2/analyze_file?file_path=<path>  # Translates to V3 file parameter
```

## Service Management

### Manual Startup
```bash
# Ensure emoji API is running (required dependency)
# Start RT-DETR service

# Activate virtual environment
source venv/bin/activate

# Start service
python REST.py
```

### Systemd Service
```bash
# Install service file
sudo cp services/rtdetr-api.service /etc/systemd/system/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable rtdetr-api.service
sudo systemctl start rtdetr-api.service

# Check status
sudo systemctl status rtdetr-api.service
```

## Performance Optimization

### Hardware Requirements

**Minimum:**
- 8GB RAM
- 4-core CPU
- 15GB disk space

**Recommended:**
- 16GB+ RAM
- 8-core CPU
- RTX 3080 or better GPU
- NVMe SSD storage

### Model Performance

| Configuration | Speed | mAP | RAM Usage | Use Case |
|--------------|-------|-----|-----------|----------|
| CPU Mode | Medium | 53.1 | 4GB | Development/Testing |
| GPU Mode | Fast | 53.1 | 8GB | Production |

### Optimization Settings

- **Confidence Threshold**: 0.25 (filters low-confidence detections)
- **IoU Threshold**: 0.3 (removes overlapping detections)
- **GPU Acceleration**: Automatically detected and enabled
- **Batch Processing**: Single image optimized for API responses

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Model not loaded` | RT-DETR model failed to load | Check model files and GPU drivers |
| `File not found` | Invalid file path | Verify file exists and path is correct |
| `File too large` | Image exceeds 8MB limit | Resize image or increase MAX_FILE_SIZE |
| `Invalid URL format` | Malformed image URL | Check URL syntax and accessibility |
| `URL does not point to an image` | Non-image content type | Ensure URL serves image content |

### Error Response Format
```json
{
  "service": "rtdetr",
  "status": "error",
  "predictions": [],
  "error": {"message": "Detailed error description"},
  "metadata": {"processing_time": 0.001}
}
```

## Integration Examples

### Python
```python
import requests

# Object detection from URL
response = requests.get('http://localhost:7780/analyze', 
                       params={'url': 'https://example.com/image.jpg'})
data = response.json()

# Object detection from file
response = requests.get('http://localhost:7780/analyze',
                       params={'file': '/path/to/image.jpg'})
result = response.json()

# Process results
for prediction in result['predictions']:
    label = prediction['label']
    confidence = prediction['confidence']
    bbox = prediction['bbox']
    emoji = prediction.get('emoji', '')
    print(f"{emoji} {label}: {confidence:.2f} at ({bbox['x']}, {bbox['y']})")
```

### JavaScript
```javascript
// Object detection from URL
const response = await fetch('http://localhost:7780/analyze?' + 
    new URLSearchParams({url: 'https://example.com/image.jpg'}));
const data = await response.json();

// Process detections
data.predictions.forEach(prediction => {
    console.log(`${prediction.emoji || ''} ${prediction.label}: ${prediction.confidence}`);
    console.log(`Location: (${prediction.bbox.x}, ${prediction.bbox.y})`);
    console.log(`Size: ${prediction.bbox.width}x${prediction.bbox.height}`);
});
```

## Troubleshooting

### Installation Issues
- **CUDA not detected**: Install NVIDIA drivers and CUDA toolkit
- **Model loading fails**: Ensure model file is present and accessible
- **Dependencies missing**: Use requirements.txt in virtual environment

### Runtime Issues
- **Slow inference**: Ensure GPU is detected, check GPU memory
- **Memory errors**: Use CPU mode or increase system RAM
- **Connection refused**: Verify service is running on correct port

### Performance Issues
- **Low detection accuracy**: Check confidence threshold settings
- **Too many false positives**: Increase confidence threshold
- **Missing objects**: Decrease confidence threshold or check IoU settings

## Security Considerations

### Access Control
- Set `PRIVATE=True` for localhost-only access
- Use reverse proxy (nginx) for production deployment
- Implement rate limiting for public endpoints

### File Security
- File uploads validated and cleaned up automatically
- Size limits prevent resource exhaustion
- Only allowed image formats accepted

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: 8MB
- **Input Methods**: URL, file upload, local path

### Output Features
- **Object Detection**: 80 COCO object classes with bounding boxes
- **Confidence Scores**: Detection confidence values (0.0-1.0)
- **Emoji Mapping**: Automatic object-to-emoji conversion
- **IoU Filtering**: Advanced overlapping detection removal

---

**Documentation Version**: 1.0  
**Last Updated**: 2025-08-13  
**Service Version**: Production  
**Maintainer**: Animal Farm ML Team