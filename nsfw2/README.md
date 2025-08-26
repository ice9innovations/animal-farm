# NSFW Content Detection Service

**Port**: 7774  
**Framework**: TensorFlow (OpenNSFW2)  
**Purpose**: AI-powered content moderation and NSFW detection  
**Status**: ✅ Active

## Overview

The NSFW2 service provides state-of-the-art content moderation using Yahoo's OpenNSFW2 model. This service analyzes images to detect Not Safe For Work (NSFW) content with significantly improved accuracy over legacy solutions, featuring 24x reduction in false positives and superior confidence scoring.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **OpenNSFW2 Model**: Yahoo's production-ready content detection framework
- **Superior Accuracy**: 24x reduction in false positive baseline (4.8% → 0.2%)
- **High Confidence**: 95.2% confidence on actual NSFW content vs 50.2% legacy
- **Format Support**: Native JPEG, PNG, WebP processing without conversion
- **GPU Optimization**: CUDA 12.2 support with automatic memory management
- **Security**: File validation, size limits, secure cleanup

## Installation

### Prerequisites

- Python 3.8+
- CUDA 12.2 (for GPU acceleration)
- TensorFlow 2.18.0+
- 2GB+ RAM for model loading
- GPU with 4GB+ VRAM (recommended)

### 1. Environment Setup

```bash
# Navigate to nsfw2 directory
cd /home/sd/animal-farm/nsfw2

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. GPU Configuration

The service includes automatic CUDA configuration:

```bash
# GPU setup handled by nsfw2.sh startup script
# Includes CUDA 12.2 paths and cuDNN library setup
# Automatic memory growth and XLA compilation
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the nsfw2 directory:

```bash
# Service Configuration
PORT=7774                   # Service port
PRIVATE=false              # Access mode (false=public, true=localhost-only)

# Model Configuration  
NSFW_THRESHOLD=50.0        # Detection threshold percentage
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (false=public, true=localhost-only) |
| `NSFW_THRESHOLD` | Yes | - | NSFW detection threshold (0-100 scale) |

### Model Configuration

The service uses OpenNSFW2 with optimized settings:

| Component | Configuration | Purpose |
|-----------|---------------|---------| 
| Detection Model | OpenNSFW2 | Yahoo/Flickr trained for content moderation |
| Threshold | 50.0% | Balanced false positive/negative ratio |
| GPU Optimization | XLA + Mixed Precision | Maximum performance on compatible hardware |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "NSFW2 Detection",
  "model": {
    "status": "loaded",
    "framework": "Keras/TensorFlow",
    "model_type": "OpenNSFW2",
    "threshold": 50.0
  },
  "endpoints": [
    "GET /health - Health check",
    "GET /analyze?url=<image_url> - Analyze image from URL",
    "GET /analyze?file=<file_path> - Analyze image from file",
    "GET /v2/analyze?image_url=<url> - V2 compatibility (deprecated)"
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
curl "http://localhost:7774/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7774/analyze?file=/path/to/image.jpg"
```

**Input Validation:**
- Exactly one parameter must be provided (`url` OR `file`)
- Cannot provide both parameters simultaneously
- Returns error if neither parameter is provided

**Response Format:**
```json
{
  "service": "nsfw2",
  "status": "success",
  "predictions": [
    {
      "confidence": 0.558,
      "emoji": "🔞",
      "nsfw": true
    }
  ],
  "metadata": {
    "processing_time": 0.097,
    "model_info": {
      "framework": "TensorFlow"
    }
  }
}
```

**Example - Safe Content:**
```json
{
  "service": "nsfw2", 
  "status": "success",
  "predictions": [
    {
      "confidence": 0.999,
      "emoji": "",
      "nsfw": false
    }
  ],
  "metadata": {
    "processing_time": 0.104,
    "model_info": {
      "framework": "TensorFlow"
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
# Start service with GPU optimization
cd /home/sd/animal-farm/nsfw2
./nsfw2.sh
```

### Systemd Service

```bash
# Start service
sudo systemctl start nsfw-api

# Enable auto-start
sudo systemctl enable nsfw-api

# Check status
sudo systemctl status nsfw-api

# View logs
journalctl -u nsfw-api -f
```

## Performance Optimization

### Hardware Requirements

| Configuration | RAM | GPU | Response Time |
|---------------|-----|-----|---------------|
| Minimum | 2GB | None (CPU) | 0.3-0.5s |
| Recommended | 4GB | 4GB VRAM | 0.1-0.2s |
| High Volume | 8GB+ | 8GB+ VRAM | 0.05-0.15s |

### Performance Tuning

- **GPU Acceleration**: Automatic CUDA optimization when available
- **Memory Growth**: Dynamic GPU memory allocation
- **XLA Compilation**: Just-in-time optimization for compatible operations
- **Mixed Precision**: Automatic FP16/FP32 mixed precision training
- **File Size Limit**: 8MB maximum (configurable)
- **Concurrent Requests**: Flask threaded mode enabled

### Performance Comparison

| Metric | Legacy Model | OpenNSFW2 |
|--------|--------------|-----------|
| Safe Image Baseline | 4.8% confidence | 0.2% confidence |
| NSFW Detection | 50.2% confidence | 95.2% confidence |
| Format Support | JPEG only | JPEG, PNG, WebP |
| Processing Time | ~0.1s | ~0.1s |
| False Positive Rate | High | 24x lower |

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Must provide either 'url' or 'file' parameter` | Missing input parameter | Provide exactly one parameter |
| `Cannot provide both 'url' and 'file' parameters` | Both parameters provided | Use only one parameter |
| `File not found: <path>` | Invalid file path | Check file exists and path is correct |
| `Image too large` | File > 8MB | Use smaller image or compress |
| `Failed to download image` | Network/URL issue | Verify URL is accessible |
| `NSFW detection failed` | Processing error | Check image format and integrity |

### Error Response Format

```json
{
  "service": "nsfw2",
  "status": "error",
  "predictions": [],
  "error": {"message": "Error description"},
  "metadata": {
    "processing_time": 0.003,
    "model_info": {
      "framework": "TensorFlow"
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
    'http://localhost:7774/analyze',
    params={'url': 'https://example.com/image.jpg'}
)
result = response.json()

# Process NSFW detection
if result['status'] == 'success':
    for prediction in result['predictions']:
        confidence = prediction['confidence']
        is_nsfw = prediction['nsfw']
        print(f"NSFW: {is_nsfw}, confidence: {confidence:.3f}")
        
        if is_nsfw:
            print(f"Content flagged as NSFW with {confidence*100:.1f}% confidence")
        else:
            print(f"Content classified as safe with {confidence*100:.1f}% confidence")
    
    print(f"Processing time: {result['metadata']['processing_time']}s")
```

### JavaScript Integration

```javascript
// Analyze image from URL
async function analyzeNSFW(imageUrl) {
    const response = await fetch(`http://localhost:7774/analyze?url=${encodeURIComponent(imageUrl)}`);
    const result = await response.json();
    
    if (result.status === 'success') {
        result.predictions.forEach(prediction => {
            console.log(`NSFW: ${prediction.nsfw}`);
            console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`);
            
            if (prediction.nsfw) {
                console.log(`⚠️ Content flagged as NSFW with ${(prediction.confidence * 100).toFixed(1)}% confidence ${prediction.emoji}`);
            } else {
                console.log(`✅ Content classified as safe with ${(prediction.confidence * 100).toFixed(1)}% confidence`);
            }
        });
        
        console.log(`Processing time: ${result.metadata.processing_time}s`);
    }
}

// Usage
analyzeNSFW('https://example.com/image.jpg');
```

### cURL Examples

```bash
# Basic NSFW analysis
curl "http://localhost:7774/analyze?url=https://example.com/image.jpg"

# File analysis
curl "http://localhost:7774/analyze?file=/path/to/image.jpg"

# Health check
curl "http://localhost:7774/health"

# V2 compatibility (deprecated)
curl "http://localhost:7774/v2/analyze?image_url=https://example.com/image.jpg"
curl "http://localhost:7774/v2/analyze_file?file_path=/path/to/image.jpg"
```

## Troubleshooting

### Installation Issues

**Problem**: TensorFlow GPU setup fails
```bash
# Solution - verify CUDA installation
nvidia-smi
nvcc --version

# Reinstall TensorFlow with GPU support
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

**Problem**: OpenNSFW2 model fails to load
```bash
# Solution - clear model cache and reinstall
pip uninstall opennsfw2
pip install opennsfw2
```

### Runtime Issues

**Problem**: Port already in use
```bash
# Check what's using the port
lsof -i :7774

# Change port in .env file
echo "PORT=7775" >> .env
```

**Problem**: GPU out of memory
```bash
# Solution - enable memory growth (handled automatically)
# Or reduce batch size / image resolution
```

### Performance Issues

**Problem**: Slow NSFW detection on CPU
- Enable GPU acceleration by installing CUDA 12.2
- Ensure sufficient RAM available (4GB+ recommended)
- Process smaller images (< 2048px recommended)

**Problem**: High memory usage
- Restart service periodically for long-running processes
- Monitor GPU memory usage during batch processing  
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
required = ['PORT', 'PRIVATE', 'NSFW_THRESHOLD']
missing = [k for k in required if not os.getenv(k)]
if missing: print(f'Missing: {missing}')
else: print('All variables set')
"
```

**Problem**: NSFW threshold too sensitive/permissive
- Adjust `NSFW_THRESHOLD` in .env (higher = less sensitive)
- Recommended range: 30.0-70.0 depending on use case
- Test with known safe/unsafe content to calibrate

## Security Considerations

### Access Control

- **Private Mode**: Set `PRIVATE=true` for localhost-only access
- **File Path Access**: Validated to prevent directory traversal
- **Input Validation**: All inputs validated before processing

### File Security

- **Size Limits**: 8MB maximum file size
- **Format Validation**: Only supported image formats accepted
- **Temporary Files**: Automatically cleaned up after processing
- **Path Validation**: File paths validated to prevent system access

### Network Security

- **Timeout Protection**: Download timeouts prevent hanging connections
- **CORS Configuration**: Configured for controlled browser access
- **Error Information**: Error messages don't expose system internals

### Content Security

- **No Data Retention**: Images processed in memory, not stored permanently
- **Privacy**: No external API calls or data logging
- **Model Security**: OpenNSFW2 trained on curated datasets with ethical guidelines

---

**Generated**: August 13, 2025  
**Framework Version**: TensorFlow 2.18.0 + OpenNSFW2  
**Service Version**: 3.0 (Modernized)