# OCR Text Extraction Service

**Port**: 7775  
**Framework**: PaddleOCR  
**Purpose**: AI-powered text extraction with emoji mapping  
**Status**: ✅ Active

## Overview

The OCR service provides advanced text extraction from images using PaddleOCR framework. The service analyzes images to extract text content with high accuracy, automatically mapping meaningful words to relevant emojis for enhanced user experience.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **Emoji Integration**: Automatic word-to-emoji mapping using local dictionary
- **Multi-language Support**: 80+ languages supported by PaddleOCR
- **High Accuracy**: Advanced OCR with angle classification and GPU acceleration
- **Text Regions**: Detailed bounding box information for detected text
- **Security**: File validation, size limits, secure cleanup
- **Performance**: GPU acceleration with optimized text processing

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.2+ (for GPU acceleration)
- 4GB+ RAM (8GB+ recommended)
- PaddlePaddle framework

### 1. Environment Setup

```bash
# Navigate to OCR directory
cd /home/sd/animal-farm/ocr

# Create virtual environment
python3 -m venv ocr_venv

# Activate virtual environment
source ocr_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. PaddleOCR Setup

PaddleOCR models are downloaded automatically on first use:

```bash
# First run will download models (English, Chinese, Multi-language)
# Models cached locally for subsequent runs
# GPU support automatically detected and enabled
```

### 3. NLTK Data Setup

Required NLTK data is downloaded automatically on startup:

```bash
# Downloads punkt tokenizer and stopwords corpus
# Used for meaningful word extraction and emoji mapping
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the ocr directory:

```bash
# Service Configuration
PORT=7775                   # Service port
PRIVATE=false              # Access mode (false=public, true=localhost-only)

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

### OCR Engine Configuration

PaddleOCR is configured for optimal performance:

| Component | Configuration | Purpose |
|-----------|---------------|---------| 
| Text Detection | High accuracy model | Locates text regions in images |
| Text Recognition | Multi-language model | Extracts text content from regions |
| Angle Classification | Enabled | Corrects text orientation |
| GPU Acceleration | Auto-detected | Maximizes processing speed |
| Language Support | 80+ languages | Handles global text content |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "PaddleOCR",
  "ocr_engine": {
    "available": true,
    "version": "PaddleOCR 2.x",
    "languages": ["English", "Chinese", "80+ others"],
    "gpu_enabled": true
  },
  "features": {
    "text_angle_classification": true,
    "multilingual_support": true,
    "gpu_acceleration": true,
    "high_accuracy": true
  },
  "endpoints": [
    "GET /health - Health check",
    "GET /analyze?url=<image_url> - Analyze image from URL",
    "GET /analyze?file=<file_path> - Analyze image from file",
    "GET /v2/analyze?image_url=<url> - V2 compatibility (deprecated)",
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
curl "http://localhost:7775/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7775/analyze?file=/path/to/image.jpg"
```

**Input Validation:**
- Exactly one parameter must be provided (`url` OR `file`)
- Cannot provide both parameters simultaneously
- Returns error if neither parameter is provided

**Response Format:**
```json
{
  "service": "ocr",
  "status": "success",
  "predictions": [
    {
      "emoji": "💬",
      "emoji_mappings": [
        {
          "emoji": "🥏",
          "word": "frisbee"
        },
        {
          "emoji": "🗺️",
          "word": "world"
        }
      ],
      "has_text": true,
      "text": "FRISBEE FREESTYLE WORLD CHAMPIONSHIP San Lazzaro di Savena-Bologna 31July-3August2008 www . frisbeefreestyle . org SEESENL",
      "text_regions": [
        {
          "bbox": {
            "height": 16,
            "width": 310,
            "x": 8,
            "y": 13
          },
          "confidence": 0.965,
          "text": "FRISBEE FREESTYLE WORLD CHAMPIONSHIP"
        },
        {
          "bbox": {
            "height": 13,
            "width": 170,
            "x": 79,
            "y": 29
          },
          "confidence": 0.964,
          "text": "San Lazzaro di Savena-Bologna"
        },
        {
          "bbox": {
            "height": 12,
            "width": 123,
            "x": 102,
            "y": 41
          },
          "confidence": 0.976,
          "text": "31July-3August2008"
        },
        {
          "bbox": {
            "height": 14,
            "width": 134,
            "x": 97,
            "y": 57
          },
          "confidence": 0.991,
          "text": "www.frisbeefreestyle.org"
        },
        {
          "bbox": {
            "height": 7,
            "width": 26,
            "x": 290,
            "y": 53
          },
          "confidence": 0.746,
          "text": "SEESENL"
        }
      ]
    }
  ],
  "metadata": {
    "model_info": {
      "framework": "PaddleOCR"
    },
    "processing_time": 0.052
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
cd /home/sd/animal-farm/ocr
python REST.py
```

### Systemd Service

```bash
# Start service
sudo systemctl start ocr-api

# Enable auto-start
sudo systemctl enable ocr-api

# Check status
sudo systemctl status ocr-api

# View logs
journalctl -u ocr-api -f
```

## Performance Optimization

### Hardware Requirements

| Configuration | RAM | GPU | Response Time |
|---------------|-----|-----|---------------|
| Minimum | 4GB | None (CPU) | 0.5-1.0s |
| Recommended | 8GB | 4GB VRAM | 0.1-0.3s |
| High Volume | 16GB+ | 8GB+ VRAM | 0.05-0.2s |

### Performance Tuning

- **GPU Acceleration**: Automatic CUDA optimization when available
- **Model Caching**: Models loaded once and cached for performance
- **Text Processing**: NLTK-based meaningful word extraction
- **Emoji Mapping**: Fast local dictionary lookup
- **File Size Limit**: 8MB maximum (configurable)
- **Concurrent Requests**: Flask threaded mode enabled

### Performance Comparison

| Metric | CPU Only | GPU Enabled |
|--------|----------|-------------|
| Text Detection | ~0.3s | ~0.1s |
| Text Recognition | ~0.2s | ~0.05s |
| Total Processing | ~0.5s | ~0.15s |
| Memory Usage | 2-4GB | 1-2GB |
| Concurrent Capacity | 2-4 requests | 8-12 requests |

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Must provide either 'url' or 'file' parameter` | Missing input parameter | Provide exactly one parameter |
| `Cannot provide both 'url' and 'file' parameters` | Both parameters provided | Use only one parameter |
| `File not found: <path>` | Invalid file path | Check file exists and path is correct |
| `Image too large` | File > 8MB | Use smaller image or compress |
| `Failed to download image` | Network/URL issue | Verify URL is accessible |
| `OCR processing failed` | Processing error | Check image format and clarity |

### Error Response Format

```json
{
  "service": "ocr",
  "status": "error",
  "predictions": [],
  "error": {"message": "Error description"},
  "metadata": {
    "processing_time": 0.003,
    "model_info": {
      "framework": "PaddleOCR"
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
    'http://localhost:7775/analyze',
    params={'url': 'https://example.com/image.jpg'}
)
result = response.json()

# Process OCR results
if result['status'] == 'success':
    for prediction in result['predictions']:
        text = prediction['text']
        has_text = prediction['has_text']
        
        print(f"Text: {text}")
        print(f"Has text: {has_text}")
        
        # Process text regions with bounding boxes
        for region in prediction['text_regions']:
            bbox = region['bbox']
            region_text = region['text']
            region_confidence = region['confidence']
            print(f"Region: '{region_text}' at ({bbox['x']}, {bbox['y']}) size ({bbox['width']}x{bbox['height']})")
        
        # Process emoji mappings
        for mapping in prediction['emoji_mappings']:
            word = mapping['word']
            emoji = mapping['emoji']
            print(f"{word} → {emoji}")
    
    print(f"Processing time: {result['metadata']['processing_time']}s")
```

### JavaScript Integration

```javascript
// Analyze image from URL
async function analyzeOCR(imageUrl) {
    const response = await fetch(`http://localhost:7775/analyze?url=${encodeURIComponent(imageUrl)}`);
    const result = await response.json();
    
    if (result.status === 'success') {
        result.predictions.forEach(prediction => {
            console.log(`Text: ${prediction.text}`);
            console.log(`Has text: ${prediction.has_text}`);
            
            // Process text regions
            prediction.text_regions.forEach(region => {
                console.log(`Region: "${region.text}" at (${region.bbox.x}, ${region.bbox.y})`);
            });
            
            // Process emoji mappings
            prediction.emoji_mappings.forEach(mapping => {
                console.log(`${mapping.word} → ${mapping.emoji}`);
            });
        });
        
        console.log(`Processing time: ${result.metadata.processing_time}s`);
    }
}

// Usage
analyzeOCR('https://example.com/image.jpg');
```

### cURL Examples

```bash
# Basic text extraction
curl "http://localhost:7775/analyze?url=https://example.com/image.jpg"

# File analysis
curl "http://localhost:7775/analyze?file=/path/to/image.jpg"

# Health check
curl "http://localhost:7775/health"

# V2 compatibility (deprecated)
curl "http://localhost:7775/v2/analyze?image_url=https://example.com/image.jpg"
curl "http://localhost:7775/v2/analyze_file?file_path=/path/to/image.jpg"
```

## Troubleshooting

### Installation Issues

**Problem**: PaddleOCR installation fails
```bash
# Solution - install with specific CUDA version
pip uninstall paddlepaddle
pip install paddlepaddle-gpu  # For GPU support
# or
pip install paddlepaddle      # For CPU only
```

**Problem**: NLTK data download fails
```bash
# Solution - manual NLTK data download
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Runtime Issues

**Problem**: Port already in use
```bash
# Check what's using the port
lsof -i :7775

# Change port in .env file
echo "PORT=7776" >> .env
```

**Problem**: GPU out of memory
```bash
# Solution - monitor GPU usage
nvidia-smi

# Reduce concurrent requests or use CPU mode
```

### Performance Issues

**Problem**: Slow OCR processing on CPU
- Enable GPU acceleration by installing CUDA and paddlepaddle-gpu
- Ensure sufficient RAM available (8GB+ recommended)
- Process smaller images (< 2048px recommended)

**Problem**: High memory usage
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
- Verify API_HOST and API_PORT point to valid emoji API
- Check API_TIMEOUT is sufficient (increase if network is slow)
- Ensure emoji API service is running and accessible

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
- **Privacy**: No external API calls except emoji mapping service
- **Text Processing**: NLTK-based text analysis with stopword filtering

---

**Generated**: August 13, 2025  
**Framework Version**: PaddleOCR 2.x + NLTK 3.x  
**Service Version**: 3.0 (Modernized)
