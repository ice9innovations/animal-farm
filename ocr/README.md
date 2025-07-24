# PaddleOCR Service

Modern OCR service using PaddleOCR as a drop-in replacement for the legacy Tesseract service.

## Features

- **Fast GPU-accelerated OCR** using PaddleOCR
- **High accuracy** - significantly better than Tesseract
- **Multilingual support** - 80+ languages supported
- **Text angle classification** - handles rotated text
- **Semantic emoji extraction** - analyzes text content for relevant emojis
- **v2 API compliant** - follows unified response schema

## Performance

- **Speed**: ~100-300ms (vs Tesseract's 500ms+)
- **Accuracy**: Significantly improved for real-world images
- **VRAM usage**: ~1-1.5GB
- **CPU usage**: Minimal (GPU-accelerated)

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create environment file:
   ```bash
   cp .env.sample .env
   ```

3. Install systemd service:
   ```bash
   sudo cp services/ocr-api.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable ocr-api
   sudo systemctl start ocr-api
   ```

## Usage

### API Endpoints

- `GET /health` - Health check
- `POST /v2/analyze` - Extract text and emojis from image

### Example Request

```bash
curl -X POST http://localhost:7775/v2/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_url": "http://example.com/image.jpg"}'
```

### Example Response

```json
{
  "service": "ocr",
  "status": "success",
  "predictions": [
    {
      "type": "text_extraction",
      "text": "Hello World",
      "emoji": "💬",
      "confidence": 0.95,
      "properties": {
        "has_text": true,
        "raw_text": "Hello World",
        "cleaned_text": "Hello World",
        "engine": "PaddleOCR",
        "processing_method": "gpu_accelerated"
      }
    },
    {
      "type": "emoji_mappings",
      "confidence": 1.0,
      "properties": {
        "mappings": [
          {"word": "world", "emoji": "🌍"}
        ],
        "source": "ocr_content_analysis"
      }
    }
  ],
  "metadata": {
    "processing_time": 0.156,
    "model_info": {
      "name": "PaddleOCR",
      "framework": "PaddlePaddle",
      "version": "2.x"
    },
    "parameters": {
      "use_gpu": true,
      "language": "en",
      "angle_classification": true
    }
  }
}
```

## Deployment

This service runs as an independent OCR service:

1. **Setup service**: `./setup_service.sh`
2. **Start service**: 
   ```bash
   sudo systemctl start ocr-api
   sudo systemctl enable ocr-api
   ```
3. **Monitor service**:
   ```bash
   sudo systemctl status ocr-api
   sudo journalctl -u ocr-api -f
   ```

The service runs on port 7775 and follows the v2 unified API schema.

## Configuration

Edit `.env` file:

```env
PORT=7775
USE_GPU=True
LANGUAGE=en
USE_ANGLE_CLASSIFICATION=True
MAX_FILE_SIZE=8388608
```

## Monitoring

Check service status:
```bash
sudo systemctl status ocr-api
curl http://localhost:7775/health
```

## Troubleshooting

- **CUDA errors**: Ensure CUDA is properly installed
- **Memory errors**: Reduce batch size or use CPU mode
- **Slow performance**: Verify GPU is being used
- **Import errors**: Check PaddlePaddle installation

## Dependencies

- PaddleOCR 2.7.0+
- PaddlePaddle-GPU 2.5.0+
- Flask 2.0.0+
- PIL, numpy, requests, nltk

## Why PaddleOCR?

- **3x faster** than legacy OCR solutions
- **Much better accuracy** on real-world images
- **GPU acceleration** for optimal performance
- **Modern architecture** with active development
- **Multilingual support** (80+ languages)
- **Semantic content analysis** with emoji extraction