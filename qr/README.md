# QR Code & Barcode Scanner Service

**Port**: 7801
**Framework**: pyzbar (libzbar)
**Purpose**: QR code and barcode detection and decoding
**Status**: Active

## Overview

The QR service detects and decodes QR codes, EAN/UPC barcodes, and other common symbologies from images. It returns the raw decoded payload and bounding box for each code found. No interpretation of content is performed — the service reports exactly what is encoded.

## Features

- **Multi-symbology**: Decodes QR codes, EAN-13, EAN-8, UPC-A, UPC-E, Code 128, Code 39, PDF417, DataMatrix, and more
- **Unified Input Handling**: Single endpoint accepts URL, local file path, or multipart file upload
- **Bounding Boxes**: Pixel coordinates for each detected code
- **Multiple Codes**: All codes in an image are returned, not just the first
- **Fast**: CPU-only, no model loading — pyzbar wraps the native libzbar library
- **Security**: File size limits, input validation, no data retention

## Installation

### Prerequisites

- Python 3.8+
- libzbar0 system library
- 512MB+ RAM

### 1. Install System Library

```bash
sudo apt-get install libzbar0
```

### 2. Environment Setup

```bash
cd /home/sd/animal-farm/qr

python3 -m venv qr_venv
source qr_venv/bin/activate

pip install -r requirements.txt
```

## Configuration

### Environment Variables (.env)

```bash
PORT=7801
```

### Configuration Details

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Service listening port |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "QR Code Scanner",
  "library": "pyzbar",
  "endpoints": [
    "GET  /health",
    "POST /analyze"
  ]
}
```

### Analyze Image

#### URL input

```bash
GET /analyze?url=<image_url>
```

**Example:**
```bash
curl "http://localhost:7801/analyze?url=https://example.com/qrcode.png"
```

#### File path input

```bash
GET /analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7801/analyze?file=/path/to/qrcode.png"
```

#### Multipart file upload

```bash
POST /analyze
Content-Type: multipart/form-data
```

**Example:**
```bash
curl -X POST -F "file=@/path/to/qrcode.png" http://localhost:7801/analyze
```

**Input Validation:**
- Exactly one input method must be used (`url`, `file`, or multipart upload)
- Cannot provide both `url` and `file` simultaneously
- Maximum file size: 8MB

**Response Format:**
```json
{
  "service": "qr",
  "status": "success",
  "predictions": [
    {
      "has_codes": true,
      "code_count": 2,
      "codes": [
        {
          "type": "QRCODE",
          "data": "https://example.com",
          "bbox": {
            "x": 10,
            "y": 10,
            "width": 120,
            "height": 120
          }
        },
        {
          "type": "EAN13",
          "data": "5901234123457",
          "bbox": {
            "x": 200,
            "y": 50,
            "width": 95,
            "height": 60
          }
        }
      ]
    }
  ],
  "metadata": {
    "processing_time": 0.012,
    "model_info": {
      "library": "pyzbar"
    }
  }
}
```

**Response when no codes are found:**
```json
{
  "service": "qr",
  "status": "success",
  "predictions": [
    {
      "has_codes": false,
      "code_count": 0,
      "codes": []
    }
  ],
  "metadata": {
    "processing_time": 0.008,
    "model_info": {
      "library": "pyzbar"
    }
  }
}
```

## Supported Symbologies

| Type | Description |
|------|-------------|
| `QRCODE` | QR Code |
| `EAN13` | EAN-13 barcode |
| `EAN8` | EAN-8 barcode |
| `UPCA` | UPC-A barcode |
| `UPCE` | UPC-E barcode |
| `CODE128` | Code 128 barcode |
| `CODE39` | Code 39 barcode |
| `CODE93` | Code 93 barcode |
| `PDF417` | PDF417 stacked barcode |
| `DATAMATRIX` | Data Matrix 2D barcode |
| `CODABAR` | Codabar barcode |
| `ITF` | Interleaved 2-of-5 |

## Service Management

### Manual Startup

```bash
cd /home/sd/animal-farm/qr
./qr.sh
```

### Systemd Service

```bash
# Install service
sudo cp services/qr-api.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start/stop service
sudo systemctl start qr-api
sudo systemctl stop qr-api

# Enable auto-start
sudo systemctl enable qr-api

# Check status
systemctl status qr-api

# View logs
journalctl -u qr-api -f
```

## Performance

pyzbar wraps the native libzbar C library. Processing is CPU-only and typically completes in under 20ms for standard images. No model loading overhead — the service is ready immediately on startup.

| Image Size | Typical Processing Time |
|------------|------------------------|
| Small (<500px) | 5-10ms |
| Medium (500-2000px) | 10-20ms |
| Large (>2000px) | 20-50ms |

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Cannot specify both url and file` | Both parameters provided | Use only one input method |
| `Provide a file upload, url parameter, or file parameter` | No input provided | Include one input method |
| `File not found: <path>` | Invalid file path | Verify the path exists |
| `File too large (max 8MB)` | File exceeds size limit | Use a smaller image |
| `Failed to download image` | Network or URL error | Verify URL is accessible |

### Error Response Format

```json
{
  "service": "qr",
  "status": "error",
  "predictions": [],
  "error": {"message": "File not found: /path/to/image.png"},
  "metadata": {"processing_time": 0.001}
}
```

## Integration Examples

### Python

```python
import requests

# URL input
response = requests.get(
    'http://localhost:7801/analyze',
    params={'url': 'https://example.com/qrcode.png'}
)
result = response.json()

if result['status'] == 'success':
    prediction = result['predictions'][0]
    print(f"Found {prediction['code_count']} code(s)")
    for code in prediction['codes']:
        print(f"{code['type']}: {code['data']}")
        bbox = code['bbox']
        print(f"  at ({bbox['x']}, {bbox['y']}) size {bbox['width']}x{bbox['height']}")
```

```python
# File upload
with open('/path/to/qrcode.png', 'rb') as f:
    response = requests.post(
        'http://localhost:7801/analyze',
        files={'file': f}
    )
result = response.json()
```

### JavaScript

```javascript
// URL input
const response = await fetch(
  `http://localhost:7801/analyze?url=${encodeURIComponent(imageUrl)}`
);
const result = await response.json();

if (result.status === 'success') {
  result.predictions[0].codes.forEach(code => {
    console.log(`${code.type}: ${code.data}`);
  });
}

// File upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
const response = await fetch('http://localhost:7801/analyze', {
  method: 'POST',
  body: formData
});
```

### cURL

```bash
# URL input
curl "http://localhost:7801/analyze?url=https://example.com/qrcode.png"

# File path input
curl "http://localhost:7801/analyze?file=/path/to/qrcode.png"

# File upload
curl -X POST -F "file=@/path/to/qrcode.png" http://localhost:7801/analyze

# Health check
curl http://localhost:7801/health
```

## Troubleshooting

**Problem**: `ImportError: Unable to find zbar shared library`
```bash
# Install the system library
sudo apt-get install libzbar0
```

**Problem**: Port already in use
```bash
# Check what is using the port
lsof -i :7801

# Change port in .env
echo "PORT=7802" > .env
```

**Problem**: No codes detected despite visible QR code in image
- Ensure the image is in focus and not heavily distorted
- Try a higher resolution version of the image
- Verify the code is not damaged or partially obscured

## Security Considerations

- **Access Control**: The service binds to `0.0.0.0`. Use a firewall or reverse proxy to restrict access if needed.
- **File Size Limit**: 8MB maximum prevents resource exhaustion
- **No Data Retention**: Images are processed in memory and not stored
- **Input Validation**: File paths and URLs are validated before use

## Docker

```bash
# Build
docker build -t qr /home/sd/animal-farm/qr/

# Run
docker run -d \
  --name qr \
  --env-file /home/sd/animal-farm/qr/.env \
  -p 7801:7801 \
  qr

# Test
curl -s http://localhost:7801/health | python3 -m json.tool
curl -s -X POST -F "file=@/home/sd/animal-farm/qr/qr.jpg" http://localhost:7801/analyze | python3 -m json.tool

# Delete
docker stop qr && docker rm qr && docker rmi qr
```

> No GPU required. `libzbar0` is installed in the image.

---

**Framework Version**: pyzbar 0.1.9+
**Service Version**: 1.0
