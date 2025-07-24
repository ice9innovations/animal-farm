# Face Detection Service

This service provides reliable face detection using Google's MediaPipe framework, designed for fairness across demographics and superior performance compared to traditional computer vision approaches.

## Features

- REST API for face detection and counting with v1 and v2 endpoints
- MediaPipe Face Detection optimized for fairness across demographics
- Bounding box detection with confidence scores
- 6-point facial landmark detection
- High performance with model initialized once at startup
- Comprehensive error handling and logging
- Health check endpoints

## Setup

### 1. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv face_venv

# Activate virtual environment
source face_venv/bin/activate

# Install dependencies
pip install flask flask-cors pillow opencv-python mediapipe python-dotenv requests
```

### 2. Configure Environment

Create a `.env` file with your configuration:

```bash
# Copy environment template
cp .env.sample .env

# Edit .env with your settings
```

**Environment variables:**
```bash
# Service Settings
PORT=7772
PRIVATE=false
```

### 3. Run Service

```bash
# Start the face detection service
python REST.py

# Or use the startup script
./face.sh
```

## API Usage

### REST API Endpoints

- `GET /health` - Health check and model status
- `GET /?url=<image_url>` - Detect faces in image from URL
- `GET /?path=<local_path>` - Detect faces in local image (if not in private mode)
- `POST /` - Upload and detect faces in image file
- `GET /v2/analyze?image_url=<url>` - V2 unified format for URL analysis
- `GET /v2/analyze_file?file_path=<path>` - V2 unified format for file analysis

### Example Usage

**Detect faces from URL:**
```bash
curl "http://localhost:7772/?url=https://example.com/image.jpg"
```

**V2 API:**
```bash
curl "http://localhost:7772/v2/analyze?image_url=https://example.com/image.jpg"
```

**Upload file:**
```bash
curl -X POST -F "uploadedfile=@image.jpg" http://localhost:7772/
```

### Response Format

**Legacy format:**
```json
{
  "FACE": {
    "faces": [
      {
        "bbox": {"x": 150, "y": 200, "width": 120, "height": 150},
        "confidence": 0.85,
        "method": "mediapipe"
      }
    ],
    "total_faces": 1,
    "image_dimensions": {"width": 640, "height": 480},
    "model_info": {
      "detection_method": "mediapipe",
      "detection_time": 0.156,
      "framework": "MediaPipe"
    },
    "status": "success"
  }
}
```

## Model Information

This service uses Google's MediaPipe Face Detection framework:

- **Framework**: MediaPipe v0.10.x (designed for fairness across demographics)
- **Model**: Full-range face detection optimized for diverse faces
- **Confidence Threshold**: 0.2 (lowered to reduce bias)
- **Performance**: ~150ms average processing time
- **Hardware**: CPU-optimized, no GPU requirements

## Configuration Options

The service supports the following configuration options in `.env`:

- **PORT**: Service port (default: 7772)
- **PRIVATE**: Enable private mode - restricts local file access (default: false)

**PRIVATE mode explanation:**
- `PRIVATE=false`: Service binds to all network interfaces (0.0.0.0) and allows local file path access via `/?path=` parameter
- `PRIVATE=true`: Service binds to localhost only (127.0.0.1) and blocks local file path access for security

## Health Check

Check service status and model information:

```bash
curl http://localhost:7772/health
```

## Troubleshooting

1. **MediaPipe installation issues**: Ensure you have the correct Python version (3.8+)
2. **No faces detected**: Check image quality and lighting conditions
3. **Service won't start**: Verify virtual environment activation and all dependencies installed
4. **Permission errors**: Check file permissions and PRIVATE mode setting