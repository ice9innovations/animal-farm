# MediaPipe Pose Estimation Service

**Port**: 7786  
**Framework**: Google MediaPipe Pose  
**Purpose**: Human pose landmark detection and joint angle analysis  
**Status**: ✅ Active

## Overview

MediaPipe Pose provides precise human pose estimation using Google's MediaPipe framework. The service analyzes images to detect 33 body landmarks with 3D coordinates and calculates joint angles for pose reconstruction. Optimized for reliability over speculation - returns only accurate MediaPipe data.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **33 Body Landmarks**: Full-body pose detection with 3D coordinates and visibility scores
- **Joint Angle Analysis**: Elbow and knee angle calculations for pose reconstruction
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **MediaPipe Precision**: Reliable landmark data without speculative classifications
- **Security**: File validation, size limits, secure cleanup
- **Performance**: Optimized processing with configurable model complexity

## Installation

### Prerequisites

- Python 3.8+
- OpenCV 4.12.0+
- 4GB+ RAM (8GB+ recommended)

### 1. Environment Setup

```bash
# Navigate to pose directory
cd /home/sd/animal-farm/pose

# Create virtual environment
python3 -m venv pose_venv

# Activate virtual environment
source pose_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Service Configuration

Copy and configure environment variables:

```bash
# Copy environment template
cp .env.sample .env

# Edit configuration as needed
nano .env
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the pose directory:

```bash
# Service Configuration
PORT=7786                                   # Service port (default: 7786)
PRIVATE=false                              # Access mode (false=public, true=localhost-only)

# API Configuration (Required for emoji mapping)
API_HOST=localhost                         # Host for emoji API
API_PORT=8080                             # Port for emoji API
API_TIMEOUT=2.0                           # Timeout for emoji API requests

# MediaPipe Pose Settings
POSE_MIN_DETECTION_CONFIDENCE=0.5         # Minimum detection confidence (0.0-1.0)
POSE_MIN_TRACKING_CONFIDENCE=0.5          # Minimum tracking confidence (0.0-1.0)
POSE_MODEL_COMPLEXITY=2                   # Model complexity (0=lite, 1=full, 2=heavy)
ENABLE_SEGMENTATION=true                  # Enable person segmentation masks

# File Processing
MAX_FILE_SIZE=8388608                     # Maximum file size in bytes (8MB)
UPLOAD_FOLDER=uploads                     # Temporary upload directory
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (false=public, true=localhost-only) |
| `API_HOST` | Yes | - | Host for emoji mapping API |
| `API_PORT` | Yes | - | Port for emoji mapping API |
| `API_TIMEOUT` | Yes | - | Timeout for emoji API requests |
| `POSE_MODEL_COMPLEXITY` | No | 2 | Model accuracy (0=fastest, 2=most accurate) |

### Model Complexity Options

| Complexity | Speed | Accuracy | Memory | Description |
|------------|-------|----------|---------|-------------|
| 0 (Lite) | Fastest | Good | Lowest | Quick pose detection |
| 1 (Full) | Fast | Better | Moderate | Balanced performance |
| 2 (Heavy) | Slower | Best | Highest | Production accuracy |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "pose",
  "capabilities": ["pose_estimation", "joint_analysis"],
  "models": {
    "pose_estimation": {
      "status": "ready",
      "model": "MediaPipe Pose",
      "landmarks": 33,
      "complexity": 2
    }
  }
}
```

### Analyze Pose (Unified Endpoint)

The unified `/v3/analyze` endpoint accepts either URL or file path input:

#### Analyze Pose from URL
```bash
GET /v3/analyze?url=<image_url>
```

**Example:**
```bash
curl "http://localhost:7786/v3/analyze?url=https://example.com/image.jpg"
```

#### Analyze Pose from File Path
```bash
GET /v3/analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7786/v3/analyze?file=/path/to/image.jpg"
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
      "framework": "MediaPipe Pose"
    },
    "processing_time": 0.101
  },
  "predictions": [
    {
      "properties": {
        "landmarks": {
          "nose": {
            "visibility": 1.0,
            "x": 0.323,
            "y": 0.442,
            "z": -0.538
          },
          "left_shoulder": {
            "visibility": 1.0,
            "x": 0.433,
            "y": 0.353,
            "z": -0.466
          },
          "left_elbow": {
            "visibility": 0.999,
            "x": 0.56,
            "y": 0.299,
            "z": -0.481
          },
          "left_wrist": {
            "visibility": 0.999,
            "x": 0.691,
            "y": 0.276,
            "z": -0.482
          }
        },
        "pose_analysis": {
          "joint_angles": {
            "left_elbow": 166.9,
            "left_knee": 68.3,
            "right_elbow": 159.1,
            "right_knee": 115.5
          }
        }
      }
    }
  ],
  "service": "pose",
  "status": "success"
}
```

## Service Management

### Manual Startup

```bash
# Start service
cd /home/sd/animal-farm/pose
./pose.sh
```

### Service Script

The `pose.sh` script handles virtual environment activation and service startup:

```bash
#!/bin/bash
cd "$(dirname "$0")"
source pose_venv/bin/activate
python3 REST.py
```

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: 8MB
- **Input Methods**: URL, file upload, local path

### Output Features
- **33 Body Landmarks**: Complete pose skeleton with 3D coordinates
- **Visibility Scores**: Confidence for each landmark (0.0-1.0)
- **Joint Angles**: Elbow and knee angles in degrees for pose reconstruction
- **Processing Metadata**: Timing and model information

## Landmark Reference

### MediaPipe Pose Landmarks (33 points)

**Face & Head**: nose, left/right eye (inner/outer), left/right ear, mouth (left/right)  
**Upper Body**: left/right shoulder, left/right elbow, left/right wrist  
**Hands**: left/right pinky, left/right index, left/right thumb  
**Lower Body**: left/right hip, left/right knee, left/right ankle  
**Feet**: left/right heel, left/right foot_index

### Joint Angles Calculated

- **left_elbow**: Angle between left shoulder → left elbow → left wrist
- **right_elbow**: Angle between right shoulder → right elbow → right wrist
- **left_knee**: Angle between left hip → left knee → left ankle
- **right_knee**: Angle between right hip → right knee → right ankle

## Performance Optimization

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|--------|
| CPU | Quad-core | 8+ cores | MediaPipe is CPU-optimized |
| RAM | 4GB | 8GB+ | Model complexity dependent |
| Storage | 1GB | 2GB+ | Dependencies + temp files |

### Optimization Settings

```bash
# Fast processing (lower accuracy)
POSE_MODEL_COMPLEXITY=0

# Balanced performance
POSE_MODEL_COMPLEXITY=1

# Best accuracy (production)
POSE_MODEL_COMPLEXITY=2

# Disable segmentation for speed
ENABLE_SEGMENTATION=false
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "MediaPipe initialization failed" | Missing dependencies | Install OpenCV and MediaPipe |
| "File too large" | File > 8MB | Resize or compress image |
| "Invalid URL" | Malformed URL | Check URL format |
| "No pose detected" | Poor image quality | Use clearer images with visible person |
| "File not found" | Invalid file path | Verify file exists and is readable |

### Error Response Format

```json
{
  "service": "pose",
  "status": "error",
  "predictions": [],
  "error": {"message": "File not found: /path/to/image.jpg"},
  "metadata": {
    "processing_time": 0.001,
    "model_info": {"framework": "MediaPipe Pose"}
  }
}
```

## Integration Examples

### Python Integration

```python
import requests

# URL input
response = requests.get(
    "http://localhost:7786/v3/analyze",
    params={"url": "https://example.com/image.jpg"}
)

# File input
response = requests.get(
    "http://localhost:7786/v3/analyze",
    params={"file": "/path/to/image.jpg"}
)

result = response.json()
if result["status"] == "success":
    prediction = result["predictions"][0]["properties"]
    landmarks = prediction["landmarks"]
    joint_angles = prediction["pose_analysis"]["joint_angles"]
    
    print(f"Nose position: {landmarks['nose']}")
    print(f"Left elbow angle: {joint_angles.get('left_elbow', 'N/A')}°")
```

### JavaScript Integration

```javascript
// URL input
const response = await fetch(
  'http://localhost:7786/v3/analyze?url=https://example.com/image.jpg'
);

// File input
const response = await fetch(
  'http://localhost:7786/v3/analyze?file=/path/to/image.jpg'
);

const result = await response.json();
if (result.status === 'success') {
  const properties = result.predictions[0].properties;
  const landmarks = properties.landmarks;
  const jointAngles = properties.pose_analysis.joint_angles;
  
  console.log('Landmarks detected:', Object.keys(landmarks).length);
  console.log('Joint angles:', jointAngles);
}
```

## Troubleshooting

### Installation Issues

**Problem**: MediaPipe import errors
```bash
# Solution: Ensure proper installation
pip install --upgrade mediapipe opencv-python
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
```

**Problem**: OpenCV errors
```bash
# Check OpenCV installation
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

### Runtime Issues

**Problem**: Service fails to start
```bash
# Check port availability
netstat -tlnp | grep 7786

# Check environment configuration
grep -E "^[^#]" .env
```

**Problem**: Poor pose detection
- Use `POSE_MODEL_COMPLEXITY=2` for best accuracy
- Ensure good lighting and clear person visibility
- Check `POSE_MIN_DETECTION_CONFIDENCE` setting
- Verify image quality and person is fully visible

### Performance Issues

**Problem**: Slow processing
- Use `POSE_MODEL_COMPLEXITY=0` for faster processing
- Set `ENABLE_SEGMENTATION=false` if segmentation not needed
- Process smaller images when possible
- Use file paths instead of URLs to avoid download overhead

**Problem**: Memory usage
- Reduce `POSE_MODEL_COMPLEXITY` for lower memory usage
- Monitor system memory with multiple concurrent requests
- Ensure adequate swap space for peak loads

## Security Considerations

### Access Control
- Set `PRIVATE=true` for localhost-only access
- Use reverse proxy with authentication for public access
- Validate all input URLs and file paths

### File Security
- Automatic cleanup of temporary files
- File type validation prevents executable uploads
- Size limits prevent DoS attacks
- Path traversal protection

---

**Documentation Version**: 2.0  
**Last Updated**: 2025-08-18  
**Service Version**: Production  
**Maintainer**: Animal Farm ML Team