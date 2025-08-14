# Metadata Extraction Service

**Port**: 7781  
**Framework**: ExifTool + PIL + OpenCV + NumPy  
**Purpose**: Comprehensive image metadata extraction with advanced analysis  
**Status**: ✅ Active

## Overview

The Metadata Extraction Service provides comprehensive image metadata analysis using multiple extraction engines. It combines traditional EXIF data extraction with advanced computer vision analysis including quality assessment, composition analysis, and accessibility features.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **Multiple Extraction Engines**: ExifTool, PIL/Pillow, and OpenCV integration
- **Advanced Analysis**: Image quality, color properties, composition, and accessibility
- **Comprehensive Metadata**: Camera settings, GPS data, technical specifications
- **Security**: File validation, size limits, secure cleanup
- **Performance**: Efficient processing with automated categorization

## Installation

### Prerequisites

- Python 3.8+
- ExifTool system package
- OpenCV and NumPy for advanced analysis
- 4GB+ RAM (8GB+ recommended for large images)

### 1. Environment Setup

```bash
# Navigate to metadata directory
cd /home/sd/animal-farm/metadata

# Create virtual environment
python3 -m venv metadata_venv

# Activate virtual environment
source metadata_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. System Dependencies

Install ExifTool system package:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install exiftool

# macOS
brew install exiftool

# CentOS/RHEL
sudo yum install perl-Image-ExifTool
```

### 3. Verify Installation

```bash
# Test ExifTool availability
exiftool -ver

# Test Python imports
python -c "import cv2, numpy, PIL; print('Dependencies OK')"
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the metadata directory:

```bash
# Service Configuration
PORT=7781                    # Service port (default: 7781)
PRIVATE=false               # Access mode (false=public, true=localhost-only)
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (false=public, true=localhost-only) |

### Advanced Analysis Features

The service provides comprehensive analysis across multiple categories:

| Category | Features | Description |
|----------|----------|-------------|
| **Quality Analysis** | Blur detection, lighting, exposure, contrast, aesthetic scoring | Technical image quality assessment |
| **Color Properties** | Saturation analysis, color statistics | Color composition analysis |
| **Composition** | Aspect ratio, rule of thirds, complexity, symmetry | Artistic composition evaluation |
| **GPS Data** | GPS coordinates, location metadata | Geographic information when available |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Metadata Extraction",
  "extraction_engines": {
    "exiftool": {
      "available": true,
      "status": "Available"
    },
    "pil_pillow": {
      "available": true,
      "status": "Available"
    }
  },
  "features": {
    "metadata_categories": ["camera", "image", "gps", "datetime", "software", "technical"],
    "extraction_methods": ["comprehensive_exif", "image_properties", "file_system_info"],
    "advanced_analysis": {
      "image_quality": ["blur_detection", "lighting_analysis", "exposure_analysis", "contrast_analysis", "aesthetic_scoring"],
      "color_properties": ["dominant_colors", "color_statistics", "color_temperature", "saturation_analysis"],
      "composition": ["aspect_ratio", "rule_of_thirds", "complexity_analysis", "symmetry_analysis"],
      "gps_data": ["coordinate_extraction", "location_metadata", "geographic_information"]
    }
  }
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
curl "http://localhost:7781/v3/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /v3/analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7781/v3/analyze?file=/path/to/image.jpg"
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
      "framework": "ExifTool + PIL + OpenCV + NumPy"
    },
    "processing_time": 0.122
  },
  "predictions": [
    {
      "color_properties": {
        "saturation_analysis": {
          "average_saturation": 81.1,
          "saturation_level": 81.1
        }
      },
      "composition": {
        "complexity_analysis": {
          "complexity_level": 19.3,
          "edge_density": 0.193
        },
        "rule_of_thirds": {
          "most_interesting_section": 7,
          "section_brightness": [93.2, 104.8, 92.5, 146.1, 167.5, 88.3, 106.1, 138.2, 54.5]
        },
        "symmetry_analysis": {
          "horizontal_symmetry_score": 0.732,
          "symmetry_level": 73.2
        }
      },
      "dimensions": {
        "height": 457,
        "width": 640
      },
      "aspect_ratio": {
        "category": "landscape",
        "ratio": 1.4
      },
      "file": {
        "file_size": 111494,
        "file_type": "jpeg"
      },
      "image_quality": {
        "blur_analysis": {
          "laplacian_variance": 2449.82,
          "sharpness_score": 1.0
        },
        "contrast_analysis": {
          "brightness_variation": 68.38,
          "contrast_quality": 26.7,
          "contrast_score": 1.07,
          "dynamic_range": 68.38
        },
        "exposure_analysis": {
          "bright_pixel_ratio": 0.09,
          "dark_pixel_ratio": 0.041,
          "exposure_quality": 86.9,
          "histogram_balance": 68.38
        },
        "lighting_analysis": {
          "lighting_quality": 110.01,
          "mean_brightness": 110.01
        }
      }
    }
  ],
  "service": "metadata",
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
cd /home/sd/animal-farm/metadata
python REST.py
```

### Systemd Service

```bash
# Install service
sudo cp services/metadata-api.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start/stop service
sudo systemctl start metadata-api
sudo systemctl stop metadata-api

# Enable auto-start
sudo systemctl enable metadata-api

# Check status
sudo systemctl status metadata-api

# View logs
sudo journalctl -u metadata-api -f
```

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP, TIFF, RAW
- **Max Size**: 8MB
- **Input Methods**: URL, file upload, local path

### Extracted Metadata Categories
- **Camera**: Make, model, lens, ISO, aperture, shutter speed, flash settings
- **Image**: Dimensions, color space, compression, orientation
- **GPS**: Latitude, longitude, altitude, location data
- **DateTime**: Creation, modification, capture timestamps
- **Software**: Processing applications, tool versions
- **Technical**: Encoding, profiles, color depth, sampling

### Advanced Analysis Output
- **Quality Metrics**: Blur detection, lighting assessment, exposure analysis
- **Color Properties**: Saturation levels, color statistics
- **Composition**: Aspect ratio, complexity, symmetry analysis
- **Accessibility**: Alt-text suggestions, content type detection

## Performance Optimization

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|--------|
| CPU | 2 cores | 4+ cores | For OpenCV processing |
| RAM | 4GB | 8GB+ | Depends on image size |
| Storage | 1GB | 2GB+ | Temp files + metadata cache |

### Processing Optimization

```python
# Adjust analysis complexity (in REST.py)
BLUR_THRESHOLD = 100      # Lower = more sensitive blur detection
MAX_COMPLEXITY = 0.02     # Higher = more detailed edge analysis
SYMMETRY_PRECISION = 0.1  # Lower = more precise symmetry analysis
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "ExifTool not available" | Missing system package | Install exiftool package |
| "File too large" | File > 8MB | Resize or compress image |
| "File not found" | Invalid file path | Check file path and permissions |
| "Invalid URL" | Malformed URL | Check URL format and accessibility |
| "Metadata extraction failed" | Corrupted image | Try different image format |

### Error Response Format

```json
{
  "service": "metadata",
  "status": "error",
  "predictions": [],
  "error": {"message": "File not found: /path/to/image.jpg"},
  "metadata": {"processing_time": 0.001}
}
```

## Integration Examples

### Python Integration

```python
import requests

# URL input
response = requests.get(
    "http://localhost:7781/v3/analyze",
    params={"url": "https://example.com/image.jpg"}
)

# File input
response = requests.get(
    "http://localhost:7781/v3/analyze",
    params={"file": "/path/to/image.jpg"}
)

result = response.json()
if result["status"] == "success":
    metadata = result["predictions"][0]["properties"]
    
    print(f"File size: {metadata['file_size']} bytes")
    print(f"Dimensions: {metadata['dimensions']['width']}x{metadata['dimensions']['height']}")
    print(f"Has EXIF: {metadata['has_exif']}")
    print(f"Has GPS: {metadata['has_gps']}")
    print(f"Aesthetic score: {metadata['quality_analysis']['aesthetic_score']}")
    print(f"Alt text: {metadata['accessibility']['alt_text_basic']}")
```

### JavaScript Integration

```javascript
// URL input
const response = await fetch(
  'http://localhost:7781/v3/analyze?url=https://example.com/image.jpg'
);

// File input
const response = await fetch(
  'http://localhost:7781/v3/analyze?file=/path/to/image.jpg'
);

const result = await response.json();
if (result.status === 'success') {
  const metadata = result.predictions[0].properties;
  
  console.log('File type:', metadata.file_type);
  console.log('Dimensions:', `${metadata.dimensions.width}x${metadata.dimensions.height}`);
  console.log('Quality score:', metadata.quality_analysis.aesthetic_score);
  console.log('Composition:', metadata.composition_analysis.aspect_ratio);
}
```

## Troubleshooting

### Installation Issues

**Problem**: ExifTool not available
```bash
# Solution: Install ExifTool
sudo apt-get install exiftool

# Verify installation
exiftool -ver
```

**Problem**: OpenCV import errors
```bash
# Solution: Install OpenCV
pip install opencv-python-headless

# Verify installation
python -c "import cv2; print(cv2.__version__)"
```

### Runtime Issues

**Problem**: Service fails to start
```bash
# Check port availability
netstat -tlnp | grep 7781

# Check environment variables
cat .env

# Test dependencies
python -c "import exiftool, cv2, PIL; print('All dependencies OK')"
```

**Problem**: Metadata extraction failures
```bash
# Test ExifTool directly
exiftool /path/to/image.jpg

# Check file permissions
ls -la /path/to/image.jpg

# Verify file format support
file /path/to/image.jpg
```

### Performance Issues

**Problem**: Slow processing
- Reduce image size before analysis
- Disable advanced analysis for basic metadata extraction
- Use SSD storage for temporary files

**Problem**: Memory usage
- Process images sequentially rather than in batches
- Clear temporary files regularly
- Monitor system memory usage

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

### Data Privacy
- No metadata is stored permanently
- Temporary files are automatically cleaned
- GPS and personal information can be filtered
- Processing logs exclude sensitive data

---

**Documentation Version**: 1.0  
**Last Updated**: 2025-08-13  
**Service Version**: Production  
**Maintainer**: Animal Farm ML Team