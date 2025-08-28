# Colors Analysis Service

**Port**: 7770  
**Framework**: Haishoku + PIL  
**Purpose**: Professional color analysis with Copic and Prismacolor marker mapping  
**Status**: ✅ Active

## Overview

The Colors service provides professional-grade color analysis using the Haishoku library combined with PIL for image processing. It extracts dominant colors, generates color palettes, and maps colors to professional art supply systems (Copic markers and Prismacolor pencils) with temperature analysis.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **Professional Color Matching**: Maps colors to Copic and Prismacolor systems
- **Temperature Analysis**: Determines warm/cool/neutral color temperature for palettes
- **Configurable Systems**: Choose between Copic, Prismacolor, or both
- **Security**: File validation, size limits, secure cleanup
- **Performance**: Fast color analysis with automatic cleanup

## Installation

### Prerequisites

- Python 3.8+
- PIL/Pillow for image processing
- Haishoku library for color analysis
- 512MB+ RAM

### 1. Environment Setup

```bash
# Navigate to colors directory
cd /home/sd/animal-farm/colors

# Create virtual environment
python3 -m venv colors_venv

# Activate virtual environment
source colors_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dependency Installation

Install the required Python packages:

```bash
# Core dependencies
pip install flask python-dotenv pillow haishoku requests flask-cors
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the colors directory:

```bash
# Service Configuration
PORT=7770                    # Service port (default: 7770)
PRIVATE=false               # Access mode (false=public, true=localhost-only)

# Color System Configuration
COLOR_SYSTEM=copic  # Options: copic, prismacolor, prismacolor_pencils, or any comma-separated combination
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (false=public, true=localhost-only) |
| `COLOR_SYSTEM` | Yes | - | Color systems to use (copic, prismacolor, prismacolor_pencils, or combinations) |

### Color System Options

The service supports multiple professional color systems:

| System | Speed | Coverage | Description |
|--------|-------|----------|-------------|
| `copic` | Fastest | 358 colors | Alcohol-based markers |
| `prismacolor` | Fast | 150 colors | Colored pencils |
| `copic,prismacolor` | Slower | Both systems | Comprehensive analysis |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Color Analysis",
  "features": {
    "color_systems": ["Copic"],
    "analysis_types": ["dominant_color", "color_palette", "emoji_mapping"],
    "supported_formats": ["JPEG", "PNG", "GIF", "BMP"]
  },
  "endpoints": [
    "GET /health - Health check",
    "GET /analyze?url=<image_url> - Analyze colors from URL", 
    "GET /analyze?file=<file_path> - Analyze colors from file",
    "POST /analyze - Analyze colors from uploaded file",
    "GET /v2/analyze?image_url=<image_url> - V2 compatibility (deprecated)",
    "GET /v2/analyze_file?file_path=<file_path> - V2 compatibility (deprecated)"
  ]
}
```

### Analyze Colors (Unified Endpoint)

The unified `/analyze` endpoint accepts either URL or file path input:

#### Analyze Image from URL
```bash
GET /analyze?url=<image_url>
```

**Example:**
```bash
curl "http://localhost:7770/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7770/analyze?file=/path/to/image.jpg"
```

#### POST Request (File Upload)
```bash
POST /analyze
Content-Type: multipart/form-data
```

**Example:**
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7770/analyze
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
      "framework": "Haishoku + PIL"
    },
    "processing_time": 0.158
  },
  "predictions": [
    {
      "palette": {
        "colors": [
          {
            "copic": "Neutral Gray (N-10)",
            "hex": "#2f2725",
            "prismacolor": "French Grey 90% (PM 163)",
            "temperature": "neutral"
          },
          {
            "copic": "Toner Gray (T-7)",
            "hex": "#7d7c78",
            "prismacolor": "French Grey 70% (PM 161)",
            "temperature": "neutral"
          },
          {
            "copic": "Champagne (E71)",
            "hex": "#e9e0da",
            "prismacolor": "Putty (PM 80)",
            "temperature": "neutral"
          },
          {
            "copic": "Dark Red (R89)",
            "hex": "#862740",
            "prismacolor": "Mahogany Red (PM 150)",
            "temperature": "warm"
          },
          {
            "copic": "Cool Gray (C-8)",
            "hex": "#585f64",
            "prismacolor": "Neutral Grey 80% (PM 223)",
            "temperature": "cool"
          },
          {
            "copic": "Light Mahogany (E07)",
            "hex": "#d3896b",
            "prismacolor": "Walnut (PM 90)",
            "temperature": "neutral"
          },
          {
            "copic": "Fig (E87)",
            "hex": "#78664b",
            "prismacolor": "Dark Brown (PM 88)",
            "temperature": "neutral"
          },
          {
            "copic": "Cool Gray (C-9)",
            "hex": "#3e4449",
            "prismacolor": "Cool Grey 90% (PM 116)",
            "temperature": "cool"
          }
        ],
        "temperature": "cool"
      },
      "primary": {
        "copic": "Neutral Gray (N-10)",
        "hex": "#282e30",
        "prismacolor": "French Grey 90% (PM 163)",
        "temperature": "neutral"
      }
    }
  ],
  "service": "colors",
  "status": "success"
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
cd /home/sd/animal-farm/colors
python3 REST.py
```

### Service Script

```bash
# Using startup script (if available)
./colors.sh
```

### Systemd Service

```bash
# Start service
sudo systemctl start colors-api

# Enable auto-start
sudo systemctl enable colors-api

# Check status
sudo systemctl status colors-api

# View logs
journalctl -u colors-api -f
```

## Performance Optimization

### Hardware Requirements

| Configuration | RAM | CPU | Response Time |
|---------------|-----|-----|---------------|
| Minimum | 512MB | 1 core | 0.2-0.5s |
| Recommended | 1GB+ | 2+ cores | 0.1-0.3s |
| High Volume | 2GB+ | 4+ cores | 0.05-0.2s |

### Performance Tuning

- **File Size Limit**: 8MB maximum (hardcoded)
- **Concurrent Requests**: Flask threaded mode enabled
- **Memory Usage**: ~50-100MB base + image size during processing
- **Color System Impact**: Single system (copic) ~30% faster than dual system

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Must provide either 'url' or 'file' parameter` | Missing input parameter | Provide exactly one parameter |
| `Cannot provide both 'url' and 'file' parameters` | Both parameters provided | Use only one parameter |
| `File not found: <path>` | Invalid file path | Check file exists and path is correct |
| `Failed to download image` | Network/URL issue | Verify URL is accessible |
| `Downloaded file too large` | Image > 8MB | Use smaller image or compress |
| `Failed to process image` | Invalid image format | Use supported formats (JPEG, PNG, GIF, BMP) |

### Error Response Format

```json
{
  "service": "colors",
  "status": "error",
  "predictions": [],
  "error": {"message": "Error description"},
  "metadata": {"processing_time": 0.001}
}
```

## Integration Examples

### Python Integration

```python
import requests

# Analyze image from URL
response = requests.get(
    'http://localhost:7770/analyze',
    params={'url': 'https://example.com/image.jpg'}
)

# POST file upload
with open('/path/to/image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:7770/analyze',
        files={'file': f}
    )
result = response.json()

# Extract dominant color
primary_color = result['predictions'][0]['primary']
print(f"Primary color: {primary_color['copic']} ({primary_color['hex']})")

# Extract palette
palette = result['predictions'][0]['palette']
print(f"Palette temperature: {palette['temperature']}")
for color in palette['colors']:
    print(f"  {color['copic']} - {color['hex']} ({color['temperature']})")
```

### JavaScript Integration

```javascript
// Analyze image from URL
async function analyzeColors(imageUrl) {
    const response = await fetch(`http://localhost:7770/analyze?url=${encodeURIComponent(imageUrl)}`);
    const result = await response.json();
    
    if (result.status === 'success') {
        const primary = result.predictions[0].primary;
        const palette = result.predictions[0].palette;
        
        console.log(`Primary: ${primary.copic} (${primary.hex})`);
        console.log(`Palette temperature: ${palette.temperature}`);
        
        palette.colors.forEach(color => {
            console.log(`${color.copic} - ${color.hex} (${color.temperature})`);
        });
    }
}

// Usage
analyzeColors('https://example.com/image.jpg');
```

### cURL Examples

```bash
# Basic color analysis
curl "http://localhost:7770/analyze?url=https://example.com/image.jpg"

# File analysis
curl "http://localhost:7770/analyze?file=/path/to/image.jpg"

# POST file upload
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7770/analyze

# Health check
curl "http://localhost:7770/health"

# V2 compatibility (deprecated)
curl "http://localhost:7770/v2/analyze?image_url=https://example.com/image.jpg"
```

## Troubleshooting

### Installation Issues

**Problem**: Import error for haishoku
```bash
# Solution
pip install haishoku
```

**Problem**: PIL/Pillow import errors
```bash
# Solution
pip uninstall pillow
pip install pillow
```

### Runtime Issues

**Problem**: Port already in use
```bash
# Check what's using the port
lsof -i :7770

# Change port in .env file
echo "PORT=7771" >> .env
```

**Problem**: Permission denied on startup script
```bash
# Fix permissions
chmod +x colors.sh
```

### Performance Issues

**Problem**: Slow response times
- Check image size (reduce if > 2MB for faster processing)
- Use single color system (`COLOR_SYSTEM=copic`) for speed
- Ensure sufficient RAM available

**Problem**: Memory usage too high
- Reduce concurrent requests
- Check for memory leaks in long-running processes
- Restart service periodically if needed

### Configuration Issues

**Problem**: Environment variable errors
```bash
# Check .env file exists and has correct format
cat .env

# Verify all required variables are set
python3 -c "from dotenv import load_dotenv; load_dotenv(); import os; print([k for k in ['PORT', 'PRIVATE', 'COLOR_SYSTEM'] if not os.getenv(k)])"
```

## Security Considerations

### Access Control

- **Private Mode**: Set `PRIVATE=true` for localhost-only access
- **File Path Access**: Disabled in private mode for security
- **Input Validation**: All inputs validated before processing

### File Security

- **Size Limits**: 8MB maximum file size
- **Format Validation**: Only supported image formats accepted  
- **Temporary Files**: Automatically cleaned up after processing
- **Path Validation**: File paths validated to prevent directory traversal

### Network Security

- **Timeout Protection**: Download timeouts prevent hanging connections
- **CORS Configuration**: Configured for direct browser access
- **Error Information**: Error messages don't expose system internals

---

**Generated**: August 13, 2025  
**Framework Version**: Haishoku 1.1.8 + PIL 9.5.0  
**Service Version**: 3.0 (Modernized)