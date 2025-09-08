# BLIP2 Image Captioning Service

**Port**: 7777  
**Framework**: LAVIS BLIP2 (Bootstrapping Language-Image Pre-training v2)  
**Purpose**: Next-generation AI-powered image captioning with enhanced accuracy  
**Status**: ✅ Active

## Overview

BLIP2 provides state-of-the-art image captioning using Salesforce's BLIP2 model through the LAVIS framework. This service offers improved caption quality over BLIP v1, with better understanding of complex scenes and more natural language generation. The service is a **drop-in replacement** for the original BLIP service, running on the same port 7777 with identical API endpoints.

## Features

- **Enhanced BLIP2 Model**: Latest generation captioning with improved accuracy
- **Drop-in Replacement**: Compatible with existing BLIP integrations (same port, same API)
- **Docker-First Approach**: Containerized deployment eliminates dependency hell
- **GPU Acceleration**: Full CUDA support for fast inference (0.166s vs 2.8s on CPU)
- **Unified Input Handling**: Single endpoint for URL, file path, and upload analysis
- **Emoji Integration**: Automatic word-to-emoji mapping using GitHub-based configuration
- **Model Flexibility**: Support for multiple BLIP2 model configurations
- **Robust Config Loading**: GitHub-first config with local caching

## Installation

### 🐳 Recommended: Docker Installation

**Prerequisites:**
- Docker installed
- NVIDIA GPU with Docker GPU support (optional, will fallback to CPU)

```bash
# Navigate to BLIP2 directory
cd /home/sd/animal-farm/BLIP2

# Build the Docker image
sudo docker build -t blip2-service .

# Run with GPU support (recommended)
sudo docker run --gpus all -p 7777:7777 blip2-service

# Or run on CPU only (slower but still works)
sudo docker run -p 7777:7777 blip2-service

# Optional: Persist model cache to avoid re-downloading 2GB model
mkdir -p ~/docker-cache/torch
sudo docker run --gpus all -p 7777:7777 -v ~/docker-cache/torch:/root/.cache/torch blip2-service
```

### 🐍 Alternative: Virtual Environment Installation

**Warning**: We strongly recommend using Docker instead. The dependency requirements for LAVIS BLIP2 are complex and fragile.

**Prerequisites:**
- Python 3.11
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended
- Significant patience for dependency resolution

```bash
# Navigate to BLIP2 directory
cd /home/sd/animal-farm/BLIP2

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install PyTorch with CUDA 11.8 support first
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install exact working dependencies (this is critical!)
pip install -r requirements.installed.txt

# Clone LAVIS repository
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS

# Install LAVIS without dependencies (to avoid conflicts)
pip install -e . --no-deps

# Return to BLIP2 directory
cd ..

# Start service
python REST.py
```

**Note**: The `requirements.installed.txt` file contains the exact working versions discovered through extensive dependency archaeology. Do not modify these versions unless you enjoy debugging ML library conflicts.

## Configuration

### Environment Variables (.env)

```bash
# Service Configuration
PORT=7777                    # Service port (drop-in replacement for BLIP)
PRIVATE=false               # Access mode (false=public, true=localhost-only)
AUTO_UPDATE=True            # Enable GitHub-based config updates
```

### Model Configuration

BLIP2 uses the LAVIS framework with the following default configuration:
- **Model**: `blip_caption` 
- **Type**: `large_coco`
- **Auto-download**: Models are downloaded automatically on first run (~2GB)

## API Endpoints

The BLIP2 service provides **identical endpoints** to the original BLIP service, making it a perfect drop-in replacement.

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "device": "cuda:0"
}
```

### Analyze Image (Unified Endpoint)

#### Analyze Image from URL
```bash
GET /analyze?url=<image_url>
GET /v3/analyze?url=<image_url>  # V3 API endpoint
```

**Example:**
```bash
curl "http://localhost:7777/v3/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /analyze?file=<file_path>
GET /v3/analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7777/v3/analyze?file=/path/to/image.jpg"
```

#### POST Request (File Upload)
```bash
POST /analyze
POST /v3/analyze
Content-Type: multipart/form-data
```

**Example:**
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7777/v3/analyze
```

**Response Format:**
```json
{
  "metadata": {
    "model_info": {
      "framework": "BLIP2 (Bootstrapping Language-Image Pre-training v2)"
    },
    "processing_time": 0.166
  },
  "predictions": [
    {
      "emoji_mappings": [
        { "emoji": "🧑", "word": "man" },
        { "emoji": "👔", "word": "military_uniform" },
        { "emoji": "✋", "word": "handing" },
        { "emoji": "🌼", "word": "flowers" }
      ],
      "text": "a man in a military uniform handing flowers to a little girl"
    }
  ],
  "service": "blip2",
  "status": "success"
}
```

## Docker Configuration Details

### GPU Support Setup

To enable GPU acceleration in Docker:

1. **Install nvidia-container-toolkit:**
```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install and configure
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2. **Run with GPU support:**
```bash
sudo docker run --gpus all -p 7777:7777 blip2-service
```

### Performance Comparison

| Configuration | Processing Time | Notes |
|---------------|----------------|-------|
| Docker + GPU | ~0.166s | Recommended |
| Docker + CPU | ~2.8s | Fallback option |
| Native GPU | ~0.1-0.2s | Fastest, but harder to install |

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: No explicit limit (Docker handles memory management)
- **Input Methods**: URL, file upload, local path

### Output Features
- **Enhanced Captions**: Improved accuracy over BLIP v1
- **Emoji Mapping**: Automatic word-to-emoji conversion via GitHub config
- **Multi-word Expressions**: Support for compound terms
- **Processing Metadata**: Timing and model information

## Integration Examples

### Drop-in Replacement Usage

Since BLIP2 runs on the same port (7777) with identical API endpoints, existing BLIP integrations work without modification:

```python
import requests

# Existing BLIP code works unchanged
response = requests.get(
    "http://localhost:7777/v3/analyze",
    params={"url": "https://example.com/image.jpg"}
)

result = response.json()
# Same response structure as BLIP v1
```

### Enhanced Features

```python
# Take advantage of improved BLIP2 captions
def analyze_image_with_blip2(image_url):
    response = requests.get(
        "http://localhost:7777/v3/analyze",
        params={"url": image_url}
    )
    
    result = response.json()
    if result["status"] == "success":
        prediction = result["predictions"][0]
        
        # Better quality captions from BLIP2
        caption = prediction["text"]
        
        # Same emoji mapping as BLIP v1
        emojis = [m["emoji"] for m in prediction["emoji_mappings"]]
        
        return {
            "caption": caption,
            "emojis": emojis,
            "processing_time": result["metadata"]["processing_time"]
        }
```

## Troubleshooting

### Docker Issues

**Problem**: `could not select device driver with capabilities: [[gpu]]`
```bash
# Solution: Install nvidia-container-toolkit (see GPU Setup section)
# Or run without GPU: sudo docker run -p 7777:7777 blip2-service
```

**Problem**: Model downloads every time
```bash
# Solution: Use persistent volume for model cache
mkdir -p ~/docker-cache/torch
sudo docker run --gpus all -p 7777:7777 -v ~/docker-cache/torch:/root/.cache/torch blip2-service
```

### Virtual Environment Issues

**Problem**: Dependency conflicts during installation
```bash
# Solution: Use exact versions from requirements.installed.txt
# DO NOT upgrade packages - they were archaeologically discovered to work
```

**Problem**: Import errors for LAVIS
```bash
# Solution: Ensure LAVIS is installed with --no-deps
cd LAVIS
pip install -e . --no-deps
```

**Problem**: CUDA not available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

### Service Issues

**Problem**: Service fails to start
```bash
# Check if port 7777 is already in use
netstat -tlnp | grep 7777
# Kill other services using the port if needed
```

**Problem**: GitHub config loading fails
```bash
# Service falls back to local cache automatically
# Check network connectivity or set AUTO_UPDATE=False in .env
```

## Migration from BLIP v1

BLIP2 is designed as a drop-in replacement:

1. **No code changes required** - Same API endpoints and response format
2. **Same port (7777)** - Just stop BLIP v1 and start BLIP2
3. **Improved accuracy** - Better captions with same emoji mapping
4. **Performance gains** - Especially with GPU acceleration

To migrate:
```bash
# Stop BLIP v1 service
sudo systemctl stop BLIP-api  # or however you're running BLIP v1

# Start BLIP2 Docker container
cd /home/sd/animal-farm/BLIP2
sudo docker run --gpus all -p 7777:7777 blip2-service

# Your existing code continues to work unchanged
```

## Security Considerations

### Docker Security
- Container runs with minimal privileges
- Automatic cleanup of temporary files
- Network isolation from host system
- No persistent data storage (unless volumes mounted)

### Access Control
- Set `PRIVATE=true` for localhost-only access
- Use reverse proxy with authentication for public access
- Validate all input URLs and file paths

## Why Docker?

The LAVIS BLIP2 environment is notoriously difficult to install due to:
- Complex ML dependency chains
- Version conflicts between PyTorch, transformers, and other libraries
- System-specific CUDA requirements
- Fragile numpy/opencv ABI compatibility

Docker solves this by:
- **Encapsulating the entire environment** in a reproducible container
- **Eliminating "it works on my machine" problems**
- **Simplifying deployment** across different systems
- **Avoiding dependency hell** for end users

As one developer noted: "DO NOT FUCK WITH THIS VENV" - Docker lets you avoid that entirely!

---

**Documentation Version**: 1.0  
**Last Updated**: 2025-09-08  
**Service Version**: Production  
**Maintainer**: Animal Farm ML Team