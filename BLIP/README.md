# BLIP Image Captioning Service

This service provides AI-powered image captioning using the BLIP (Bootstrapping Language-Image Pre-training) model.

## Features

- REST API for image captioning with v1 and v2 endpoints
- Multiple model sizes for different performance needs
- Secure file handling with validation
- Comprehensive error handling and logging
- Emoji extraction from captions
- CORS support for direct browser access

## Setup

### 1. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv blip_venv

# Activate virtual environment
source blip_venv/bin/activate

# Install dependencies
pip install torch torchvision transformers flask flask-cors pillow requests python-dotenv nltk
```

### 2. Download BLIP Models

First, you need to install the BLIP model implementation:

```bash
# Clone BLIP repository
git clone https://github.com/salesforce/BLIP.git
cp -r BLIP/models ./
```

Then download the required model:

```bash
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth
```

### 3. Configure Environment

Create a `.env` file with your configuration:

```bash
# Service Settings
PORT=7777
PRIVATE=False

# API Configuration (required for emoji lookup)
API_HOST=localhost
API_PORT=8080
API_TIMEOUT=2.0
```

**PRIVATE mode explanation:**
- `PRIVATE=False`: Service binds to all network interfaces (0.0.0.0) and allows local file path access via `/?path=` parameter
- `PRIVATE=True`: Service binds to localhost only (127.0.0.1) and disables local file path access for security

### 4. Run Services

**REST API:**
```bash
./BLIP.sh
```
## API Usage

### REST API Endpoints

**V1 Endpoints:**
- `GET /health` - Health check
- `GET /?url=<image_url>` - Caption image from URL
- `GET /?path=<local_path>` - Caption local image (if not in private mode)
- `POST /` - Upload and caption image file

**V2 Endpoints (Unified Response Format):**
- `GET /v2/analyze?image_url=<image_url>` - Caption image from URL with v2 format
- `GET /v2/analyze_file?file_path=<file_path>` - Caption image from file path with v2 format

### Example Usage

**Caption from URL:**
```bash
curl "http://localhost:7777/?url=https://example.com/image.jpg"
```

**Upload file:**
```bash
curl -X POST -F "uploadedfile=@image.jpg" http://localhost:7777/
```

**Response format:**
```json
{
  "BLIP": {
    "caption": "a dog sitting on a chair",
    "emojis": ["🐕", "🪑"],
    "status": "success"
  }
}
```

## Security Features

- File type validation
- File size limits (8MB default)
- Input sanitization
- Private mode for API access control
- Secure file cleanup
- Database credential protection

## Troubleshooting

1. **Model loading errors**: Ensure BLIP models are downloaded and the `models/` directory exists
2. **Import errors**: Install BLIP dependencies: `pip install transformers torch torchvision`
3. **Discord bot not responding**: Check token and channel configuration
4. **API connection issues**: Verify API_URL and API_PORT settings

## Model Information

| Model | Size | Description |
|-------|------|-------------|
| model_base_14M.pth | 14M | Fastest, lowest quality |
| model_base.pth | 113M | Good balance (recommended) |
| model_large.pth | 447M | Better quality, slower |
| model_base_capfilt_large.pth | 447M | Highest quality |

## Systemd Service

Service files are available in the `services/` directory for production deployment.
