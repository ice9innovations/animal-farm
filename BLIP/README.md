# BLIP Image Captioning Service

This service provides AI-powered image captioning using the BLIP (Bootstrapping Language-Image Pre-training) model.

## Features

- REST API for image captioning
- Multiple model sizes for different performance needs
- Secure file handling with validation
- Comprehensive error handling and logging

## Setup

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Download BLIP Models

First, you need to install the BLIP model implementation:

```bash
# Clone BLIP repository
git clone https://github.com/salesforce/BLIP.git
cp -r BLIP/models ./
```

Then download the pre-trained models:

```bash
python download_model.py
```

**Recommended**: Download `model_base.pth` for the best balance of speed and quality.

### 3. Configure Environment

Create a `.env` file with your configuration:

```bash
# API Configuration
PORT=7777
PRIVATE=True

# API URL
API_URL=http://localhost
API_PORT=8080
API_TIMEOUT=2.0
```

### 4. Run Services

**REST API:**
```bash
./BLIP.sh
```
## API Usage

### REST API Endpoints

- `GET /health` - Health check
- `GET /?url=<image_url>` - Caption image from URL
- `GET /?path=<local_path>` - Caption local image (if not in private mode)
- `POST /` - Upload and caption image file

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
