# CLIP Image Classification Service

This service provides AI-powered image classification using OpenAI's CLIP (Contrastive Language-Image Pre-training) model with custom label sets and emoji mapping.

## Features

- REST API for image classification with v1 and v2 endpoints
- Custom label categories (animals, objects, foods, people, etc.)
- Confidence threshold filtering
- Emoji extraction from classifications
- CORS support for direct browser access
- Configurable prediction limits

## Setup

### 1. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv clip_venv

# Activate virtual environment
source clip_venv/bin/activate

# Install dependencies
pip install torch torchvision clip-by-openai flask flask-cors pillow requests python-dotenv numpy
```

### 2. Configure Environment

Create a `.env` file with your configuration:

```bash
# Service Settings
PORT=7778
PRIVATE=False

# CLIP Model Configuration
CLIP_MODEL=ViT-B/32
CLIP_CONFIDENCE_THRESHOLD=0.01
CLIP_MAX_PREDICTIONS=10

# API Configuration (required for emoji lookup)
API_HOST=localhost
API_PORT=8080
API_TIMEOUT=2.0
```

**PRIVATE mode explanation:**
- `PRIVATE=False`: Service binds to all network interfaces (0.0.0.0) and allows local file path access via `/?path=` parameter
- `PRIVATE=True`: Service binds to localhost only (127.0.0.1) and disables local file path access for security

### 3. Run Services

**REST API:**
```bash
./CLIP.sh
```

## API Usage

### REST API Endpoints

**V1 Endpoints:**
- `GET /health` - Health check
- `GET /?url=<image_url>` - Classify image from URL
- `GET /?path=<local_path>` - Classify local image (if not in private mode)
- `POST /` - Upload and classify image file

**V2 Endpoints (Unified Response Format):**
- `GET /v2/analyze?image_url=<image_url>` - Classify image from URL with v2 format
- `GET /v2/analyze_file?file_path=<file_path>` - Classify image from file path with v2 format

### Example Usage

**Classify from URL:**
```bash
curl "http://localhost:7778/?url=https://example.com/image.jpg"
```

**Upload file:**
```bash
curl -X POST -F "uploadedfile=@image.jpg" http://localhost:7778/
```

**V1 Response format:**
```json
{
  "CLIP": {
    "predictions": [
      {"label": "dog", "confidence": 0.85},
      {"label": "animal", "confidence": 0.72}
    ],
    "emojis": ["🐕", "🐾"],
    "status": "success"
  }
}
```

**V2 Response format:**
```json
{
  "service": "clip",
  "status": "success",
  "predictions": [
    {
      "type": "classification",
      "label": "dog",
      "confidence": 0.85,
      "properties": {
        "category": "animals"
      }
    }
  ],
  "metadata": {
    "processing_time": 0.234,
    "model_info": {
      "name": "ViT-B/32",
      "framework": "CLIP"
    }
  }
}
```

## Label Categories

The service uses curated label files from the `labels/` directory:

- `labels_animals.txt` - Animals and pets
- `labels_coco.txt` - COCO dataset objects (default)
- `labels_foods.txt` - Food items and dishes
- `labels_objects.txt` - Common objects and items
- `labels_people.txt` - People, professions, roles
- `labels_plants.txt` - Plants and vegetation
- `labels_sports.txt` - Sports and activities
- `labels_transport.txt` - Vehicles and transportation
- `labels_weather.txt` - Weather conditions

See `LABELS_README.md` for detailed information about managing labels.

## Configuration Options

- `CLIP_MODEL`: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14, etc.)
- `CLIP_CONFIDENCE_THRESHOLD`: Minimum confidence for predictions (0.0-1.0)
- `CLIP_MAX_PREDICTIONS`: Maximum number of predictions to return

## Security Features

- File type validation
- File size limits (8MB default)
- Input sanitization
- Private mode for API access control
- Secure file cleanup
- Database credential protection

## Troubleshooting

1. **Model loading errors**: Ensure CLIP is installed: `pip install clip-by-openai`
2. **Import errors**: Install dependencies: `pip install torch torchvision`
3. **Low confidence predictions**: Adjust `CLIP_CONFIDENCE_THRESHOLD` in .env
4. **API connection issues**: Verify API_HOST and API_PORT settings

## Systemd Service

Service files are available in the `services/` directory for production deployment.