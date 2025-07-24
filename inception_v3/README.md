# Inception v3 Image Classification Service

This service provides advanced image classification using Google's Inception v3 deep neural network trained on ImageNet, capable of recognizing 1000 different object categories with high accuracy.

## Features

- REST API for image classification with v1 and v2 endpoints
- Discord bot integration with emoji reactions
- Support for 1000 ImageNet object categories
- Modern TensorFlow 2.x implementation
- GPU acceleration with CUDA support
- Confidence scoring and filtering
- Secure file handling with validation
- Comprehensive error handling and logging
- Health check endpoints
- CORS support for direct browser access

## Setup

### 1. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv inception_v3_venv

# Activate virtual environment
source inception_v3_venv/bin/activate

# Install dependencies
pip install tensorflow flask flask-cors pillow requests python-dotenv discord.py
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
PORT=7779
PRIVATE=false

# API Configuration (required for emoji lookup)
API_HOST=localhost
API_PORT=8080
API_TIMEOUT=2.0

# Discord Bot Configuration (optional)
DISCORD_TOKEN=your_discord_token
DISCORD_GUILD=your_guild_id
DISCORD_CHANNEL=your_channel_id
```

**PRIVATE mode explanation:**
- `PRIVATE=false`: Service binds to all network interfaces (0.0.0.0) and allows local file path access via `/?path=` parameter
- `PRIVATE=true`: Service binds to localhost only (127.0.0.1) and disables local file path access for security

### 3. CuDNN Library Path Issue

If you get `DNN library initialization failed` errors with CUDA/CuDNN version mismatches, you need to override the system CuDNN with the pip-installed version.

**Find your CuDNN path:**
```bash
# Activate your venv first
source inception_v3_venv/bin/activate

# Find the nvidia cudnn lib directory
find $(python -c "import site; print(site.getsitepackages()[0])") -name "libcudnn*" -type f | head -1 | xargs dirname
```

**Add to your shell before running:**
```bash
export LD_LIBRARY_PATH="path/from/above:$LD_LIBRARY_PATH"
```

**Or the inception.sh script automatically handles this:**
```bash
CUDNN_PATH="$(dirname "$0")/inception_v3_venv/lib/python3.9/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"
```

### 4. Run Services

**REST API:**
```bash
python REST.py
# or
./inception.sh
```

**Discord Bot:**
```bash
python inception-discord-rest.py
# or
./discord.sh
```

## API Usage

### REST API Endpoints

**V1 Endpoints:**
- `GET /health` - Health check and model status
- `GET /?url=<image_url>` - Classify image from URL
- `GET /?path=<local_path>` - Classify local image (if not in private mode)
- `POST /` - Upload and classify image file

**V2 Endpoints (Unified Response Format):**
- `GET /v2/analyze?image_url=<url>` - Classify image from URL with v2 format
- `GET /v2/analyze_file?file_path=<path>` - Classify image from file path with v2 format

### Example Usage

**Classify from URL:**
```bash
curl "http://localhost:7779/?url=https://example.com/image.jpg"
```

**V2 API:**
```bash
curl "http://localhost:7779/v2/analyze?image_url=https://example.com/image.jpg"
```

**Upload file:**
```bash
curl -X POST -F "uploadedfile=@image.jpg" http://localhost:7779/
```

### Response Format

**V2 format (recommended):**
```json
{
  "INCEPTION": {
    "classifications": [
      {
        "class_id": "n02084071",
        "class_name": "dog",
        "confidence": 0.892,
        "rank": 1,
        "emoji": "🐕"
      }
    ],
    "total_classifications": 1,
    "image_dimensions": {
      "width": 640,
      "height": 480
    },
    "model_info": {
      "confidence_threshold": 0.15,
      "classification_time": 0.234,
      "framework": "TensorFlow",
      "model": "Inception v3"
    },
    "status": "success"
  }
}
```

## Discord Bot

The Discord bot automatically:
- Processes image attachments in configured channels
- Classifies objects using the Inception v3 API
- Adds emoji reactions for top classifications (up to 3 unique emojis)
- Logs classifications to database (if configured)
- Ignores duplicate emojis and low-confidence predictions
- Handles multiple concurrent image requests

## Model Information

This service uses Google's Inception v3 model:

- **Framework**: TensorFlow 2.x with Keras applications
- **Model**: Inception v3 trained on ImageNet (1000 classes)
- **Input Size**: 299x299 pixels (automatically resized)
- **Performance**: ~200ms average processing time on GPU
- **Hardware**: GPU strongly recommended, CPU supported but slow
- **Memory**: ~2GB GPU memory required

## ImageNet Classes

The service can classify 1000 different object categories from ImageNet, including:

**Animals:** Various dog breeds, cats, birds, wildlife, marine life, insects
**Vehicles:** Cars, trucks, motorcycles, aircraft, boats, trains
**Objects:** Furniture, electronics, tools, instruments, sports equipment
**Food:** Fruits, vegetables, prepared foods, beverages
**Nature:** Plants, flowers, trees, landscapes, weather phenomena

*Full class list available in `imagenet_classes.txt`*

## Configuration Options

- **Confidence Threshold**: 0.15 (adjustable, minimum confidence for classifications)
- **Max File Size**: 8MB for uploaded images
- **GPU Memory Growth**: Enabled to prevent memory allocation issues
- **Max Emoji Reactions**: 3 unique emojis per Discord message

## Security Features

- File type validation (JPEG, PNG, WebP, GIF)
- File size limits (8MB default)
- Input sanitization and validation
- Private mode for API access control
- Secure file cleanup after processing
- Database credential protection

## Troubleshooting

1. **CuDNN version mismatch**: Use the LD_LIBRARY_PATH fix above
2. **Model loading errors**: Ensure TensorFlow is properly installed with GPU support
3. **Memory errors**: Configure TensorFlow memory growth or use CPU mode
4. **Import errors**: Install missing dependencies from requirements
5. **Discord bot not responding**: Check token, guild, and channel configuration
6. **API connection issues**: Verify API_HOST and API_PORT settings
7. **GPU not detected**: Check CUDA installation and TensorFlow GPU support

## Performance Optimization

- **GPU Usage**: Service automatically detects and uses available GPUs
- **Memory Management**: TensorFlow memory growth enabled to prevent allocation issues
- **Batch Processing**: Single image processing optimized for low latency
- **Concurrent Requests**: Flask handles multiple simultaneous classification requests

## Technical Details

- **Framework**: Flask (REST API), Discord.py (bot), TensorFlow 2.x (classification)
- **Database**: MySQL (optional, for logging)
- **Deployment**: Systemd services available
- **Hardware**: NVIDIA GPU with CUDA 11.2+ recommended

## Files Structure

```
inception_v3/
├── REST.py                           # REST API service
├── inception-discord-rest.py         # Discord bot
├── inception.sh                      # API startup script
├── discord.sh                        # Discord bot startup script
├── .env.sample                       # Environment template
├── services/                         # Systemd service files
│   ├── inception-api.service
│   └── inception.service
├── imagenet_classes.txt              # ImageNet class names
├── emoji_mappings.json               # Emoji mappings
└── README.md                         # This file
```

## Systemd Service

Service files are available in the `services/` directory for production deployment:

```bash
# Install services
sudo cp services/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable inception-api inception
sudo systemctl start inception-api inception
```

## Integration Notes

This service is part of the Animal Farm voting ensemble system. It provides:
- High-accuracy image classification with confidence scores
- 1000-class ImageNet object recognition
- Consistent API format for the voting algorithm
- Emoji reactions for Discord integration
- Reliable error handling and logging
- GPU acceleration for fast inference

## CUDA/GPU Requirements

- NVIDIA GPU with Compute Capability 3.5+
- CUDA 11.2 or later
- CuDNN 8.1 or later (handled via pip packages)
- 2GB+ GPU memory recommended
- TensorFlow GPU support verified during startup