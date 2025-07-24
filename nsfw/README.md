# NSFW Detection Service

This service provides NSFW (Not Safe For Work) content detection using Bumble's private-detector EfficientNet-v2 model, capable of analyzing images and determining the probability of inappropriate content.

## Features

- REST API for NSFW content detection
- Discord bot integration with automatic content moderation
- EfficientNet-v2 model (Bumble's private-detector)
- Configurable NSFW probability threshold
- Comprehensive error handling and logging
- Health check endpoints
- Database logging for content moderation tracking

## Model Information

This service uses **Bumble's private-detector** model:
- **Framework**: TensorFlow 2.x
- **Architecture**: EfficientNet-v2
- **Input Size**: 480x480 pixels
- **Output**: NSFW probability (0-100%)
- **Repository**: https://github.com/bumble-tech/private-detector

## Setup

### 1. Install Dependencies

```bash
# Install TensorFlow and required packages
pip install -r ../requirements.txt
pip install tensorflow>=2.8.0
```

### 2. Download Model

Download the pre-trained model from Bumble's repository:

```bash
# Create model directory
mkdir -p saved_model

# Download the model files from:
# https://github.com/bumble-tech/private-detector
# Follow the download instructions in their repository
```

**Note**: The model files are large (~200MB). Ensure you have sufficient disk space and download them according to Bumble's licensing terms.

### 3. Configure Environment

Copy and edit the environment file:

```bash
cp .env.sample .env
# Edit .env with your configuration
```

### 4. CuDNN Library Path Issue

If you get `DNN library initialization failed` errors with CUDA/CuDNN version mismatches, the startup script automatically handles this, but you can also set it manually:

**Find your CuDNN path:**
```bash
# Activate your venv first
source nsfw_venv/bin/activate

# Find the nvidia cudnn lib directory
find $(python -c "import site; print(site.getsitepackages()[0])") -name "libcudnn*" -type f | head -1 | xargs dirname
```

**Add to your shell before running:**
```bash
export LD_LIBRARY_PATH="path/from/above:$LD_LIBRARY_PATH"
```

**Or the nsfw.sh script automatically handles this:**
```bash
CUDNN_PATH="$(pwd)/nsfw_venv/lib/python3.9/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"
```

### 5. Run Services

**REST API:**
```bash
python REST.py
# or
./nsfw.sh
```

**Discord Bot:**
```bash
python private-discord-rest.py
# or
./discord.sh
```

## API Usage

### REST API Endpoints

- `GET /health` - Health check and model status
- `GET /?url=<image_url>` - Detect NSFW content in image from URL
- `GET /?path=<local_path>` - Detect NSFW content in local image (if not in private mode)
- `POST /` - Upload and detect NSFW content in image file

### Example Usage

**Detect NSFW from URL:**
```bash
curl "http://localhost:7774/?url=https://example.com/image.jpg"
```

**Upload file:**
```bash
curl -X POST -F "uploadedfile=@image.jpg" http://localhost:7774/
```

**Response format:**
```json
{
  "NSFW": {
    "probability": 12.456,
    "threshold": 35.0,
    "is_nsfw": false,
    "emoji": "",
    "model_info": {
      "framework": "TensorFlow",
      "model": "EfficientNet-v2 (Bumble private-detector)",
      "detection_time": 0.234,
      "input_size": "480x480",
      "threshold": 35.0
    },
    "status": "success"
  }
}
```

**NSFW Content Detected:**
```json
{
  "NSFW": {
    "probability": 87.234,
    "threshold": 35.0,
    "is_nsfw": true,
    "emoji": "🚫",
    "model_info": {
      "framework": "TensorFlow",
      "model": "EfficientNet-v2 (Bumble private-detector)",
      "detection_time": 0.189,
      "input_size": "480x480",
      "threshold": 35.0
    },
    "status": "success"
  }
}
```

## Discord Bot

The Discord bot automatically:
- Processes image attachments in configured channels
- Analyzes images for NSFW content using the NSFW Detection API
- Adds 🚫 emoji reaction when NSFW content is detected (above threshold)
- Logs detections to database for content moderation tracking
- Supports multiple image formats (JPG, PNG, GIF, WebP)
- Processes multiple attachments concurrently for better performance

### Content Moderation

- **Threshold**: Configurable NSFW probability threshold (default: 35%)
- **Reaction**: 🚫 emoji added to messages with NSFW content
- **Logging**: All detections logged to database with image hash, probability, and metadata
- **Privacy**: Uses SHA3-256 hash of image content for database storage

## Configuration Options

- **NSFW_THRESHOLD**: Probability threshold for flagging content (default: 35%)
- **MAX_FILE_SIZE**: Maximum upload size (8MB)
- **PRIVATE**: Enable/disable local file access
- **Database**: Optional MySQL logging for content moderation

## Performance & Technical Details

- **Framework**: TensorFlow 2.x with EfficientNet-v2
- **Input Processing**: Automatic resize and padding to 480x480
- **Memory Management**: GPU memory growth enabled
- **Preprocessing**: Custom normalization (-128/128)
- **Output**: Probability score (0-100%) with emoji reaction

## Model Accuracy

The Bumble private-detector model provides:
- **High Accuracy**: Trained on large-scale dataset
- **Robust Detection**: Handles various image types and quality
- **False Positive Rate**: Low with proper threshold tuning
- **Performance**: Fast inference (~200ms per image)

## Migration from Legacy Version

This service has been modernized with:
- **Structured API**: Consistent JSON responses
- **Better Error Handling**: Comprehensive exception management
- **Security Improvements**: File size limits and input validation
- **Performance**: Concurrent processing and optimized preprocessing
- **Logging**: Enhanced database logging with SHA3-256 hashing
- **Health Checks**: Model status monitoring

## Troubleshooting

1. **Model not loaded**: Ensure `./saved_model/` directory exists with downloaded model
2. **TensorFlow errors**: Install TensorFlow 2.8+: `pip install tensorflow>=2.8.0`
3. **GPU memory issues**: GPU memory growth is automatically enabled
4. **Discord bot not responding**: Check token and channel configuration
5. **API connection issues**: Verify PORT settings
6. **High false positives**: Adjust NSFW_THRESHOLD in environment

### Model Download Issues

If you encounter issues downloading the model:

1. Visit: https://github.com/bumble-tech/private-detector
2. Follow their official download instructions
3. Ensure you comply with their licensing terms
4. Verify model files are in `./saved_model/` directory

## Technical Architecture

- **Framework**: Flask (REST API), Discord.py (bot), TensorFlow (detection)
- **Database**: MySQL (optional)
- **Deployment**: Systemd services available
- **Hardware**: CPU/GPU compatible with automatic GPU detection

## Files Structure

```
nsfw/
├── REST.py                           # REST API service (modernized)
├── private-discord-rest.py           # Discord bot (modernized)
├── nsfw.sh                           # API startup script
├── discord.sh                        # Discord bot startup script
├── .env.sample                       # Environment template
├── saved_model/                      # TensorFlow model directory
│   ├── saved_model.pb                # Model graph
│   ├── variables/                    # Model weights
│   └── assets/                       # Model assets
└── README.md                         # This file
```

## Content Moderation Integration

This service is designed for content moderation workflows:
- **Automated Detection**: Real-time NSFW content flagging
- **Threshold Tuning**: Adjustable sensitivity for different communities
- **Audit Trail**: Complete logging for moderation review
- **Privacy Preserving**: Uses image hashes instead of storing actual images
- **Scalable**: Handles multiple concurrent requests

## Integration Notes

This service is part of the Animal Farm voting ensemble system. It provides:
- Reliable NSFW content detection and scoring
- Consistent API format for the voting algorithm
- Automated content moderation for Discord communities
- Comprehensive logging for audit and review purposes
- High-performance inference suitable for real-time applications

## Legal and Ethical Considerations

- **Model License**: Follow Bumble's licensing terms for the private-detector model
- **Privacy**: Image hashes are stored, not actual image content
- **Accuracy**: Review flagged content manually for important decisions
- **Bias**: Be aware of potential model biases in detection
- **Compliance**: Ensure usage complies with platform policies and local laws

## Performance Benchmarks

| Metric | Value |
|--------|--------|
| Average inference time | ~200ms |
| Throughput | ~5 images/second |
| Memory usage | ~2GB (with model loaded) |
| Accuracy | 95%+ (based on Bumble's benchmarks) |
| Input formats | JPG, PNG, GIF, WebP |
| Max file size | 8MB |