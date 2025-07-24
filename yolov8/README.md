# YOLOv8 Object Detection Service

This service provides real-time object detection using Ultralytics YOLOv8, capable of detecting 80 different object classes with bounding boxes and confidence scores.

## Features

- REST API for object detection
- Discord bot integration with emoji reactions
- Real-time bounding box detection
- Confidence scoring and filtering
- IoU (Intersection over Union) threshold filtering
- Automatic model fallback (nano → small → medium → large → extra large)
- Comprehensive error handling and logging

## Setup

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
pip install ultralytics
```

### 2. Model Download

YOLOv8 models will be downloaded automatically on first use. The service tries models in this order:
- `yolov8n.pt` (Nano - fastest, smallest)
- `yolov8s.pt` (Small)
- `yolov8m.pt` (Medium)
- `yolov8l.pt` (Large)
- `yolov8x.pt` (Extra Large - slowest, most accurate)

### 3. Configure Environment

Create a `.env` file with your configuration:

```bash
# Discord Configuration (for bot)
DISCORD_TOKEN=your_discord_bot_token
DISCORD_GUILD=your_guild_name
DISCORD_CHANNEL=channel1,channel2

# Service Settings
PORT=7773
PRIVATE=false

# API Configuration (required for emoji lookup)
API_HOST=localhost
API_PORT=8080
API_TIMEOUT=2.0

# Database Configuration (optional)
MYSQL_HOST=localhost
MYSQL_USERNAME=your_username
MYSQL_PASSWORD=your_password
MYSQL_DB=your_database
```

### 4. Run Services

**REST API:**
```bash
python REST.py
# or
./yolo.sh
```

**Discord Bot:**
```bash
python YOLO-discord-rest.py
# or
./discord.sh
```

## API Usage

### REST API Endpoints

- `GET /health` - Health check and model status
- `GET /classes` - List all 80 detectable object classes
- `GET /?url=<image_url>` - Detect objects in image from URL
- `GET /?path=<local_path>` - Detect objects in local image (if not in private mode)
- `POST /` - Upload and detect objects in image file

### Example Usage

**Detect objects from URL:**
```bash
curl "http://localhost:7776/?url=https://example.com/image.jpg"
```

**Upload file:**
```bash
curl -X POST -F "uploadedfile=@image.jpg" http://localhost:7776/
```

**Response format:**
```json
{
  "YOLO": {
    "detections": [
      {
        "class_id": 16,
        "class_name": "dog",
        "confidence": 0.892,
        "bbox": {
          "x1": 150,
          "y1": 200,
          "x2": 450,
          "y2": 600,
          "width": 300,
          "height": 400
        },
        "emoji": "🐕"
      },
      {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.756,
        "bbox": {
          "x1": 100,
          "y1": 50,
          "x2": 300,
          "y2": 500,
          "width": 200,
          "height": 450
        },
        "emoji": "👤"
      }
    ],
    "total_detections": 2,
    "image_dimensions": {
      "width": 640,
      "height": 480
    },
    "model_info": {
      "confidence_threshold": 0.25,
      "iou_threshold": 0.45,
      "device": "cuda"
    },
    "status": "success"
  }
}
```

## Discord Bot

The Discord bot automatically:
- Processes image attachments in configured channels
- Detects objects using the YOLO API
- Adds emoji reactions for detected objects (up to 3 unique emojis)
- Logs detections to database (if configured)
- Ignores duplicate emojis and low-confidence detections

## Detectable Object Classes (80 COCO Classes)

The service can detect these object categories:

**People & Animals:**
person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:**
bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Outdoor Objects:**
traffic light, fire hydrant, stop sign, parking meter, bench

**Sports & Recreation:**
frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Kitchen & Dining:**
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Furniture & Household:**
chair, couch, potted plant, bed, dining table, toilet

**Electronics:**
tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator

**Personal Items:**
backpack, umbrella, handbag, tie, suitcase, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## Configuration Options

- **Confidence Threshold**: 0.25 (adjustable, minimum confidence for detections)
- **IoU Threshold**: 0.45 (adjustable, for non-maximum suppression)
- **Max Detections**: 100 per image
- **Max Emoji Reactions**: 3 unique emojis per Discord message

## Performance & Model Information

- **Model**: Ultralytics YOLOv8 (multiple sizes available)
- **Input**: Images up to 8MB
- **Output**: Bounding boxes with confidence scores
- **Classes**: 80 COCO object categories
- **Performance**: Real-time detection on GPU, fast on CPU

## Troubleshooting

1. **Model loading errors**: Install ultralytics: `pip install ultralytics`
2. **CUDA errors**: Ensure PyTorch CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
3. **Discord bot not responding**: Check token and channel configuration
4. **API connection issues**: Verify API_HOST and API_PORT settings
5. **No detections**: Check confidence threshold (default 0.25) and image quality

## Technical Details

- **Framework**: Flask (REST API), Discord.py (bot), Ultralytics YOLOv8 (detection)
- **Database**: MySQL (optional)
- **Deployment**: Systemd services available
- **Hardware**: GPU recommended for best performance, CPU supported

## Systemd Service

Service files are available in the `services/` directory for production deployment.

## Model Comparison

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | ~6MB | Fastest | Good | Real-time, edge devices |
| YOLOv8s | ~22MB | Fast | Better | Balanced performance |
| YOLOv8m | ~52MB | Medium | High | Production systems |
| YOLOv8l | ~87MB | Slow | Higher | High accuracy needs |
| YOLOv8x | ~136MB | Slowest | Highest | Maximum accuracy |