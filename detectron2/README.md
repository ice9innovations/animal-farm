# Detectron2 Object Detection and Instance Segmentation Service

This service provides advanced object detection and instance segmentation using Facebook's Detectron2 framework, capable of detecting and segmenting objects from the 80 COCO classes with high accuracy.

## Features

- REST API for object detection and instance segmentation
- Discord bot integration with emoji reactions
- Support for multiple model architectures (Mask R-CNN, RetinaNet, etc.)
- Bounding box detection with confidence scores
- Instance segmentation masks
- Automatic configuration file discovery
- Comprehensive error handling and logging
- Health check endpoints

## Setup

### 1. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv detectron2_venv

# Activate virtual environment
source detectron2_venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install flask flask-cors pillow opencv-python python-dotenv requests

# Install Detectron2 (choose appropriate version for your system)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### 2. Model Configuration

The service uses the Detectron2 configuration file:
- `/home/sd/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`

This file is included with the Detectron2 installation and provides the Faster R-CNN model configuration for COCO object detection.

### 3. Configure Environment

Create a `.env` file with your configuration:

```bash
# Copy environment template
cp .env.sample .env

# Edit .env with your settings
```

### 4. Run Services

**REST API:**
```bash
python REST.py
# or
./detectron2.sh
```

**Discord Bot:**
```bash
python detectron-discord-rest.py
# or
./discord.sh
```

## API Usage

### REST API Endpoints

- `GET /health` - Health check and model status
- `GET /classes` - List all 80 detectable COCO classes
- `GET /?url=<image_url>` - Detect objects in image from URL
- `GET /?path=<local_path>` - Detect objects in local image (if not in private mode)
- `POST /` - Upload and detect objects in image file

### Example Usage

**Detect objects from URL:**
```bash
curl "http://localhost:7771/?url=https://example.com/image.jpg"
```

**Upload file:**
```bash
curl -X POST -F "uploadedfile=@image.jpg" http://localhost:7771/
```

**Response format:**
```json
{
  "DETECTRON": {
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
      }
    ],
    "total_detections": 1,
    "image_dimensions": {
      "width": 640,
      "height": 480
    },
    "model_info": {
      "confidence_threshold": 0.5,
      "detection_time": 0.234,
      "framework": "Detectron2"
    },
    "status": "success"
  }
}
```

## Discord Bot

The Discord bot automatically:
- Processes image attachments in configured channels
- Detects objects using the Detectron2 API
- Performs instance segmentation when available
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

- **Confidence Threshold**: 0.5 (adjustable, minimum confidence for detections)
- **Max File Size**: 8MB for uploaded images
- **Device**: Automatic GPU/CPU detection
- **Max Emoji Reactions**: 3 unique emojis per Discord message

## Performance & Model Information

- **Framework**: Facebook Detectron2
- **Models**: Mask R-CNN, RetinaNet, Panoptic FPN (configurable)
- **Input**: Images up to 8MB
- **Output**: Bounding boxes, confidence scores, instance segmentation masks
- **Classes**: 80 COCO object categories
- **Performance**: High accuracy, optimized for GPU

## Troubleshooting

1. **Model loading errors**: Ensure Detectron2 is properly installed
2. **Config file not found**: Check config file paths and permissions
3. **CUDA errors**: Verify PyTorch CUDA compatibility
4. **Discord bot not responding**: Check token and channel configuration
5. **API connection issues**: Verify API_URL and API_PORT settings
6. **No detections**: Check confidence threshold and image quality

## Technical Details

- **Framework**: Flask (REST API), Discord.py (bot), Detectron2 (detection)
- **Database**: MySQL (optional)
- **Deployment**: Systemd services available
- **Hardware**: GPU recommended for best performance, CPU supported

## Files Structure

```
detectron2/
├── REST.py                           # REST API service
├── detectron-discord-rest.py         # Discord bot
├── detectron2.sh                     # API startup script
├── discord.sh                        # Discord bot startup script
├── .env.sample                       # Environment template
├── services/                         # Systemd service files
│   ├── detectron2-api.service
│   └── detectron2.service
├── coco_classes.txt                  # COCO class names
├── emojis.json                       # Emoji mappings
└── README.md                         # This file
```

## Systemd Service

Service files are available in the `services/` directory for production deployment:

```bash
# Install services
sudo cp services/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable detectron2-api detectron2
sudo systemctl start detectron2-api detectron2
```

## Model Architectures

Detectron2 supports multiple architectures:

| Architecture | Use Case | Speed | Accuracy |
|-------------|----------|-------|----------|
| Mask R-CNN | Instance segmentation | Medium | High |
| RetinaNet | Object detection | Fast | Good |
| Panoptic FPN | Panoptic segmentation | Slow | Highest |
| FCOS | Single-stage detection | Fast | Good |

## Integration Notes

This service is part of the Animal Farm voting ensemble system. It provides:
- High-accuracy object detection with bounding boxes
- Instance segmentation capabilities
- Consistent API format for the voting algorithm
- Emoji reactions for Discord integration
- Reliable error handling and logging