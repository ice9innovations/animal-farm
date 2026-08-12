# 🐷 Animal Farm

**All animals are equal, but some animals are more equal than others**

Animal Farm is a distributed AI inference platform that orchestrates specialized models across diverse compute resources. It consumes a minimum of 25W of power in its smallest configuration. Workloads can be matched to the hardware best suited to run them allowing Animal Farm to support both low-power deployments and substantially more capable multi-GPU systems. 

The system is extensible. New models and capabilities can be added without requiring the rest of the architecture to understand how those models work. Instead of relying on a single general-purpose model, Animal Farm combines the outputs of specialized models through a consensus-based architecture. Each model contributes what it can observe, and those results can be combined and analyzed to produce a richer understanding of the input. 

## Architecture

**From each service according to its capability, to each according to its needs**

Animal Farm consists of multiple specialized AI services that work together in a distributed ensemble. Each service runs independently and communicates through RESTful APIs, allowing for scalable deployment across multiple machines or containers.

## Services

The current Animal Farm services live in this repository. Older model zoo services
have been moved to the [Zoo repository](https://github.com/ice9innovations/zoo)
and are listed separately below.

| Service | Port | Description |
|---------|------|-------------|
| **[gemini-api](gemini-api/)** | 7767 | Google Gemini vision API wrapper with emoji mapping |
| **[rembg](rembg/)** | 7768 | Background removal returning alpha masks |
| **[colors](colors/)** | 7770 | Color analysis and palette extraction |
| **[face](face/)** | 7772 | Face detection and facial keypoint analysis |
| **[yolov8](yolov8/)** | 7773 | Real-time COCO object detection |
| **[nsfw2](nsfw2/)** | 7774 | OpenNSFW2 content safety detection |
| **[ocr](ocr/)** | 7775 | GPU-backed OCR text extraction |
| **[BLIP2](BLIP2/)** | 7777 | BLIP2 image captioning |
| **[metadata](metadata/)** | 7781 | Image metadata extraction (EXIF, GPS, camera info) |
| **[llama-cpp](llama-cpp/)** | 7782 | Local LLaVA/Llama vision-language inference via llama.cpp |
| **[pose](pose/)** | 7786 | Human pose landmark detection and joint angle analysis |
| **[nudenet](nudenet/)** | 7789 | NudeNet+ category-level content moderation |
| **[moondream](moondream/)** | 7795 | Lightweight local vision-language captioning |
| **[qwen-cpp](qwen-cpp/)** | 7796 | Local Qwen3-VL inference via llama.cpp |
| **[claude-api](claude-api/)** | 7797 | Anthropic Claude/Haiku vision API wrapper |
| **[joycaption](joycaption/)** | 7798 | Local JoyCaption Hugging Face vision model wrapper |
| **[caption-summary](caption-summary/)** | 7799 | Caption and noun/verb consensus summarization |
| **[gpt-nano](gpt-nano/)** | 7800 | OpenAI GPT nano vision API wrapper |
| **[qr](qr/)** | 7801 | QR code and barcode detection and decoding |
| **[florence2](florence2/)** | 7803 | Florence-2 multi-task vision: captioning, detection, OCR, grounding, and segmentation |
| **[xai](xai/)** | 7805 | xAI Grok vision API wrapper |
| **[sam3](sam3/)** | 9779 | Text-prompted open-vocabulary image segmentation |

The root `docker-compose.yaml` currently covers the main RunPod deployment set:
BLIP2, Florence-2, Moondream, llama-cpp, qwen-cpp, Claude/Haiku, Gemini,
GPT Nano, YOLOv8, NudeNet, rembg, OCR, pose, face, colors, metadata, QR, and
caption-summary. Services that are not in the compose file can still be run from
their own directory using their service README.

### Archived Services

Archived and retired services are kept in the
[Zoo repository](https://github.com/ice9innovations/zoo) for reference. These
include BLIP v1, CLIP, CLIP_detection, clip-score, Detectron2, HAILO YOLO,
Ollama API, RT-DETRv2, RTMDet, SpeciesNet, Xception, Xception detection,
YOLO Objects365, and YOLO Open Images v7.

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for optimal performance)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ice9innovations/animal-farm.git
   cd animal-farm
   ```

2. **Install service dependencies:**
   Each service has its own virtual environment and requirements. See individual service README files for specific setup instructions.

3. **Configure services:**
   Copy `.env.sample` to `.env` in each service directory and configure as needed.

4. **Start services:**
   Each service can be started individually:
   ```bash
   cd [service-name]
   ./[service-name].sh
   ```

## Windmill Orchestration

Voting, consensus, and downstream post-processing are handled by the Windmill
pipeline. Keep voting behavior documented with the Windmill flows and workers;
this repository documents the service APIs and deployment wrappers that Windmill
calls.

## API Integration

Services are designed to work independently or as part of a larger distributed deployment:

- **Service APIs**: Each AI service exposes unified `/analyze` endpoints supporting URL, file, and POST inputs
- **Unified Response Format**: Services follow consistent JSON response schemas with metadata, predictions, and processing times
- **In-Memory Processing**: Services process images entirely in RAM using PIL Images for security and performance

### Example API Usage

All services follow the same unified endpoint pattern with three input methods:
- **URL Parameter**: `GET /analyze?url=<image_url>`
- **File Parameter**: `GET /analyze?file=<file_path>`  
- **POST Upload**: `POST /analyze` with multipart/form-data

Services process images entirely in memory using PIL Images, eliminating temporary file creation for improved security and performance.

#### Service Endpoints

All services now use the standardized `/analyze` endpoint with support for URL, file path, and POST file upload:

```bash
# Local file path
curl "http://192.168.0.101:7777/analyze?file=/path/to/image/file" | jq

# Image URL
curl "http://192.168.0.101:7777/analyze?url=https://example.com/image.jpg" | jq

# POST file upload
curl -X POST -F "file=@/path/to/image.jpg" http://192.168.0.101:7777/analyze | jq
```


## Features

- **Multi-Modal AI**: Text, vision, and multimodal analysis capabilities
- **Distributed Architecture**: Services can run on separate machines for load distribution
- **GPU Acceleration**: Optimized for NVIDIA GPU deployment with CUDA support
- **Windmill Integration**: Service APIs are designed to be orchestrated by Windmill flows and workers
- **Edge Computing Ready**: Low power consumption suitable for field deployment
- **Docker Support**: Containerized deployment options available
- **Comprehensive Logging**: Detailed logging and monitoring across all services

## Development

### Adding New Services

1. Create service directory with standard structure:
   ```
   service-name/
   ├── README.md
   ├── REST.py
   ├── .env.sample
   ├── requirements.txt
   ├── service-name.sh
   └── services/
       └── service-name-api.service
   ```

2. Follow established patterns for API integration and response formats
3. Add service to the port allocation table above
4. Include comprehensive documentation in service README

### Service Standards

- **Port Allocation**: Each service has a dedicated port (see table above)
- **Environment Configuration**: Use `.env` files for configuration
- **API Integration**: Expose a consistent HTTP interface for downstream consumers and orchestrators
- **Error Handling**: Comprehensive error handling with detailed logging
- **Documentation**: Complete README with setup instructions and API documentation

## Deployment

### Production Deployment

- **Systemd Services**: Service files provided for Linux deployment
- **Docker Containers**: Containerized deployment for scalability
- **Load Balancing**: Services can be load balanced across multiple instances
- **Monitoring**: Built-in health checks and status endpoints

### Hardware Requirements

- **Minimum**: 8GB RAM, modern CPU, 50GB storage
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM, 100GB+ SSD storage
- **Optimal**: 32GB+ RAM, NVIDIA RTX 3090/4090, NVMe SSD storage

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Code standards and best practices
- Testing requirements
- Data handling policies  
- Security considerations

## License

This project is part of the Window to the World technology suite and as such is licensed under the GPL. Additional licensing terms are available upon request. See individual service directories for specific licensing information.

## Support

For issues, questions, or contributions:

- Create an issue in the GitHub repository
- Follow the troubleshooting guides in individual service README files
- Check the [CONTRIBUTING.md](CONTRIBUTING.md) for common pitfalls and solutions
