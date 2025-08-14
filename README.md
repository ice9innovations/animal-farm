# Animal Farm

"All animals are equal, but some animals are more equal than others"

Animal Farm is an edge-computing, distributed AI solution based on the technology developed in Window to the World. Using less than 100W of power, Animal Farm supports field deployments in low power or battery-only environments. Window to the World, the technology that Animal Farm is based on, is a new form of emergent AI that uses a novel consensus-based architecture and detailed data storage to create and map a complex system. It uses emergent intelligence to make decisions about any data the system can analyze. The result is an open system that is a powerful and generalizable AI. It is able to make decisions about any form of data that it has the models to analyze.

## Architecture

Animal Farm consists of multiple specialized AI services that work together in a distributed ensemble. Each service runs independently and communicates through RESTful APIs, allowing for scalable deployment across multiple machines or containers.

## Services

| Service | Port | Status | Description |
|---------|------|--------|-------------|
| **BLIP** | 7777 | ✅ Active | Image captioning (Bootstrapping Language-Image Pre-training) |
| **CLIP** | 7778 | ✅ Active | Image-text similarity analysis |
| **colors** | 7770 | ✅ Active | Color analysis and palette extraction |
| **detectron2** | 7771 | ✅ Active | Object detection and instance segmentation (Facebook Detectron2) |
| **face** | 7772 | ✅ Active | Face detection and analysis |
| **inception_v3** | 7779 | ✅ Active | ImageNet classification (Google Inception v3) |
| **metadata** | 7781 | ✅ Active | Image metadata extraction (EXIF, GPS, camera info) |
| **nsfw** | 7774 | ✅ Active | Content safety detection (Bumble private-detector) |
| **ocr** | 7775 | ✅ Active | Optical character recognition (PaddleOCR) |
| **ollama-api** | 7782 | ✅ Active | Large language model analysis (Ollama/LLaMA integration) |
| **rtdetr** | 7780 | ✅ Active | Real-time object detection (RT-DETR transformer) |
| **yolov8** | 7773 | ✅ Active | Real-time object detection (Ultralytics YOLOv8) |

## Central API

| Service | Port | Description |
|---------|------|-------------|
| **api** | 8080 | Central coordination API, emoji mappings, and voting system |

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

## API Integration

Services are designed to work together through a centralized API architecture:

- **Central API (port 8080)**: Provides emoji mappings, voting coordination, and system-wide configuration
- **Service APIs**: Each AI service exposes REST endpoints for analysis
- **Unified Response Format**: All services follow consistent JSON response schemas for easy integration

### Example API Usage

#### V3 Unified Endpoints (Recommended)

```bash
# Local file path
curl "http://192.168.0.101:7777/v3/analyze?file=/path/to/image/file" | jq

# Image URL
curl "http://192.168.0.101:7777/v3/analyze?url=https://example.com/image.jpg" | jq
```

#### Legacy V2 Endpoints (Backward Compatibility)

```bash
# Object detection
curl "http://localhost:7771/v2/analyze?image_url=https://example.com/image.jpg"

# Image captioning  
curl "http://localhost:7777/v2/analyze?image_url=https://example.com/image.jpg"

# NSFW detection
curl "http://localhost:7774/?url=https://example.com/image.jpg"
```

## Features

- **Multi-Modal AI**: Text, vision, and multimodal analysis capabilities
- **Distributed Architecture**: Services can run on separate machines for load distribution
- **GPU Acceleration**: Optimized for NVIDIA GPU deployment with CUDA support
- **Consensus Voting**: Multiple AI models vote on decisions for improved accuracy
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
- **API Integration**: Connect to central API (port 8080) for emoji mappings
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

This project is part of the Window to the World technology suite and as such is licensed under the GPL. Additional icensing terms are available upon request. See individual service directories for specific licensing information.

## Support

For issues, questions, or contributions:

- Create an issue in the GitHub repository
- Follow the troubleshooting guides in individual service README files
- Check the [CONTRIBUTING.md](CONTRIBUTING.md) for common pitfalls and solutions
