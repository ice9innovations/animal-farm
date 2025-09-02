# 🐷 Animal Farm

**All animals are equal, but some animals are more equal than others**

Animal Farm is an edge-computing, distributed AI solution based on the technology developed in Window to the World. Using less than 100W of power, Animal Farm supports field deployments in low power or battery-only environments. Window to the World, the technology that Animal Farm is based on, is a new form of emergent AI that uses a novel consensus-based architecture and detailed data storage to create and map a complex system. It uses emergent intelligence to make decisions about any data the system can analyze. The result is an open system that is a powerful and generalizable AI. It is able to make decisions about any form of data that it has the models to analyze.

## Architecture

**From each service according to its capability, to each according to its needs**

Animal Farm consists of multiple specialized AI services that work together in a distributed ensemble. Each service runs independently and communicates through RESTful APIs, allowing for scalable deployment across multiple machines or containers.

## Services

| Service | Port | Description |
|---------|------|-------------|
| **BLIP** | 7777 | Image captioning |
| **CLIP** | 7778 | Image-text similarity analysis |
| **colors** | 7770 | Color analysis and palette extraction |
| **detectron2** | 7771 | Object detection and instance segmentation |
| **face** | 7772 | Face detection and analysis |
| **metadata** | 7781 | Image metadata extraction (EXIF, GPS, camera info) |
| **nsfw2** | 7774 | Content safety detection |
| **ocr** | 7775 | Optical character recognition |
| **ollama-api** | 7782 | Multi-modal large language model analysis |
| **rtdetr** | 7780 | Transformer-based object detection |
| **xception** | 7779 | ImageNet classification |
| **yolov8** | 7773 | Real-time object detection |

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

## Animal Farm Democracy: The Voting System

**Consensus is Truth, Evidence is Democracy, Algorithms are Liberation**

Animal Farm employs a sophisticated democratic voting system inspired by Orwell's famous quote: *"All animals are equal, but some animals are more equal than others."* 

### How Voting Works

**Democratic Foundation**: Every AI service gets exactly one vote per detection - no arbitrary service favoritism or weighting.

**Evidence-Based Consensus**: Services become "more equal than others" through verifiable evidence:

- **Spatial Consensus**: When multiple object detection services (YOLO, Detectron2, RT-DETR) agree on bounding box locations, they receive consensus bonuses equal to `detection_count - 1`
- **Content Consensus**: When semantic services (BLIP, Ollama) and classification services (CLIP, Inception) agree on the same emoji, they receive consensus bonuses equal to `total_agreeing_services - 1`
- **Instance Weighting**: Multiple detected instances vote proportionally - 3 people detected gets 3x the spatial votes of 1 person detected

**Specialist Authority**: Each specialist service is authoritative but not exclusive in their domain:
- **Face**: Authoritative for face detection, validates person emojis
- **OCR**: Authoritative for text reading, contributes to emoji discovery through text mining  
- **NSFW**: Authoritative for content safety, requires human context for validation

**Post-Processing Curation**: Clean +1/-1 adjustments ensure logical consistency:
- Face detection validates person detection (+1)
- Pose detection validates person detection (+1)
- NSFW detection without human context receives skepticism (-1)

This creates an intelligent democracy where consensus amplifies evidence rather than arbitrary algorithmic favoritism determining outcomes.

## API Integration

Services are designed to work together through a centralized API architecture:

- **Central API (port 8080)**: Provides emoji mappings, voting coordination, and system-wide configuration
- **Service APIs**: Each AI service exposes unified `/analyze` endpoints supporting URL, file, and POST inputs
- **Unified Response Format**: All services follow consistent JSON response schemas with metadata, predictions, and processing times
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
- **Animal Farm Democracy**: Intelligent consensus voting system where all services are equal, but some are more equal than others
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
