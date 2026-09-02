# OCR Text Extraction Service

**Port**: 7775  
**Framework**: EasyOCR  
**Runtime**: CUDA PyTorch  
**Purpose**: GPU-backed OCR text extraction with emoji mapping  

## Overview

This service is a Paddle-free OCR replacement built on EasyOCR and keeps the existing Animal Farm OCR API shape: combined text, per-region bounding boxes, confidence scores, and emoji mappings for meaningful words.

GPU execution is the default: if CUDA PyTorch is unavailable, the service exits at startup unless `USE_GPU=false` is set in `.env` (see Configuration).

## Features

- Unified `/analyze` endpoint for URL, file path, and upload input
- Backward-compatible `/v2` and `/v3` routes
- Text regions with bounding boxes and confidence scores
- Emoji enrichment using local or auto-updated mappings
- No Paddle or PaddlePaddle dependency
- Startup failure when GPU OCR is required but unavailable

## Install

```bash
cd /home/sd/animal-farm/ocr
bash install.sh
```

`install.sh` is the only command needed on any platform. It detects Jetson, desktop/server NVIDIA GPU, Raspberry Pi/non-Jetson aarch64 CPU, or other CPU-only systems, installs the matching PyTorch build, installs EasyOCR and the rest of the dependencies, and generates the systemd service file in one run. It also recreates the venv automatically if it's missing, broken, or needs system-site packages.

- **Jetson**: hands off to `install_jetson.sh`, which uses JetPack's system PyTorch/torchvision (`--system-site-packages` venv, `--no-deps` EasyOCR install).
- **Desktop/server NVIDIA GPU**: calls `enable_gpu_desktop.sh` to install CUDA PyTorch from the PyTorch CUDA wheel index.
- **Raspberry Pi / non-Jetson aarch64 CPU**: creates a `--system-site-packages` venv and uses Debian/Raspberry Pi OS `python3-torch` and `python3-torchvision` packages. This avoids PyPI aarch64 Torch wheels that can pull CUDA packages and fail with `Bus error` on a Pi.
- **Other CPU-only**: installs CPU PyTorch from the PyTorch CPU wheel index.

On CPU-only installs, `install.sh` also sets `USE_GPU=false` in `.env` so the service starts in CPU mode.

If a Raspberry Pi already has a broken PyPI Torch install in `ocr/venv`, rerun `bash install.sh`. The installer recreates the venv with system-site packages and removes local PyPI/CUDA Torch packages that would shadow the system CPU packages. If apt cannot run automatically, install the system packages manually and rerun:

```bash
sudo apt-get update
sudo apt-get install -y python3-torch python3-torchvision
bash install.sh
```

`enable_gpu_desktop.sh` and `install_jetson.sh` remain runnable directly if you need to redo just the PyTorch step, but `install.sh` alone takes a fresh or partially broken install all the way to a runnable service.

## Configuration

Create `.env` from `.env.sample`:

```bash
PORT=7775
PRIVATE=false
TIMEOUT=30
AUTO_UPDATE=true
OCR_REQUIRE_GPU=true
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Preserved for service compatibility |
| `TIMEOUT` | Yes | - | Timeout for remote config downloads |
| `AUTO_UPDATE` | Yes | - | Refresh emoji/MWE config from GitHub on startup |
| `OCR_REQUIRE_GPU` | No | `true` | Exit at startup when CUDA PyTorch is unavailable (also settable as `REQUIRE_GPU` or `OCR_USE_CUDA`) |
| `USE_GPU` | No | `true` | Set to `false` to run on CPU, even when a GPU is present, and to skip the `OCR_REQUIRE_GPU` startup check |
| `MAX_FILE_SIZE` | No | `33554432` | Maximum upload/download size in bytes |

## API

### Health

```bash
curl "http://localhost:7775/health"
```

`/health` reports `gpu_required`, `gpu_enabled`, `device`, `torch_version`, and `cuda_device`.

## Benchmark

With the OCR service already running, benchmark the actual HTTP path:

```bash
cd /home/sd/animal-farm/ocr
./venv/bin/python benchmark.py --image /path/to/image.jpg --runs 20 --warmup 3 --timeout 10 --sla-ms 1000
```

For a synthetic local image:

```bash
./venv/bin/python benchmark.py --generate --generated-size 1280x720 --runs 20 --timeout 10 --sla-ms 1000
```

To sweep generated image sizes and find the point where the Pi crosses the 1s SLA:

```bash
./venv/bin/python benchmark.py --runs 5 --warmup 1 --timeout 10 --sla-ms 1000
```

The script posts the image to `/analyze`, measures wall-clock latency, validates terminal JSON, reports percentiles, counts timeouts, and exits non-zero if any measured request fails or misses the SLA. Keep `--timeout` above `--sla-ms` when measuring latency; using a 1s HTTP timeout only measures client cancellation and can leave unfinished OCR work running on the server.

### Analyze URL

```bash
curl "http://localhost:7775/analyze?url=https://example.com/image.jpg"
```

### Analyze File

```bash
curl "http://localhost:7775/analyze?file=/path/to/image.jpg"
```

### Upload File

```bash
curl -F "file=@/path/to/image.jpg" "http://localhost:7775/analyze"
```

Successful responses keep the existing schema:

```json
{
  "service": "ocr",
  "status": "success",
  "predictions": [
    {
      "text": "Detected text",
      "emoji": "💬",
      "has_text": true,
      "text_regions": [
        {
          "text": "Detected text",
          "confidence": 0.98,
          "bbox": {"x": 10, "y": 20, "width": 120, "height": 24}
        }
      ],
      "emoji_mappings": []
    }
  ],
  "metadata": {
    "processing_time": 0.123,
    "model_info": {
      "framework": "EasyOCR",
      "runtime": "PyTorch",
      "device": "cuda"
    }
  }
}
```

## Service Management

```bash
cd /home/sd/animal-farm/ocr
bash run.sh
```

For systemd installs generated by `install.sh`:

```bash
sudo systemctl start ocr
sudo systemctl status ocr
```
