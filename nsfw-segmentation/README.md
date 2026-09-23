# NSFW Segmentation Service

Content-moderation API backed by
[`NSFW-API/NSFW_Segmentation`](https://huggingface.co/NSFW-API/NSFW_Segmentation).
It runs the breast, vagina, and penis segmentation heads and combines their
instances into one response containing moderation status, labels, bounding
boxes, polygons, confidence, and total image coverage.

The upstream checkpoint is an anatomical detector, not a complete policy
engine. Calibrate thresholds on your own traffic and combine its result with
other classifiers when your policy covers more than visible anatomy.

## Setup

### NVIDIA desktop/server

Requires Python 3.11 and a current NVIDIA driver. The installer creates the
venv, installs CUDA-enabled PyTorch, downloads all three selected checkpoints,
and generates `nsfw-segmentation.service`:

```bash
cd nsfw-segmentation
./install.sh
./run.sh
```

For a CPU-only desktop installation, run `DEVICE=cpu ./install.sh`. The CUDA
PyTorch build still supports CPU inference.

### NVIDIA Jetson

The Jetson installer targets JetPack 6 with Python 3.10. It deliberately uses
`--system-site-packages` so Torch and Torchvision come from JetPack rather than
incompatible desktop PyPI wheels:

```bash
cd nsfw-segmentation
./install_jetson.sh
./run.sh
```

Install NVIDIA's JetPack-compatible Torch and Torchvision packages first if
they are not already present. The installer fails early unless CUDA is visible.

Both installers write a complete runtime configuration to `.env`; pass
environment variables to customize those values. For example:

```bash
MODEL_SIZE=x MODEL_DIR=/mnt/models/nsfw-segmentation ./install.sh
```

The default is the three YOLO11s checkpoints. To use the more accurate,
substantially larger YOLO11x checkpoints, set `MODEL_SIZE=x` in `.env`, run
`./download_models.sh`, and restart the service. The downloader and runtime
both read the same `.env` configuration.

## API

The service listens on port `7806` by default.

```bash
curl -F "file=@image.jpg" http://localhost:7806/analyze
curl --data-binary @image.jpg -H 'Content-Type: image/jpeg' http://localhost:7806/analyze
curl 'http://localhost:7806/analyze?url=https://example.com/image.jpg'
curl 'http://localhost:7806/analyze?file=/data/image.jpg&include_masks=false'
```

Successful response (values abbreviated):

```json
{
  "service": "nsfw-segmentation",
  "status": "success",
  "nsfw": true,
  "score": 0.94,
  "labels": ["breast", "nipple"],
  "coverage_ratio": 0.123,
  "predictions": [
    {
      "label": "breast",
      "task": "breast",
      "confidence": 0.94,
      "bbox": {"x": 10, "y": 20, "width": 100, "height": 120},
      "area_ratio": 0.12,
      "polygon": [[10.0, 20.0], [110.0, 20.0], [100.0, 140.0]],
      "polygon_normalized": [[0.01, 0.02], [0.11, 0.02], [0.1, 0.14]]
    }
  ]
}
```

Set `include_masks=false` to omit polygons and reduce response size. This does
not change detection or the aggregate `coverage_ratio`.

Additional routes:

- `GET /health`
- `GET /classes`
- `GET,POST /v3/analyze` (alias of `/analyze`)
- `POST /` (upload compatibility alias of `/analyze`)

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `PORT` | `7806` | HTTP port |
| `PRIVATE` | `false` | Bind only to loopback when true |
| `MODEL_DIR` | `<service directory>/models` | Checkpoint directory |
| `MODEL_SIZE` | `s` | `s` or `x` checkpoint family |
| `CONFIDENCE_THRESHOLD` | `0.1` | Instance confidence threshold |
| `IOU_THRESHOLD` | `0.45` | Inference NMS IoU threshold |
| `IMAGE_SIZE` | `832` | YOLO inference resolution |
| `DEVICE` | auto | Ultralytics device (`cuda`, `cpu`, etc.) |
| `MAX_FILE_SIZE` | 32 MiB | Maximum upload/download size |
| `ALLOW_PRIVATE_URLS` | `false` | Permit URL requests to non-public IPs |

`BREAST_MODEL_PATH`, `VAGINA_MODEL_PATH`, and `PENIS_MODEL_PATH` can override
individual checkpoint paths. Remote URL analysis blocks private and loopback
targets by default to reduce SSRF risk.
