## Deployment with Docker Compose (per service)

Use the per-service Compose files to deploy services independently or in groups.

### Examples

Run BLIP on a GPU host:
```bash
docker compose -f deploy/compose/blip.yaml up -d --build
```

Run YOLOv8 on a GPU host:
```bash
docker compose -f deploy/compose/yolov8.yaml up -d --build
```

Run CPU-only services:
```bash
docker compose -f deploy/compose/colors.yaml up -d --build
docker compose -f deploy/compose/metadata.yaml up -d --build
```

Run Ollama API with scale 3 (requires queue orchestrator upstream):
```bash
docker compose -f deploy/compose/ollama-api.yaml up -d --build --scale ollama-api=3
```

### Notes
- GPU access uses `device_requests` (NVIDIA runtime).
- For CPU-only runs, remove the `device_requests` section.

