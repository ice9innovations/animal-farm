# JoyCaption Vision Service

**Port**: 7797  
**Framework**: Hugging Face Transformers + Flask  
**Model**: `fancyfeast/llama-joycaption-beta-one-hf-llava`  
**Purpose**: Local JoyCaption image captioning behind the Animal Farm REST API shape

## Overview

This service keeps JoyCaption loaded in one Flask process and exposes `/analyze` and `/v3/analyze` endpoints compatible with the other Animal Farm vision wrappers. It also clones JoyCaption into `joycaption-src/` beside this wrapper during install, so no local destination path is required.

`qwen-cpp` remains the lighter llama.cpp VLM path. This wrapper is for running the native JoyCaption HF model directly.

## Install

```bash
cd ~/animal-farm/joycaption
bash install.sh
```

Edit `.env` after installation if you need a different port, model cache path, precision, or prompt.

## Configuration

```bash
PORT=7797
PRIVATE=False
MODEL_ID=fancyfeast/llama-joycaption-beta-one-hf-llava
MODEL_DIR=${HOME}/.cache/huggingface
DEVICE=auto
PRECISION=8bit
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
MAX_NEW_TOKENS=256
TEMPERATURE=0.6
TOP_P=0.9
GREEDY=false
```

`PRECISION=8bit` is the default because the full bf16 model needs roughly 17GB VRAM. `PRECISION=4bit` is also available for a smaller footprint. Both quantized modes require CUDA and `bitsandbytes`, and leave the vision tower and multimodal projector unquantized to match the smoke-test script. CPU mode is intentionally limited to `DEVICE=cpu` and `PRECISION=fp32` because JoyCaption is large.

The install script uses PyTorch CUDA 12.1 wheels by default through `TORCH_INDEX_URL`. That matches systems whose NVIDIA driver reports CUDA 12.2 support; default PyPI Torch wheels may require a newer driver.

`MODEL_DIR` maps to `HF_HOME`. The default uses the normal shared Hugging Face cache at `${HOME}/.cache/huggingface`, so offline mode can reuse models already downloaded by smoke tests or other services. If you point `MODEL_DIR` at a service-local directory, run once with `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` to populate that cache before enabling offline mode.

## Run

```bash
./run.sh
```

Systemd:

```bash
./install.sh
sudo cp services/joycaption-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now joycaption-api
```

## API

```bash
GET /health
GET /analyze?url=<image_url>
GET /analyze?file=<local_path>
POST /analyze multipart file=@image.jpg
POST /analyze JSON {"image_base64": "..."}
```

Optional request parameters: `prompt`, `system_prompt`, `max_new_tokens`, `temperature`, `top_p`, and `greedy`.

Example:

```bash
curl -F "file=@/path/to/image.jpg" \
  -F "prompt=Write a concise factual caption for this image." \
  http://127.0.0.1:7797/analyze
```

Response:

```json
{
  "service": "joycaption",
  "status": "success",
  "predictions": [
    {
      "text": "A brown dog sits on a red couch in a living room."
    }
  ],
  "metadata": {
    "processing_time": 3.219,
    "model_info": {
      "framework": "transformers/llava",
      "model": "fancyfeast/llama-joycaption-beta-one-hf-llava"
    }
  }
}
```
