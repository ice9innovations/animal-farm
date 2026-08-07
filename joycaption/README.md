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
MODEL_DIR=
DEVICE=auto
PRECISION=8bit
BNB_SKIP_MODULES=vision_tower,multi_modal_projector
DISABLE_CUDNN=auto
CUDA_ALLOW_TF32=true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCH_VERSION=
TORCHVISION_VERSION=
TORCH_INDEX_URL=
JOYCAPTION_CACHE_ROOT=
JOYCAPTION_VENV_DIR=
PIP_CACHE_DIR=
TMPDIR=
MAX_NEW_TOKENS=96
TEMPERATURE=0.6
TOP_P=0.9
GREEDY=false
```

`PRECISION=8bit` is the default because the full bf16 model needs roughly 17GB VRAM. `PRECISION=4bit` is also available for a smaller footprint. Both quantized modes require CUDA and `bitsandbytes`, and leave the vision tower and multimodal projector unquantized to match the smoke-test script. CPU mode is intentionally limited to `DEVICE=cpu` and `PRECISION=fp32` because JoyCaption is large.

`BNB_SKIP_MODULES` controls which modules bitsandbytes leaves unquantized. The default matches the JoyCaption smoke test. On very tight GPUs, try `BNB_SKIP_MODULES=multi_modal_projector` or `BNB_SKIP_MODULES=` with `PRECISION=4bit` to reduce VRAM further.

The install script chooses PyTorch wheels from the detected GPU. Compute capability 12.x GPUs, such as RTX 5090, use CUDA 12.8 wheels. Older GPUs default to CUDA 12.1 wheels for driver compatibility. Override `TORCH_VERSION`, `TORCHVISION_VERSION`, and `TORCH_INDEX_URL` in `.env` only when a machine needs a specific build.

`DISABLE_CUDNN=auto` disables cuDNN on compute capability 12.x GPUs. This avoids cuDNN workspace allocation failures in the SigLIP vision tower on RTX 5090 while leaving cuDNN enabled on older GPUs. Set `DISABLE_CUDNN=false` to force cuDNN back on.

`MODEL_DIR` maps to `HF_HOME`. If `MODEL_DIR` is blank, `install.sh` uses `/mnt/models/workspace/huggingface` when `/mnt/models/workspace` exists and is writable, otherwise it falls back to a service-local `.cache/huggingface`. The same cache root is used for pip cache and temporary files, which keeps large CUDA wheel installs off small root volumes. Set `JOYCAPTION_VENV_DIR` if the virtualenv itself should live on the model/workspace drive; `install.sh` will symlink `./venv` to it so runtime scripts keep working.

## Run

```bash
./run.sh
```

## Download model cache

The service normally downloads the Hugging Face model on first startup. If the machine
was restored after a crash, is configured with offline Hugging Face flags, or you want
to prefill the cache before starting CUDA inference, run:

```bash
./download_model.sh
```

This uses `MODEL_ID` and `MODEL_DIR` from `.env`, sets `HF_HOME` to `MODEL_DIR`, and
temporarily clears `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` for the download.

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
