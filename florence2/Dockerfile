FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# cu128 confirmed working for florence2 (see requirements.txt)
RUN pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
    transformers==4.56.1 \
    timm==1.0.25 \
    einops==0.8.2 \
    Pillow \
    nltk

RUN pip install \
    flask==3.1.3 \
    flask-cors==6.0.2 \
    python-dotenv \
    requests

COPY REST.py /app/
COPY florence2_analyzer.py /app/
COPY emoji_mappings.json /app/
COPY mwe.txt /app/

# HuggingFace model cache — mount a host volume here to avoid re-downloading
# the ~1.5GB Florence-2 model on every container start:
#   -v /home/sd/.cache/huggingface:/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 7803

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

CMD ["python", "REST.py"]
