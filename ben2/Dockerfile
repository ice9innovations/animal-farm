FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Clone BEN2 source (provides BEN2.py module)
RUN git clone https://github.com/PramaLLC/BEN2.git /app/BEN2

RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

RUN pip install \
    numpy==2.2.6 \
    pillow==11.3.0 \
    opencv-python-headless \
    timm \
    einops

RUN pip install \
    Flask==3.1.1 \
    flask-cors==6.0.1 \
    python-dotenv \
    requests

# Model weights are large — mount via volume at runtime
# Default path matches BEN2_MODEL_PATH in .env
RUN mkdir -p /app/models

COPY REST.py /app/
COPY ben2_analyzer.py /app/
COPY .env /app/

EXPOSE 7769

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
# BEN2 code is cloned above; override BEN2_CODE_DIR in .env if needed
ENV BEN2_CODE_DIR=/app/BEN2

CMD ["python", "REST.py"]
