FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FACE_BACKEND=auto \
    FACE_MODEL_PATH=/models/face/blaze.onnx \
    USE_GPU=true \
    REQUIRE_GPU=true \
    ONNX_PROVIDER_ORDER=cuda,tensorrt,cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    libpng-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    ffmpeg \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
ARG ORT_PACKAGE=""
RUN if [ -n "$ORT_PACKAGE" ]; then \
      pip3 uninstall -y onnxruntime onnxruntime-gpu && \
      pip3 install --no-cache-dir "$ORT_PACKAGE"; \
    fi

COPY . .

EXPOSE 7772

CMD ["python3", "REST.py"]
