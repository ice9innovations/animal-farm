FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/cuda/bin:$PATH

# Install Python 3.10 and system dependencies including CUDA runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    gcc \
    curl \
    libjpeg62 \
    zlib1g-dev \
    libpng-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libopencv-dev \
    libcublas11 \
    libcurand10 \
    libcufft10 \
    libcusparse11 \
    libcusolver11 \
    libnppc11 \
    libnppial11 \
    libnppicc11 \
    libnppidei11 \
    libnppif11 \
    libnppig11 \
    libnppim11 \
    libnppist11 \
    libnppisu11 \
    libnppitc11 \
    libnpps11 \
    libnvjpeg11 \
    libnvtoolsext1 \
 && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /usr/src/app

COPY requirements.txt ./

# Verify CUDA and cuDNN installation
RUN nvidia-smi && \
    python3 -c "import paddle; print('PaddlePaddle version:', paddle.__version__)" || echo "PaddlePaddle not yet installed" && \
    ls -la /usr/local/cuda/lib64/ | grep cudnn || echo "cuDNN libraries location check"

RUN pip3 install --no-cache-dir -r requirements.txt

# Verify PaddlePaddle GPU installation
RUN python3 -c "import paddle; print('PaddlePaddle version:', paddle.__version__); print('CUDA available:', paddle.device.is_compiled_with_cuda())"

COPY . .

EXPOSE 7775

CMD ["python3", "REST.py"]
