FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including ffmpeg for moviepy
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone LAVIS repository
RUN git clone https://github.com/salesforce/LAVIS.git /app/LAVIS

# Set LAVIS as working directory for installation
WORKDIR /app/LAVIS

# Install PyTorch first (with CUDA 11.8 support) - using highest available version
RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install core ML dependencies with specific working versions and numpy from your working env
RUN pip install \
    "numpy==2.2.6" \
    transformers==4.25.0 \
    peft==0.4.0 \
    huggingface-hub==0.15.1 \
    accelerate==0.20.3 \
    tokenizers==0.13.3 \
    safetensors==0.6.2

# Install other critical dependencies with same numpy version as your working env
RUN pip install \
    pillow==11.3.0 \
    scipy \
    nltk==3.9.1 \
    Flask==3.1.2 \
    flask-cors==6.0.1 \
    python-dotenv==1.1.1 \
    requests==2.32.5 \
    einops==0.8.1 \
    omegaconf==2.3.0 \
    timm==0.4.12

# Install LAVIS dependencies manually (avoiding version conflicts)
RUN pip install \
    decord==0.6.0 \
    fairscale==0.4.4 \
    pycocoevalcap==1.2 \
    webdataset==1.0.2 \
    iopath==0.1.10 \
    open3d==0.19.0 \
    imageio==2.37.0 \
    imageio-ffmpeg==0.6.0 \
    moviepy==1.0.3 \
    easydict==1.9 \
    diffusers==0.16.0 \
    spacy==3.8.7 \
    h5py==3.14.0

# Install LAVIS without dependencies to avoid version conflicts
RUN pip install -e . --no-deps

# Install OpenCV with same version as your working environment
RUN pip install opencv-python-headless==4.12.0.88

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Create uploads directory
RUN mkdir -p /app/uploads

# Copy our working code files
COPY REST.py /app/
COPY blip2_analyzer.py /app/
COPY .env /app/

# Set final working directory
WORKDIR /app

# Expose port
EXPOSE 7777

# Set environment variables
ENV PYTHONPATH="/app/LAVIS"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Run the service
CMD ["python", "REST.py"]
