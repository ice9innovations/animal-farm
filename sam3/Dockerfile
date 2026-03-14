FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Clone SAM3 from GitHub and install
RUN git clone https://github.com/facebookresearch/sam3.git /app/sam3
RUN pip install -e "/app/sam3[notebooks]"

RUN pip install \
    Flask==3.1.1 \
    flask-cors==6.0.1 \
    python-dotenv \
    requests \
    pillow \
    numpy

COPY REST.py /app/

# Checkpoint is large — mount from host:
#   -v /home/sd/sam3/checkpoints:/app/sam3/checkpoints:ro
# Set SAM3_CHECKPOINT=/app/sam3/checkpoints/sam3.pt in .env

ENV SAM3_PATH=/app/sam3
ENV PYTHONPATH=/app/sam3

EXPOSE 9779

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

CMD ["python", "REST.py"]
