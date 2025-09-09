FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
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
    ffmpeg \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Clone opennsfw2 repository for model and code
RUN git clone https://github.com/bhky/opennsfw2.git /app/opennsfw2_repo

# Install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Install opennsfw2 from the cloned repo
RUN pip3 install -e /app/opennsfw2_repo

COPY . .

EXPOSE 7774

CMD ["python3", "REST.py"]