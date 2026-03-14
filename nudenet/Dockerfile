FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    "nudenet>=3.4.0" \
    "onnxruntime-gpu>=1.23.0" \
    "flask>=2.0.0" \
    "flask-cors>=3.0.0" \
    "requests>=2.25.0" \
    "python-dotenv>=0.19.0" \
    "Pillow>=8.3.0"

COPY emoji_mappings.json /app/
COPY REST.py /app/

# NudeNet downloads its model to ~/.NudeNet on first run.
# Mount a host directory to cache it across container restarts:
#   -v /home/sd/.NudeNet:/root/.NudeNet
ENV NUDENET_MODEL_PATH=/root/.NudeNet

EXPOSE 7789

CMD ["python", "REST.py"]
