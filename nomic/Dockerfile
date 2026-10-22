FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY REST.py .
ENV PORT=7808 PRIVATE=false TEXT_MODEL=nomic-ai/nomic-embed-text-v1.5 VISION_MODEL=nomic-ai/nomic-embed-vision-v1.5
EXPOSE 7808
CMD ["python", "REST.py"]
