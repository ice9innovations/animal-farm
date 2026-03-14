FROM python:3.11-slim

WORKDIR /app

# llama-server (the C++ binary) is NOT included in this image.
# It runs on the host as llama-cpp-server.service and is reached via
# LLAMA_SERVER_HOST in .env. Use --network=host so 127.0.0.1 resolves
# to the host loopback where llama-server is listening.

RUN pip install \
    flask==3.1.1 \
    flask-cors==6.0.1 \
    python-dotenv==1.1.1 \
    requests==2.32.4 \
    pillow==11.3.0 \
    nltk==3.9.1

COPY emoji_mappings.json /app/
COPY mwe.txt /app/
COPY REST.py /app/

EXPOSE 7782

CMD ["python", "REST.py"]
