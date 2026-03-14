FROM python:3.11-slim

WORKDIR /app

RUN pip install \
    flask==3.1.1 \
    flask-cors==6.0.1 \
    python-dotenv==1.1.1 \
    requests==2.32.4 \
    pillow==11.3.0 \
    nltk==3.9.1 \
    anthropic==0.49.0

# Local data files (auto-updated from GitHub at startup if AUTO_UPDATE=true)
COPY emoji_mappings.json /app/
COPY mwe.txt /app/
COPY REST.py /app/

# .env is NOT copied — it contains a live API key.
# Pass all vars via --env-file at runtime.

EXPOSE 7797

CMD ["python", "REST.py"]
