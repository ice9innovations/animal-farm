FROM python:3.11-slim

WORKDIR /app

RUN pip install \
    flask==3.1.1 \
    flask-cors==6.0.1 \
    python-dotenv==1.1.1 \
    requests==2.32.4 \
    "anthropic>=0.50.0"

COPY REST.py /app/
COPY .env /app/

EXPOSE 7799

CMD ["python", "REST.py"]
