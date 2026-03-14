FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    Flask==2.3.3 \
    Flask-Cors==4.0.0 \
    "Pillow>=10.0.0" \
    "pyzbar>=0.1.9" \
    "python-dotenv>=1.0.0" \
    "requests>=2.31.0" \
    "Werkzeug>=3.0.0"

COPY REST.py /app/

EXPOSE 7801

CMD ["python", "REST.py"]
