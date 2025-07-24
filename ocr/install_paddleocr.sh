#!/bin/bash

# Proper PaddleOCR installation with Python venv

set -e

echo "Creating isolated Python virtual environment for PaddleOCR..."

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies in isolated environment
echo "Installing PaddleOCR dependencies in isolated environment..."
pip install paddleocr==2.7.3
pip install paddlepaddle-gpu==2.5.2
pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install python-dotenv==1.0.0
pip install Pillow==10.0.0
pip install numpy==1.24.3
pip install requests==2.31.0
pip install nltk==3.8.1

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "PaddleOCR environment created successfully!"
echo "Virtual environment: ./venv"
echo ""
echo "To activate: source venv/bin/activate"
echo "To test: python -c 'from paddleocr import PaddleOCR; print(\"PaddleOCR imported successfully\")'"