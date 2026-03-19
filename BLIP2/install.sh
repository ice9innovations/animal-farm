#!/bin/bash
# Install BLIP2 service: clones LAVIS and installs all dependencies into BLIP2/venv.
# Run once before first use. Requires Python 3.11.
#
# Usage:
#   bash install.sh
#
# After install, start the service with:
#   systemctl start blip2
#   (or: bash BLIP2.sh)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LAVIS_DIR="$(realpath "$SCRIPT_DIR/../LAVIS")"
SERVICE_NAME="blip2"
CURRENT_USER="$(whoami)"

# Clone LAVIS if not already present
if [ ! -d "$LAVIS_DIR" ]; then
    echo "Cloning LAVIS..."
    git clone https://github.com/salesforce/LAVIS.git "$LAVIS_DIR"
else
    echo "LAVIS already present at $LAVIS_DIR"
fi

python3.11 -m venv "$SCRIPT_DIR/venv"
source "$SCRIPT_DIR/venv/bin/activate"

pip install --upgrade pip

# PyTorch with CUDA 11.8 (works on CUDA 12.x hardware)
pip install \
    torch==2.7.1+cu118 \
    torchvision==0.22.1+cu118 \
    torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Core ML dependencies (pinned to match working Dockerfile)
pip install \
    "numpy==2.2.6" \
    transformers==4.25.0 \
    peft==0.4.0 \
    huggingface-hub==0.15.1 \
    accelerate==0.20.3 \
    tokenizers==0.13.3 \
    safetensors==0.6.2

# Service and LAVIS dependencies
pip install \
    pillow==11.3.0 \
    scipy \
    nltk==3.9.1 \
    Flask==3.1.2 \
    flask-cors==6.0.1 \
    python-dotenv==1.1.1 \
    requests==2.32.5 \
    einops==0.8.1 \
    omegaconf==2.3.0 \
    timm==0.4.12 \
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
    h5py==3.14.0 \
    opencv-python-headless==4.12.0.88

# Install LAVIS itself (no deps — already installed above)
pip install -e "$LAVIS_DIR" --no-deps

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm BLIP2 Caption Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
Environment=PYTHONPATH=$LAVIS_DIR
EnvironmentFile=$SCRIPT_DIR/.env
ExecStart=$SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/REST.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "Generated $SERVICE_FILE"

if [ "$(id -u)" = "0" ]; then
    cp "$SERVICE_FILE" /etc/systemd/system/
    systemctl daemon-reload
    echo "Service installed. Run: systemctl start $SERVICE_NAME"
else
    echo ""
    echo "To install the service, run:"
    echo "  sudo cp $SERVICE_FILE /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl start $SERVICE_NAME"
fi
