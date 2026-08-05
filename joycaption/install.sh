#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
CURRENT_USER="$(id -un)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.sample" "$SCRIPT_DIR/.env"
fi

source "$SCRIPT_DIR/.env"

DEFAULT_REPO_URL="https://github.com/fpgaminer/joycaption.git"
REPO_URL="${JOYCAPTION_REPO_URL:-$DEFAULT_REPO_URL}"
REPO_DIR="$SCRIPT_DIR/joycaption-src"

if [[ "$REPO_URL" == /* || "$REPO_URL" == ./* || "$REPO_URL" == ../* ]]; then
    if [ ! -d "$REPO_URL/.git" ]; then
        echo "Configured local JoyCaption repo does not exist: $REPO_URL"
        echo "Falling back to https://github.com/fpgaminer/joycaption.git"
        REPO_URL="https://github.com/fpgaminer/joycaption.git"
    fi
fi

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi

rm -rf "$SCRIPT_DIR/venv"
python3.11 -m venv "$SCRIPT_DIR/venv"
source "$SCRIPT_DIR/venv/bin/activate"

pip install --upgrade pip
pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

chmod +x "$SCRIPT_DIR/run.sh" "$SCRIPT_DIR/rest.sh" "$SCRIPT_DIR/joycaption.sh"

mkdir -p "$SCRIPT_DIR/services"
cat > "$SCRIPT_DIR/services/joycaption-api.service" <<EOF
[Unit]
Description=JoyCaption Vision REST API
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=10
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/rest.sh

[Install]
WantedBy=multi-user.target
EOF

echo "JoyCaption installed. Edit $SCRIPT_DIR/.env if needed, then run:"
echo "  $SCRIPT_DIR/run.sh"
echo ""
echo "Generated systemd unit:"
echo "  $SCRIPT_DIR/services/joycaption-api.service"
