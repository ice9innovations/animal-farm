#!/bin/bash
# Install nudenet Flask API dependencies into nudenet/venv.
# Run once before first use. Requires Python 3.11.
#
# Usage:
#   bash install.sh
#
# After install, start the service with:
#   bash run.sh  (RunPod)
#   systemctl start nudenet-api  (systemd)
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SERVICE_NAME="nudenet-api"
SERVICE_USER="${SUDO_USER:-$(id -un)}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

VENV="$SCRIPT_DIR/venv"

rm -rf "$VENV"
if ! python3.11 -m venv "$VENV"; then
    echo "Error: failed to create virtualenv at $VENV." >&2
    echo "Install Python 3.11 with venv support, then rerun this script." >&2
    exit 1
fi

if [ ! -x "$VENV/bin/pip" ]; then
    echo "Error: virtualenv was created without pip at $VENV." >&2
    exit 1
fi

export TMPDIR="${TMPDIR:-$WORKSPACE_DIR/tmp}"
mkdir -p "$TMPDIR"

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

# Generate systemd service file
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Animal Farm NudeNet Detection Service
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$SCRIPT_DIR
EnvironmentFile=$SCRIPT_DIR/.env
ExecStart=$SCRIPT_DIR/run.sh
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
