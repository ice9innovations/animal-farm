#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
source "$SCRIPT_DIR/.env"

EXIFTOOL_DAEMON_SOCKET="/tmp/animal-farm-metadata-exiftool-${PORT}.sock"

echo "Waiting for exiftool_daemon to start..."
if [ -S "$EXIFTOOL_DAEMON_SOCKET" ]; then
    echo "exiftool_daemon already running."
else
    # Start exiftool_daemon in the background
    perl "$SCRIPT_DIR/exiftool_daemon.pl" "$EXIFTOOL_DAEMON_SOCKET" &

    DAEMON_PID=$!
    trap 'kill $DAEMON_PID 2>/dev/null' EXIT

    until [ -S "$EXIFTOOL_DAEMON_SOCKET" ]; do
        if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
            wait "$DAEMON_PID"
            exit $?
        fi
        sleep 0.1
    done
fi
echo "exiftool_daemon ready."

cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/venv/bin/python" REST.py
