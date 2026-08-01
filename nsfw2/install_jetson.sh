#!/bin/bash
# Install NSFW2 for Nvidia Jetson Orin / JetPack 6 / CUDA 12.6.
#
# This intentionally installs TensorFlow from the Jetson AI Lab CUDA index
# instead of normal PyPI, then verifies that TensorFlow is a CUDA build.
set -eu

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV="$SCRIPT_DIR/venv"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
JETSON_INDEX="${JETSON_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
JETSON_TENSORFLOW_VERSION="${JETSON_TENSORFLOW_VERSION:-2.18.0}"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "Error: install_jetson.sh must run on the Jetson aarch64 host." >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN was not found. JetPack 6 should provide Python 3.10." >&2
    echo "Set PYTHON_BIN=/path/to/python3.10 if needed." >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"Error: JetPack 6 NSFW2 install expects Python 3.10; "
        f"got {sys.version_info.major}.{sys.version_info.minor}"
    )
PY

rm -rf "$VENV"
"$PYTHON_BIN" -m venv "$VENV"

if grep -q "include-system-site-packages = true" "$VENV/pyvenv.cfg"; then
    sed -i 's/include-system-site-packages = true/include-system-site-packages = false/' "$VENV/pyvenv.cfg"
fi

"$VENV/bin/python" -m pip install --upgrade pip

# Install the CUDA TensorFlow wheel from the Jetson index. PyPI is available
# only for dependencies; diagnose_tf_gpu.py below fails if TensorFlow is CPU-only.
"$VENV/bin/python" -m pip install --no-cache-dir \
    --index-url "$JETSON_INDEX" \
    --extra-index-url https://pypi.org/simple \
    "tensorflow==$JETSON_TENSORFLOW_VERSION"

"$VENV/bin/python" -m pip install --no-cache-dir -r "$SCRIPT_DIR/requirements-jetson.txt"

"$VENV/bin/python" -m pip check

"$VENV/bin/python" "$SCRIPT_DIR/diagnose_tf_gpu.py"

echo ""
echo "NSFW2 Jetson venv ready at $VENV"
echo "Start with: ./run.sh"
