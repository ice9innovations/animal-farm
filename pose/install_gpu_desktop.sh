#!/bin/bash
# Compatibility alias for the default GPU-first Pose installer.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
"$SCRIPT_DIR/install.sh"
