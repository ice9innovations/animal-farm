#!/bin/bash
# Build llama-server binary with CUDA support.
# Run once per pod before starting llama-cpp or qwen-cpp services.
# The binary is shared by both services.
#
# Output: ${WORKSPACE_DIR:-/workspace}/llama-server/build/bin/llama-server
#
# Usage:
#   bash build_server.sh [CUDA_ARCH]
#
# CUDA_ARCH defaults to the local GPU's compute capability.
# Override if auto-detection is unavailable:
#   bash build_server.sh 86    # RTX 3090 / A40
#   bash build_server.sh 89    # RTX 4090 / L40
#   bash build_server.sh 90    # H100
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
ENV_FILE="$SCRIPT_DIR/.env"

if ! command -v cmake &>/dev/null; then
    echo "cmake not found — installing..."
    apt-get update -qq && apt-get install -y cmake
fi

detect_cuda_arch() {
    if ! command -v nvidia-smi &>/dev/null; then
        return 1
    fi

    local compute_caps
    compute_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null)" || return 1

    printf '%s\n' "$compute_caps" \
        | awk 'NF { gsub(/\./, "", $1); print $1 }' \
        | sort -u \
        | paste -sd';' -
}

REQUESTED_CUDA_ARCH="${1:-}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
BUILD_DIR="${LLAMA_SERVER_BUILD_DIR:-$WORKSPACE_DIR/llama-server}"
BINARY="$BUILD_DIR/build/bin/llama-server"

set_env_value() {
    local key="$1"
    local value="$2"

    if [ -f "$ENV_FILE" ] && grep -q "^${key}=" "$ENV_FILE"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
        printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    fi
}

if [ -f "$BINARY" ]; then
    echo "llama-server already built at $BINARY — skipping."
    set_env_value LLAMA_SERVER_BIN "$BINARY"
    exit 0
fi

if [ -n "$REQUESTED_CUDA_ARCH" ]; then
    CUDA_ARCH="$REQUESTED_CUDA_ARCH"
else
    if ! CUDA_ARCH="$(detect_cuda_arch)"; then
        CUDA_ARCH=""
    fi
    if [ -z "$CUDA_ARCH" ]; then
        echo "Could not detect CUDA architecture. Pass it explicitly, for example: bash build_server.sh 89"
        exit 1
    fi
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "Cloning llama.cpp to $BUILD_DIR..."
    git clone https://github.com/ggerganov/llama.cpp "$BUILD_DIR"
else
    echo "llama.cpp source already at $BUILD_DIR — skipping clone, building..."
fi

cd "$BUILD_DIR"

is_conda_toolchain() {
    if [ -n "${CONDA_PREFIX:-}" ]; then
        case "$1" in
            "$CONDA_PREFIX"/*) return 0 ;;
        esac
    fi
    case "$1" in
        *conda*) return 0 ;;
        *) return 1 ;;
    esac
}

pick_system_toolchain() {
    if [ -x /usr/bin/gcc-12 ] && [ -x /usr/bin/g++-12 ]; then
        CC=/usr/bin/gcc-12
        CXX=/usr/bin/g++-12
    elif [ -x /usr/bin/gcc-11 ] && [ -x /usr/bin/g++-11 ]; then
        CC=/usr/bin/gcc-11
        CXX=/usr/bin/g++-11
    else
        CC="$(command -v gcc)"
        CXX="$(command -v g++)"
    fi
}

if [ -z "${CC:-}" ] || [ -z "${CXX:-}" ] || is_conda_toolchain "$CC" || is_conda_toolchain "$CXX"; then
    pick_system_toolchain
fi
if [ -n "${CMAKE_CUDA_HOST_COMPILER:-}" ] && is_conda_toolchain "$CMAKE_CUDA_HOST_COMPILER"; then
    unset CMAKE_CUDA_HOST_COMPILER
fi
CMAKE_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER:-$CXX}"
CUDA_COMPILER="${CUDA_COMPILER:-${CUDACXX:-}}"
if [ -z "$CUDA_COMPILER" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    else
        CUDA_COMPILER="$(command -v nvcc || true)"
    fi
fi
if [ -z "$CUDA_COMPILER" ]; then
    echo "CUDA nvcc compiler not found. Install the CUDA Toolkit, then rerun."
    exit 1
fi
CUDA_ROOT="$(dirname "$(dirname "$CUDA_COMPILER")")"

echo "Using C compiler: $CC"
echo "Using C++ compiler: $CXX"
echo "Using CUDA host compiler: $CMAKE_CUDA_HOST_COMPILER"
echo "Using CUDA compiler: $CUDA_COMPILER"
echo "Using CUDA architecture(s): $CUDA_ARCH"

if [ -f build/CMakeCache.txt ] && grep -q "conda" build/CMakeCache.txt; then
    echo "Existing CMake cache references conda compilers — removing stale build directory."
    rm -rf build
fi

# Conda can put a stub nvcc, wrong gcc, and incompatible sysroot flags on PATH.
# Use resolved system compilers explicitly and clear conda compile/link flags.
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
CC="$CC" CXX="$CXX" CUDAHOSTCXX="$CMAKE_CUDA_HOST_COMPILER" \
    cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DCMAKE_CUDA_HOST_COMPILER="$CMAKE_CUDA_HOST_COMPILER" \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$CUDA_ROOT/lib64/stubs" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,$CUDA_ROOT/lib64/stubs"

cmake --build build --config Release -j$(nproc) --target llama-server

echo ""
echo "Built: $BUILD_DIR/build/bin/llama-server"
set_env_value LLAMA_SERVER_BIN "$BUILD_DIR/build/bin/llama-server"
echo "Set LLAMA_SERVER_BIN=$BUILD_DIR/build/bin/llama-server in .env"
