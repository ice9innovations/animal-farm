#!/bin/bash
# Build llama-server binary with CUDA support.
# Run once per pod before starting this service.
# qwen-cpp has its own copy of this script.
#
# Output: ${WORKSPACE_DIR:-/workspace}/llama-server/build/bin/llama-server
#
# Usage:
#   bash build_server.sh [CUDA_ARCH]
#
# CUDA_ARCH defaults to 80;86;89;90 (covers A100, RTX 3090/4090, A40, H100).
# Override for your specific GPU:
#   bash build_server.sh 86    # RTX 3090 / A40
#   bash build_server.sh 89    # RTX 4090 / L40
#   bash build_server.sh 90    # H100
set -e

if ! command -v cmake &>/dev/null; then
    echo "cmake not found — installing..."
    apt-get update -qq && apt-get install -y cmake
fi

CUDA_ARCH="${1:-80;86;89;90}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
BUILD_DIR="${LLAMA_SERVER_BUILD_DIR:-$WORKSPACE_DIR/llama-server}"
BINARY="$BUILD_DIR/build/bin/llama-server"

if [ -f "$BINARY" ]; then
    echo "llama-server already built at $BINARY — skipping."
    exit 0
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "Cloning llama.cpp to $BUILD_DIR..."
    git clone https://github.com/ggerganov/llama.cpp "$BUILD_DIR"
else
    echo "llama.cpp source already at $BUILD_DIR — skipping clone, building..."
fi

cd "$BUILD_DIR"

if [ -z "${CC:-}" ]; then
    if [ -x /usr/bin/gcc-11 ]; then
        CC=/usr/bin/gcc-11
    else
        CC="$(command -v gcc)"
    fi
fi
if [ -z "${CXX:-}" ]; then
    if [ -x /usr/bin/g++-11 ]; then
        CXX=/usr/bin/g++-11
    else
        CXX="$(command -v g++)"
    fi
fi
CMAKE_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER:-$CXX}"

echo "Using C compiler: $CC"
echo "Using C++ compiler: $CXX"
echo "Using CUDA host compiler: $CMAKE_CUDA_HOST_COMPILER"

# Conda can put a stub nvcc and wrong gcc on PATH — use resolved system compilers explicitly
CC="$CC" CXX="$CXX" \
    cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER="$CMAKE_CUDA_HOST_COMPILER" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs"

cmake --build build --config Release -j$(nproc) --target llama-server

echo ""
echo "Built: $BUILD_DIR/build/bin/llama-server"
echo "Set LLAMA_SERVER_BIN=$BUILD_DIR/build/bin/llama-server in .env"
