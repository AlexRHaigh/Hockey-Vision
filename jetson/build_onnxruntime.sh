#!/usr/bin/env bash
# Fallback for jetson/setup.sh: build ONNX Runtime with CUDA + TensorRT from source on the Jetson
# and point jetson/ort.env at it. Takes 1-3 hours on an Orin Nano and needs ~8 GB of swap
# (see jetson/README.md). From the repo root:  jetson/build_onnxruntime.sh [version, default 1.22.0]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
VERSION=${1:-1.22.0}
SRC=${ORT_SRC:-$HOME/onnxruntime-$VERSION}

python3 -m venv .venv-build && .venv-build/bin/pip install -U pip "cmake>=3.28" numpy packaging wheel
export PATH="$ROOT/.venv-build/bin:/usr/local/cuda/bin:$PATH"

[ -d "$SRC" ] || git clone --recursive --depth 1 --branch "v$VERSION" https://github.com/microsoft/onnxruntime "$SRC"
cd "$SRC"
# Orin = compute capability 8.7. Two jobs keep the 8 GB from running out.
./build.sh --config Release --update --build --parallel 2 --build_shared_lib --skip_tests \
    --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
    --tensorrt_home /usr/lib/aarch64-linux-gnu \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 onnxruntime_BUILD_UNIT_TESTS=OFF

LIB=$(ls "$SRC"/build/Linux/Release/libonnxruntime.so.* | head -1)
cd "$ROOT"
cat > jetson/ort.env <<EOF
# Written by jetson/build_onnxruntime.sh: the ONNX Runtime that hockey-vision --device tensorrt loads.
export ORT_DYLIB_PATH="$LIB"
export LD_LIBRARY_PATH="$(dirname "$LIB"):/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PATH="/usr/local/cuda/bin:/usr/src/tensorrt/bin:\$PATH"
EOF
echo "wrote jetson/ort.env -> $LIB; now run jetson/setup.sh again to build hockey-vision"
