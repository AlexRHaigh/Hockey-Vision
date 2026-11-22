#!/usr/bin/env bash
# One-time setup of a Jetson (JetPack 6, Orin Nano) for the Rust pipeline with TensorRT and for
# jetson/bench.sh. Safe to run again. From the repo root:  jetson/setup.sh
#
# What it does:
#   1. apt packages: ffmpeg, GNU time, build tools, JetPack's CUDA / cuDNN / TensorRT / trtexec /
#      Nsight Systems (nvidia-jetpack), and jetson-stats (jtop, jetson_release)
#   2. Rust (rustup), if cargo isn't there
#   3. ONNX Runtime with the CUDA and TensorRT providers (1.22 or newer, which the `ort` crate
#      needs), from the Jetson AI Lab pip index into .venv-ort/, and writes jetson/ort.env pointing
#      ORT_DYLIB_PATH at its library. If that doesn't work, use jetson/build_onnxruntime.sh.
#   4. builds rust/target/release/hockey-vision with --features tensorrt
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

# JetPack 6.x (L4T r36) ships CUDA 12.6; change if `nvcc --version` says otherwise.
ORT_INDEX=${ORT_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}

echo "== 1. packages"
sudo apt-get update
sudo apt-get install -y ffmpeg time git curl build-essential pkg-config libssl-dev python3-pip python3-venv binutils
if ! dpkg -s nvidia-jetpack >/dev/null 2>&1; then
    sudo apt-get install -y nvidia-jetpack
fi
command -v jtop >/dev/null 2>&1 || sudo pip3 install -U jetson-stats

echo "== 2. rust"
if ! command -v cargo >/dev/null 2>&1; then
    [ -f "$HOME/.cargo/env" ] || curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustc --version

echo "== 3. onnxruntime (CUDA + TensorRT)"
write_env() { # $1 = the ONNX Runtime library
    local lib=$1 dir
    dir=$(dirname "$lib")
    cat > jetson/ort.env <<EOF
# Written by jetson/setup.sh: the ONNX Runtime that hockey-vision --device tensorrt loads.
export ORT_DYLIB_PATH="$lib"
export LD_LIBRARY_PATH="$dir:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PATH="/usr/local/cuda/bin:/usr/src/tensorrt/bin:\$PATH"
EOF
    echo "wrote jetson/ort.env -> $lib"
}

if [ -f jetson/ort.env ] && grep -q ORT_DYLIB_PATH jetson/ort.env && [ -f "$(. jetson/ort.env; echo "$ORT_DYLIB_PATH")" ]; then
    echo "jetson/ort.env already set up: $(grep ORT_DYLIB_PATH jetson/ort.env)"
else
    python3 -m venv .venv-ort
    .venv-ort/bin/pip install -U pip
    .venv-ort/bin/pip install -U onnxruntime-gpu --index-url "$ORT_INDEX" --extra-index-url https://pypi.org/simple
    .venv-ort/bin/python - <<'EOF'
import onnxruntime as o
print("onnxruntime", o.__version__, o.get_available_providers())
assert "TensorrtExecutionProvider" in o.get_available_providers(), "this onnxruntime has no TensorRT provider"
minor = int(o.__version__.split(".")[1])
assert minor >= 22, f"onnxruntime {o.__version__} is too old: the ort crate needs 1.22 or newer (use jetson/build_onnxruntime.sh)"
EOF
    CAPI=$(.venv-ort/bin/python -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))")
    LIB=$(ls "$CAPI"/libonnxruntime.so* 2>/dev/null | head -1 || true)
    if [ -z "$LIB" ]; then
        # Some wheels have no separate libonnxruntime.so; the Python module exports the C API itself.
        for f in "$CAPI"/onnxruntime_pybind11_state*.so; do
            if nm -D "$f" 2>/dev/null | grep -q " T OrtGetApiBase"; then LIB=$f; fi
        done
    fi
    if [ -z "$LIB" ]; then
        echo "No ONNX Runtime C library in the wheel ($CAPI). Build it instead: jetson/build_onnxruntime.sh" >&2
        exit 1
    fi
    write_env "$LIB"
fi
# shellcheck disable=SC1091
source jetson/ort.env

echo "== 4. build hockey-vision (TensorRT)"
(cd rust && cargo build --release --features tensorrt)
ls -la rust/target/release/hockey-vision

echo "== checks"
ls CV_Models/Models/onnx/*.onnx >/dev/null 2>&1 || echo "!! CV_Models/Models/onnx/*.onnx missing: copy them from your Mac (see jetson/README.md)"
command -v trtexec >/dev/null || ls /usr/src/tensorrt/bin/trtexec >/dev/null || echo "!! trtexec not found"
sudo nvpmodel -q || true
echo "Setup done. Before benchmarking: sudo nvpmodel -m 0 (MAXN / MAXN SUPER), sudo jetson_clocks"
