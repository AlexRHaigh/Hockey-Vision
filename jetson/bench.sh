#!/usr/bin/env bash
# Benchmark hockey-vision on the Jetson and collect every metric we can, into
# bench_results/<timestamp>/, then summarise them (summary.md, summary.json).
#
#   jetson/bench.sh <video> [options] [-- extra hockey-vision options]
#
# Options:
#   --start S            where the benchmark segment starts in the video, seconds (default 180)
#   --duration S|full    its length in seconds (default 300), or "full" for the whole video
#   --precisions "..."   TensorRT precisions to run (default "fp32 fp16"; also tf32)
#   --quality-frames N   frames for the accuracy check against a CPU FP32 reference (default 1200;
#                        0 skips it; the CPU is slow, about 1-3 frames/s)
#   --profile-frames N   frames for the ONNX Runtime per-operator profile (default 300; 0 skips)
#   --no-trtexec         skip the per-model trtexec benchmarks
#   --nsys               also record an Nsight Systems timeline (600 frames)
#   --cooldown S         seconds of rest before each measured run (default 30)
#
# For each precision, in bench_results/<timestamp>/<precision>/:
#   cold/      a short first run: builds the TensorRT engines (or loads them from the cache)
#   run/       the measured run, with tegrastats.log (power, GPU/CPU use, memory, temperatures
#              every 100 ms), time.txt (GNU time: peak memory, CPU time), stdout/stderr, and the
#              pipeline's own run_metrics.json + frame_times.csv
#   profile/   ONNX Runtime per-operator profiles (and which provider ran each operator)
#   quality/   a run of --quality-frames frames, compared with the CPU FP32 reference (compare_*.txt)
#   trtexec/   each model alone: latency, throughput, GPU compute time, per-layer times
#   nsys/      the Nsight Systems timeline and its stats, with --nsys
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

VIDEO=${1:?usage: jetson/bench.sh <video> [options] [-- extra hockey-vision options]}
shift
START=180 DURATION=300 PRECISIONS="fp32 fp16" QUALITY=1200 PROFILE=300 TRTEXEC=1 NSYS=0 COOLDOWN=30
EXTRA=()
while [ $# -gt 0 ]; do
    case $1 in
        --start) START=$2; shift 2 ;;
        --duration) DURATION=$2; shift 2 ;;
        --precisions) PRECISIONS=$2; shift 2 ;;
        --quality-frames) QUALITY=$2; shift 2 ;;
        --profile-frames) PROFILE=$2; shift 2 ;;
        --no-trtexec) TRTEXEC=0; shift ;;
        --nsys) NSYS=1; shift ;;
        --cooldown) COOLDOWN=$2; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

[ -f jetson/ort.env ] || { echo "run jetson/setup.sh first" >&2; exit 1; }
# shellcheck disable=SC1091
source jetson/ort.env
B=$ROOT/rust/target/release/hockey-vision
[ -x "$B" ] || { echo "no $B: run jetson/setup.sh" >&2; exit 1; }
TRTEXEC_BIN=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
MODELS_DIR=$ROOT/CV_Models/Models/onnx
export HV_TRT_CACHE=${HV_TRT_CACHE:-$MODELS_DIR/trt_cache}

# This game's roster: team_a = San Jose, team_b = Montreal.
TEAMS=(--teams SJS MTL --roster rosters/2026-03-03_MTL_at_SJS.csv)

OUT=$ROOT/bench_results/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT/system"
echo "Results: $OUT"
sudo -v
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
SUDO_KEEPALIVE=$!
stop_tegrastats() { sudo tegrastats --stop >/dev/null 2>&1 || true; }
trap 'stop_tegrastats; kill $SUDO_KEEPALIVE 2>/dev/null || true' EXIT
trap 'echo "!! bench.sh failed at line $LINENO; see the *stderr.txt files under $OUT" >&2' ERR

echo "== system"
{
    cat /proc/device-tree/model 2>/dev/null | tr -d '\0'; echo
    cat /etc/nv_tegra_release 2>/dev/null
    uname -a
} > "$OUT/system/device.txt"
jetson_release > "$OUT/system/jetson_release.txt" 2>&1 || true
sudo nvpmodel -q > "$OUT/system/nvpmodel.txt" 2>&1 || true
sudo jetson_clocks --show > "$OUT/system/jetson_clocks.txt" 2>&1 || true
dpkg -l | grep -E "nvidia-jetpack|tensorrt|libnvinfer|cuda-toolkit|cudnn" > "$OUT/system/packages.txt" || true
"$TRTEXEC_BIN" --help 2>&1 | head -3 > "$OUT/system/trtexec_version.txt" || true
nvcc --version > "$OUT/system/cuda.txt" 2>&1 || true
echo "$ORT_DYLIB_PATH" > "$OUT/system/onnxruntime.txt"
git rev-parse HEAD > "$OUT/system/git_commit.txt"
git status --short > "$OUT/system/git_status.txt"
free -m > "$OUT/system/memory.txt"
nproc > "$OUT/system/nproc.txt"
grep -q "MAXN" "$OUT/system/nvpmodel.txt" || echo "!! power mode is not MAXN (see system/nvpmodel.txt): sudo nvpmodel -m 0 && sudo jetson_clocks"

echo "== segment"
if [ "$DURATION" = full ]; then
    CLIP=$VIDEO
else
    CLIP=$OUT/segment.mp4
    # Stream copy (no re-encode): starts at the keyframe at or before --start.
    ffmpeg -v error -y -ss "$START" -i "$VIDEO" -t "$DURATION" -an -c:v copy "$CLIP"
fi
echo "$VIDEO start=$START duration=$DURATION" > "$OUT/system/segment.txt"
ffprobe -v error -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration -of default "$CLIP" >> "$OUT/system/segment.txt"
STEM=$(basename "${CLIP%.*}")

echo "== idle power (20 s)"
stop_tegrastats
sudo tegrastats --interval 100 --logfile "$OUT/idle_tegrastats.log" --start
sleep 20
stop_tegrastats

for P in $PRECISIONS; do
    D=$OUT/$P
    mkdir -p "$D"
    echo "== $P: cold start (engine build or cache load)"
    "$B" detect "$CLIP" --device tensorrt --precision "$P" "${TEAMS[@]}" --max-frames 120 --metrics \
        --output "$D/cold" "${EXTRA[@]}" > "$D/cold.stdout.txt" 2> "$D/cold.stderr.txt"

    echo "== $P: measured run (cooling down ${COOLDOWN}s first)"
    sleep "$COOLDOWN"
    mkdir -p "$D/run"
    sudo tegrastats --interval 100 --logfile "$D/run/tegrastats.log" --start
    /usr/bin/time -v -o "$D/run/time.txt" \
        "$B" run "$CLIP" --device tensorrt --precision "$P" "${TEAMS[@]}" --time --metrics \
        --output "$D/run" "${EXTRA[@]}" > "$D/run/stdout.txt" 2> "$D/run/stderr.txt"
    stop_tegrastats
    tail -3 "$D/run/stdout.txt"

    if [ "$PROFILE" -gt 0 ]; then
        echo "== $P: ONNX Runtime profile ($PROFILE frames)"
        HV_ORT_PROFILE=$D/profile/ort "$B" detect "$CLIP" --device tensorrt --precision "$P" "${TEAMS[@]}" \
            --max-frames "$PROFILE" --metrics --output "$D/profile" "${EXTRA[@]}" > "$D/profile.stdout.txt" 2> "$D/profile.stderr.txt"
    fi

    if [ "$QUALITY" -gt 0 ]; then
        echo "== $P: quality run ($QUALITY frames)"
        "$B" run "$CLIP" --device tensorrt --precision "$P" "${TEAMS[@]}" --max-frames "$QUALITY" --metrics \
            --output "$D/quality" "${EXTRA[@]}" > "$D/quality.stdout.txt" 2> "$D/quality.stderr.txt"
    fi

    if [ "$TRTEXEC" = 1 ]; then
        echo "== $P: trtexec per model"
        mkdir -p "$D/trtexec"
        case $P in fp32) FLAGS=(--noTF32) ;; tf32) FLAGS=() ;; fp16) FLAGS=(--fp16) ;; esac
        for M in "$MODELS_DIR"/*.onnx; do
            N=$(basename "$M" .onnx)
            # 1080p frames are shrunk to 640x360 and padded to 640x384; jersey crops to 640x640.
            if [ "$N" = new_nums ]; then SHAPE=1x3x640x640; else SHAPE=1x3x384x640; fi
            "$TRTEXEC_BIN" --onnx="$M" --shapes=images:$SHAPE "${FLAGS[@]}" --memPoolSize=workspace:1024M \
                --warmUp=2000 --duration=20 --iterations=1000 --avgRuns=100 --separateProfileRun \
                --exportTimes="$D/trtexec/$N.times.json" --exportProfile="$D/trtexec/$N.layers.json" \
                > "$D/trtexec/$N.log" 2>&1 || echo "!! trtexec failed for $N (see $D/trtexec/$N.log)"
        done
    fi

    if [ "$NSYS" = 1 ] && command -v nsys >/dev/null; then
        echo "== $P: Nsight Systems (600 frames)"
        mkdir -p "$D/nsys"
        nsys profile --trace=cuda,nvtx,osrt,cudnn --force-overwrite=true -o "$D/nsys/timeline" \
            "$B" detect "$CLIP" --device tensorrt --precision "$P" "${TEAMS[@]}" --max-frames 600 \
            --output "$D/nsys/out" "${EXTRA[@]}" > "$D/nsys/stdout.txt" 2>&1 || true
        nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,osrt_sum \
            "$D/nsys/timeline.nsys-rep" > "$D/nsys/stats.txt" 2>&1 || true
    fi
done

if [ "$QUALITY" -gt 0 ]; then
    echo "== CPU FP32 reference ($QUALITY frames; slow)"
    "$B" run "$CLIP" --device cpu "${TEAMS[@]}" --max-frames "$QUALITY" --time --metrics \
        --output "$OUT/reference_cpu" "${EXTRA[@]}" > "$OUT/reference_cpu.stdout.txt" 2> "$OUT/reference_cpu.stderr.txt"
    R=$OUT/reference_cpu/$STEM
    for P in $PRECISIONS; do
        Q=$OUT/$P/quality/$STEM
        python3 rust/bench/compare_detections.py "$R/detections.json" "$Q/detections.json" > "$OUT/$P/compare_detections.txt" 2>&1 || true
        python3 rust/bench/compare_positions.py "$R/positions.json" "$Q/positions.json" > "$OUT/$P/compare_positions.txt" 2>&1 || true
        python3 rust/bench/compare_exports.py "$R/export" "$Q/export" > "$OUT/$P/compare_exports.txt" 2>&1 || true
    done
fi

echo "== summary"
python3 jetson/summarize.py "$OUT"
echo "Done: $OUT/summary.md"
