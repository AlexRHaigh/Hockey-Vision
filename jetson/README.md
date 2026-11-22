# Running and benchmarking on a Jetson Orin Nano

The Rust pipeline runs its five models with ONNX Runtime's TensorRT provider on the Jetson
(`--device tensorrt`), and `jetson/bench.sh` measures everything about a run: speed, latency
percentiles, power and energy, GPU / CPU use, memory, temperatures, each model on its own
(`trtexec`), where each operator ran, and whether the outputs still match a CPU FP32 run.

## 1. One-time setup (on the Jetson)

JetPack 6 (L4T r36). From the repo root:

```bash
sudo nvpmodel -m 0        # MAXN / MAXN SUPER: check `sudo nvpmodel -q` and /etc/nvpmodel.conf
sudo jetson_clocks        # clocks at max (resets on reboot)
jetson/setup.sh           # apt packages, Rust, ONNX Runtime GPU, builds hockey-vision --features tensorrt
```

`setup.sh` installs ONNX Runtime with the CUDA and TensorRT providers from the Jetson AI Lab pip
index into `.venv-ort/` and writes `jetson/ort.env`, which points `ORT_DYLIB_PATH` at its library.
The `ort` crate needs ONNX Runtime 1.22 or newer. If the wheel is older or has no C library,
build it from source instead (1-3 hours; add swap first):

```bash
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
jetson/build_onnxruntime.sh        # writes jetson/ort.env
jetson/setup.sh                    # builds hockey-vision against it
```

The ONNX models aren't in git: copy `CV_Models/Models/onnx/*.onnx` over from a machine that has
them (`python rust/export_onnx.py` makes them).

## 2. Running

```bash
source jetson/ort.env
rust/target/release/hockey-vision run videos/<clip>.mp4 --device tensorrt --precision fp16 \
    --teams SJS MTL --time --metrics
```

`--precision` is `fp32` (strict: TF32 off), `tf32` (TensorRT's default for FP32 models on the
Orin's Ampere GPU) or `fp16`. The first run of each model, shape and precision builds a TensorRT
engine (minutes); later runs load it from `CV_Models/Models/onnx/trt_cache/` (`HV_TRT_CACHE`).

`--metrics` writes, next to the clip's outputs, `run_metrics.json` (FPS, latency percentiles per
model and stage, load and engine-build times, boxes found, rink-fit rate, peak memory, CPU time)
and `frame_times.csv` (one row per frame). `HV_ORT_PROFILE=<dir>` writes ONNX Runtime profiles.

## 3. Benchmarking

```bash
jetson/bench.sh videos/<video>.mp4                                     # 5-minute segment from 3:00, fp32 and fp16
jetson/bench.sh videos/<video>.mp4 --duration full --precisions fp16   # the whole video
```

To keep it running if SSH drops, start it with `jetson/bench_detached.sh` instead (same
arguments). It runs `bench.sh` in a tmux session called `hv-bench`: type the sudo password, detach
with Ctrl-b then d, and later `tmux attach -t hv-bench` or `tail -f bench_results/*/bench.log`.

It writes `bench_results/<timestamp>/` (with the whole console output in `bench.log`) and ends with `summary.md` and `summary.json`
(`python3 jetson/summarize.py bench_results/<timestamp>` makes them again). See the comment at
the top of `bench.sh` for every option and file.

Speed only, reading every 2nd frame (no accuracy check, profile or trtexec):

```bash
jetson/bench_detached.sh videos/<video>.mp4 --precisions fp32 --quality-frames 0 --profile-frames 0 \
    --no-trtexec -- --frame-step 2
```

With `--frame-step N`, real time is the video's fps / N, and the summary's real-time factor is
measured against that.

Smaller, batched jersey crops (the number stage is the bottleneck) go the same way, e.g.
`-- --frame-step 2 --number-shape 320x192 --number-batch`. The trtexec step still times the number
model at 640x640.

## 4. Results so far

Jetson Orin Nano Super (MAXN SUPER, JetPack r36.4.7, ONNX Runtime 1.24.0), `bench.sh` defaults:
a 5-minute segment from 3:00 of the Canadiens vs Sharks game (Mar 3, 2026), 1080p at 59.94 fps,
every frame (17,983 frames). Results in `bench_results/20261121-210102` (not in git).

| | FP32 | FP16 |
|---|---|---|
| Frames needed per second (real time) | 59.94 | 59.94 |
| Detect FPS / steady FPS | 12.52 / 12.56 | 16.32 / 16.38 |
| End-to-end FPS | 12.49 | 16.28 |
| Real-time factor (1.0 = real time) | 0.209x | 0.272x |
| Frame interval p50 / p95 / p99 (max) | 82 / 123 / 152 ms (2198) | 54 / 98 / 320 ms (955) |
| Board power mean / max (idle 7.02 W) | 17.52 / 19.55 W | 13.46 / 15.62 W |
| Energy per frame | 1.403 J | 0.827 J |
| FPS per watt | 0.71 | 1.21 |
| Max temperature | 67.9 C | 62.9 C |
| Peak process memory | 3,650 MB | 3,529 MB |

FP32 on TensorRT matches the CPU FP32 run (1.04 FPS on the Jetson's CPU); FP16 changes about 5%
of jersey numbers. The jersey-number stage is the bottleneck (about 79 ms a frame at FP32).
With `--frame-step 2` real time is 29.97 frames a second, so at the same cost per frame FP32
would reach about 0.42x.
