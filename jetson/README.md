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
