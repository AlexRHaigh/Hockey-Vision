# Hockey-Vision in Rust

The Rust port of the pipeline, built one stage at a time. [ANALYSIS.md](ANALYSIS.md) explains
why, what each stage gains and the plan. The Python pipeline in the repo root is unchanged and
stays the reference: each Rust stage reads and writes the same files as the Python stage it
replaces, so they can be swapped and compared one at a time.

## What's here

| Stage | Python | Rust | Status |
|---|---|---|---|
| Rink fit: detections → rink positions | `homography.py` | `hockey-vision homography` | **Ported** (no optional videos). 22x faster on one core, 75-88x on four |
| Jersey numbers from digit boxes | `run_models.py` (`assemble_number`, `JerseyVotes`) | `hv_core::jersey` | Ported, for the Rust detector to come |
| Detection, tracking, decoding | `run_models.py`, `video_io.py` | — | Phase 2 |
| Tables and play events | `export_data.py`, `events.py` | — | Not planned yet (0.18 ms/frame in Python) |

```
rust/
├── Cargo.toml              workspace
├── ANALYSIS.md             the analysis: where the time goes, what Rust changes, the plan
├── crates/hv-core/         the library
│   └── src/
│       ├── rink.rs         rink keypoints in feet (rink.py)
│       ├── geometry.rs     cv2.findHomography / perspectiveTransform, OpenCV-exact
│       ├── homography.rs   keypoint matching, robust fit, tracker, projection (homography.py)
│       ├── pipeline.rs     the per-clip loop and positions.json / .csv, homographies.csv
│       ├── jersey.rs       assemble_number, JerseyVotes (run_models.py)
│       ├── detections.rs   detections.json
│       └── pyfmt.rs        Python's float repr and csv quoting, for diffable outputs
├── crates/hv-cli/          the `hockey-vision` binary
└── bench/                  parity and timing against the Python pipeline
```

## Build and run

Needs Rust 1.77 or newer ([rustup.rs](https://rustup.rs)).

```bash
cd rust
cargo build --release
cargo test --release
```

```bash
rust/target/release/hockey-vision homography outputs/<clip>/detections.json --frame-size 1920x1080
```

This writes `positions.json`, `positions.csv` and `homographies.csv` next to `detections.json`,
the same files as `python homography.py outputs/<clip>/detections.json <video>`, so
`export_data.py` runs on them unchanged. `--frame-size` is the source video's width x height
(`homography.py` reads it from the video). Other options: `--output DIR`, `--threads N`,
`--sequential` (one thread; same output), `--time`.

Differences from `homography.py`:

- No `--save-video` / `--debug` videos; use `homography.py` for those.
- `detections.json` must come from a `run_models.py` that records camera cuts (any recent one).
  For older files, `homography.py` decodes the video to find them.
- A singular fit counts as no fit. `homography.py` crashes on one instead (see ANALYSIS.md, section 8).

## Parity with Python

```bash
rust/bench/run_parity.sh 3000 0     # from the repo root: synthetic clip, both implementations, timed and compared
```

| Script | Does |
|---|---|
| `bench/synth_detections.py` | Synthetic `detections.json` from a simulated broadcast camera (no models or video needed) |
| `bench/run_python_homography.py` | `homography.py` with the singular-fit guard, so it finishes the clip |
| `bench/compare_positions.py` | Frame-by-frame diff of two `positions.json` files |
| `bench/dump_fits.py` + `cargo run --release --example dump_fits` | Each frame's raw fit from both, to find where they first differ |
| `bench/ultralytics_overhead.py` | How much of `run_models.py`'s frame time is Python around the models (`--device cuda` on the Jetson) |

On the synthetic clips, 99.7-99.8% of frames get the same fit, and on those the player positions
agree to 0.00000 ft at the 99th percentile. ANALYSIS.md, section 6, explains the remaining frames.
