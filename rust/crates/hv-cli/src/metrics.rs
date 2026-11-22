//! `--metrics`: everything measurable about one clip's run, written next to its outputs as
//! `run_metrics.json` (summary statistics) and `frame_times.csv` (one row per frame).
//!
//! Latency statistics leave out the first `WARMUP_FRAMES` frames (TensorRT builds or loads its
//! engines on the first frame, and caches warm up); the first frame is reported on its own.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hv_infer::detect::{FrameStat, LoadTime, Timings, FULL_MODELS};
use hv_infer::video::VideoInfo;
use serde_json::{json, Map, Value};

pub const WARMUP_FRAMES: usize = 30;

/// n, mean, std, min, p50, p90, p95, p99, max of the finite values.
fn stats(values: impl IntoIterator<Item = f64>) -> Value {
    let mut v: Vec<f64> = values.into_iter().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return Value::Null;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    let q = |p: f64| v[(((v.len() - 1) as f64) * p).round() as usize];
    let r = |x: f64| (x * 1000.0).round() / 1000.0;
    json!({
        "n": v.len(), "mean": r(mean), "std": r(std), "min": r(v[0]),
        "p50": r(q(0.5)), "p90": r(q(0.9)), "p95": r(q(0.95)), "p99": r(q(0.99)), "max": r(v[v.len() - 1]),
    })
}

fn secs(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e4).round() / 1e4
}

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// A "Key:   123 kB" line of /proc/self/status, in MB.
fn proc_status_mb(status: &str, key: &str) -> Option<f64> {
    let line = status.lines().find(|l| l.starts_with(key))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some((kb / 1024.0 * 10.0).round() / 10.0)
}

/// Peak and current memory, CPU time and context switches of this process (Linux only).
pub fn process() -> Value {
    let Some(status) = read("/proc/self/status") else { return Value::Null };
    // /proc/self/stat: fields 14 and 15 (after the ")" of the name) are user and system time in ticks.
    let cpu = read("/proc/self/stat").and_then(|s| {
        let rest: Vec<&str> = s.rsplit_once(')')?.1.split_whitespace().collect();
        let tick = 100.0; // USER_HZ, 100 on Linux / Jetson
        Some((rest.get(11)?.parse::<f64>().ok()? / tick, rest.get(12)?.parse::<f64>().ok()? / tick))
    });
    let ctx = |k: &str| status.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok());
    json!({
        "peak_rss_mb": proc_status_mb(&status, "VmHWM:"),
        "rss_mb": proc_status_mb(&status, "VmRSS:"),
        "peak_virtual_mb": proc_status_mb(&status, "VmPeak:"),
        "user_cpu_s": cpu.map(|c| c.0),
        "system_cpu_s": cpu.map(|c| c.1),
        "threads": ctx("Threads:"),
        "voluntary_context_switches": ctx("voluntary_ctxt_switches:"),
        "involuntary_context_switches": ctx("nonvoluntary_ctxt_switches:"),
    })
}

/// The machine: OS, CPU count, memory, and on a Jetson its model and L4T release.
pub fn host() -> Value {
    let first_line = |p: &str| read(p).map(|s| s.trim_matches(char::from(0)).trim().lines().next().unwrap_or("").to_string());
    let mem_total = read("/proc/meminfo").and_then(|s| proc_status_mb(&s, "MemTotal:"));
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "hostname": first_line("/etc/hostname"),
        "cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "mem_total_mb": mem_total,
        "device_model": first_line("/proc/device-tree/model"),
        "l4t_release": first_line("/etc/nv_tegra_release"),
        "kernel": first_line("/proc/version"),
    })
}

fn git_commit() -> Option<String> {
    let out = std::process::Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub struct Stage {
    pub name: &'static str,
    pub time: Duration,
}

/// Everything about one clip's run, as JSON.
#[allow(clippy::too_many_arguments)]
pub fn build(
    settings: &Value,
    video_path: &Path,
    info: &VideoInfo,
    t: &Timings,
    loads: &[LoadTime],
    load_wall: Duration,
    later_stages: &[Stage],
    fit: Option<&hv_core::pipeline::Summary>,
    profiles: &[String],
) -> Value {
    let f = &t.per_frame;
    let warm: &[FrameStat] = if f.len() > WARMUP_FRAMES * 2 { &f[WARMUP_FRAMES..] } else { f };
    let frames = t.frames.max(1) as f64;
    let wall = t.total.as_secs_f64();
    let fps = t.frames as f64 / wall.max(1e-9);

    // Steady state: frames finished per second after the warm-up.
    let skipped = f.len() - warm.len();
    let steady = match (warm.first(), warm.last()) {
        // From the last warm-up frame's finish to the last frame's: warm.len() frames.
        (Some(_), Some(last)) if skipped > 0 => Some(warm.len() as f64 / (last.done_s - f[skipped - 1].done_s).max(1e-9)),
        (Some(first), Some(last)) if warm.len() >= 2 => Some((warm.len() - 1) as f64 / (last.done_s - first.done_s).max(1e-9)),
        _ => None,
    };
    let intervals: Vec<f64> = f.windows(2).skip(WARMUP_FRAMES.min(f.len().saturating_sub(2))).map(|w| (w[1].done_s - w[0].done_s) * 1e3).collect();

    let mut models = Map::new();
    for (i, name) in FULL_MODELS.iter().enumerate() {
        let runs: Vec<f64> = f.iter().map(|s| s.models_ms[i]).filter(|x| x.is_finite()).collect();
        if runs.is_empty() {
            continue;
        }
        models.insert(
            name.to_string(),
            json!({
                "frames_run": runs.len(),
                "first_ms": runs[0],
                "ms": stats(warm.iter().map(|s| s.models_ms[i])),
                "total_s": secs(*t.models.get(name).unwrap_or(&Duration::ZERO)),
                "boxes_per_frame": stats(f.iter().filter(|s| s.models_ms[i].is_finite()).map(|s| s.boxes[i] as f64)),
                "frames_with_boxes": f.iter().filter(|s| s.models_ms[i].is_finite() && s.boxes[i] > 0).count(),
            }),
        );
    }
    let crops: usize = f.iter().map(|s| s.number_crops).sum();
    let number_ms: f64 = f.iter().map(|s| s.number_model_ms).sum();
    if crops > 0 {
        models.insert(
            "number".into(),
            json!({
                "crops": crops,
                "crops_per_frame": stats(f.iter().map(|s| s.number_crops as f64)),
                "ms_per_crop": (number_ms / crops as f64 * 1000.0).round() / 1000.0,
                "ms_per_frame": stats(warm.iter().map(|s| s.number_model_ms)),
                "digits_found": f.iter().map(|s| s.boxes[4]).sum::<usize>(),
            }),
        );
    }

    let stage_list: Vec<Value> = later_stages
        .iter()
        .map(|s| json!({"stage": s.name, "s": secs(s.time), "ms_per_frame": (s.time.as_secs_f64() * 1e6 / frames).round() / 1000.0}))
        .collect();
    let end_to_end = t.total + later_stages.iter().map(|s| s.time).sum::<Duration>();

    json!({
        "schema": 1,
        "created_unix_s": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "git_commit": git_commit(),
        "command": std::env::args().collect::<Vec<_>>(),
        "env": {
            "NVIDIA_TF32_OVERRIDE": std::env::var("NVIDIA_TF32_OVERRIDE").ok(),
            "ORT_DYLIB_PATH": std::env::var("ORT_DYLIB_PATH").ok(),
            "HV_TRT_CACHE": std::env::var("HV_TRT_CACHE").ok(),
            "HV_ORT_PROFILE": std::env::var("HV_ORT_PROFILE").ok(),
        },
        "onnxruntime": hv_infer::model::ort_build_info(),
        "host": host(),
        "settings": settings,
        "video": {
            "path": video_path.display().to_string(),
            "width": info.width, "height": info.height, "fps": info.fps,
            "frames_in_file": info.frames, "pix_fmt": info.pix_fmt,
            "duration_s": if info.fps > 0.0 { t.frames as f64 / info.fps } else { 0.0 },
        },
        "load": {
            "wall_s": secs(load_wall),
            "models": loads.iter().map(|l| json!({
                "model": l.model,
                "dynamic_session_s": secs(l.dynamic),
                // With TensorRT: building the engine (first run) or loading it from the cache.
                "fixed_shape_sessions": l.fixed.iter().map(|((n, h, w), d)| json!({"shape": format!("{n}x3x{h}x{w}"), "s": secs(*d)})).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        },
        "detect": {
            "frames": t.frames,
            "wall_s": secs(t.total),
            "fps": (fps * 100.0).round() / 100.0,
            "ms_per_frame": (wall * 1e6 / frames).round() / 1000.0,
            "real_time_factor": if info.fps > 0.0 { (fps / info.fps * 1000.0).round() / 1000.0 } else { 0.0 },
            "steady_fps": steady.map(|v| (v * 100.0).round() / 100.0),
            "warmup_frames_excluded": f.len() - warm.len(),
            "first_frame_s": f.first().map(|s| s.done_s),
            // Time between consecutive frames finishing: the throughput's jitter.
            "frame_interval_ms": stats(intervals),
            "frames_cut": f.iter().filter(|s| s.cut).count(),
            "frames_rink_run": f.iter().filter(|s| s.rink_run).count(),
            // Per-frame stage times, after the warm-up. With parallel stages these overlap, so
            // they add up to more than the frame interval.
            "stages_ms": {
                "decode_wait": stats(warm.iter().map(|s| s.decode_wait_ms)),
                "resize": stats(warm.iter().map(|s| s.resize_ms)),
                "tracking": stats(warm.iter().map(|s| s.tracking_ms)),
                "numbers": stats(warm.iter().map(|s| s.numbers_ms)),
                "write": stats(warm.iter().map(|s| s.write_ms)),
            },
            "stage_totals_s": {
                "decode_wait": secs(t.decode_wait), "resize": secs(t.resize), "tracking": secs(t.tracking),
                "numbers": secs(t.numbers), "write": secs(t.write),
            },
            "models": models,
        },
        "later_stages": stage_list,
        "rink_fit": fit.map(|s| json!({
            "frames": s.frames, "frames_with_fit": s.with_fit,
            "fit_rate": if s.frames > 0 { (s.with_fit as f64 / s.frames as f64 * 1e4).round() / 1e4 } else { 0.0 },
            "frames_with_puck_on_rink": s.with_puck,
        })),
        "end_to_end": {
            "s": secs(end_to_end),
            "fps": ((t.frames as f64 / end_to_end.as_secs_f64().max(1e-9)) * 100.0).round() / 100.0,
        },
        "process": process(),
        "ort_profiles": profiles,
    })
}

/// run_metrics.json and frame_times.csv in `dir`.
pub fn write(dir: &Path, metrics: &Value, t: &Timings) -> Result<(), String> {
    let p = dir.join("run_metrics.json");
    std::fs::write(&p, serde_json::to_string_pretty(metrics).unwrap() + "\n").map_err(|e| format!("{}: {e}", p.display()))?;
    let p = dir.join("frame_times.csv");
    let file = std::fs::File::create(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut w = std::io::BufWriter::new(file);
    let mut head = String::from("frame,done_s,cut,rink_run,decode_wait_ms,resize_ms");
    for m in FULL_MODELS {
        let _ = write!(head, ",{m}_ms");
    }
    head.push_str(",tracking_ms,numbers_ms,number_model_ms,number_crops,write_ms");
    for m in FULL_MODELS {
        let _ = write!(head, ",{m}_boxes");
    }
    head.push_str(",digits\n");
    w.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    let num = |x: f64| if x.is_finite() { format!("{x:.3}") } else { String::new() };
    for s in &t.per_frame {
        let mut row = format!("{},{:.4},{},{},{},{}", s.frame, s.done_s, s.cut as u8, s.rink_run as u8, num(s.decode_wait_ms), num(s.resize_ms));
        for m in s.models_ms {
            let _ = write!(row, ",{}", num(m));
        }
        let _ = write!(row, ",{},{},{},{},{}", num(s.tracking_ms), num(s.numbers_ms), num(s.number_model_ms), s.number_crops, num(s.write_ms));
        for b in &s.boxes {
            let _ = write!(row, ",{b}");
        }
        row.push('\n');
        w.write_all(row.as_bytes()).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    eprintln!("  metrics: {}/run_metrics.json, frame_times.csv", dir.display());
    Ok(())
}
