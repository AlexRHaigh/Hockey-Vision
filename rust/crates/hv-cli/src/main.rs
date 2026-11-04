//! `hockey-vision`: command-line front end for the Rust port (see rust/README.md).
//!
//!     hockey-vision homography outputs/<clip>/detections.json --frame-size 1920x1080
//!
//! writes positions.json, positions.csv and homographies.csv next to detections.json, like
//! `python homography.py outputs/<clip>/detections.json <video>` does without --save-video.
//! The frame size is the source video's (homography.py reads it from the video).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use hv_core::{detections, pipeline};

const USAGE: &str = "usage: hockey-vision homography <detections.json> --frame-size WxH [--output DIR] [--threads N] [--sequential] [--time]";

struct Args {
    detections: PathBuf,
    frame_size: (f64, f64),
    output: Option<PathBuf>,
    threads: Option<usize>,
    sequential: bool,
    time: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    match it.next().as_deref() {
        Some("homography") => {}
        Some("-h" | "--help") | None => return Err(USAGE.into()),
        Some(other) => return Err(format!("unknown command {other}\n{USAGE}")),
    }
    let (mut detections, mut frame_size, mut output, mut threads) = (None, None, None, None);
    let (mut sequential, mut time) = (false, false);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--frame-size" => {
                let v = it.next().ok_or("--frame-size needs WxH")?;
                let (w, h) = v
                    .split_once('x')
                    .ok_or("--frame-size takes WxH, e.g. 1920x1080")?;
                frame_size = Some((
                    w.parse().map_err(|_| "bad width")?,
                    h.parse().map_err(|_| "bad height")?,
                ));
            }
            "--output" => output = Some(PathBuf::from(it.next().ok_or("--output needs a folder")?)),
            "--threads" => {
                threads = Some(
                    it.next()
                        .ok_or("--threads needs N")?
                        .parse()
                        .map_err(|_| "bad --threads")?,
                )
            }
            "--sequential" => sequential = true,
            "--time" => time = true,
            "-h" | "--help" => return Err(USAGE.into()),
            _ if detections.is_none() => detections = Some(PathBuf::from(a)),
            _ => return Err(format!("unexpected argument {a}\n{USAGE}")),
        }
    }
    Ok(Args {
        detections: detections.ok_or(USAGE)?,
        frame_size: frame_size
            .ok_or("--frame-size is required (the source video's width x height)")?,
        output,
        threads,
        sequential,
        time,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .expect("thread pool");
    }
    let t0 = Instant::now();
    let frames = match detections::load(&args.detections) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let t_load = t0.elapsed();
    let t1 = Instant::now();
    let (positions, s) = match pipeline::run(&frames, args.frame_size, !args.sequential) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let t_fit = t1.elapsed();
    let out_dir = args.output.unwrap_or_else(|| {
        args.detections
            .parent()
            .map(PathBuf::from)
            .unwrap_or_default()
    });
    let t2 = Instant::now();
    if let Err(e) = std::fs::create_dir_all(&out_dir)
        .and_then(|_| pipeline::write_outputs(&positions, &out_dir))
    {
        eprintln!("writing {}: {e}", out_dir.display());
        return ExitCode::FAILURE;
    }
    let t_write = t2.elapsed();
    println!(
        "Rink fit on {}/{} frames (of the other {}: {} with clustered players, {} with implausible player size, rest too few rink markings); puck on the rink in {}",
        s.with_fit,
        s.frames,
        s.frames - s.with_fit,
        s.players_clustered,
        s.player_scale,
        s.with_puck
    );
    println!(
        "Wrote {}",
        ["positions.json", "positions.csv", "homographies.csv"]
            .map(|n| out_dir.join(n).display().to_string())
            .join(", ")
    );
    if args.time {
        let per_frame = t_fit.as_secs_f64() * 1e3 / s.frames.max(1) as f64;
        println!(
            "time: load {:.3}s, fit {:.3}s ({per_frame:.3} ms/frame, {} threads), write {:.3}s",
            t_load.as_secs_f64(),
            t_fit.as_secs_f64(),
            if args.sequential {
                1
            } else {
                rayon::current_num_threads()
            },
            t_write.as_secs_f64()
        );
    }
    ExitCode::SUCCESS
}
