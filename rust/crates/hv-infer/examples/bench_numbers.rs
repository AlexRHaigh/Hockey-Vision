//! What one jersey crop costs in the number stage, per input shape: preparing it (resize + pad +
//! planar float), the number model one crop at a time and in a batch, and NMS on its output.
//!
//!     bench_numbers <new_nums.onnx> <cpu|coreml|tensorrt-fp32|tensorrt-fp16> [threads] [batch] [runs] [HxW ...]
//!
//! Defaults: 4 threads, batch 5 (about a frame's crops), 50 runs, shapes 640x640 480x480 416x416
//! 320x320 384x224 320x192 256x160 224x128. The crop is 108x200 px, the median player crop at 1080p.
use std::time::Instant;

use hv_core::image::{letterbox_into, letterbox_params, Image};
use hv_core::yolo::nms;
use hv_infer::model::{Device, Model, Precision};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: bench_numbers <model.onnx> <cpu|coreml|tensorrt-fp32|tensorrt-fp16> [threads] [batch] [runs] [HxW ...]");
        std::process::exit(2);
    }
    let device = match a[2].as_str() {
        "cpu" => Device::Cpu,
        "coreml" => Device::CoreMl,
        "tensorrt-fp32" => Device::TensorRt(Precision::Fp32),
        "tensorrt-tf32" => Device::TensorRt(Precision::Tf32),
        "tensorrt-fp16" => Device::TensorRt(Precision::Fp16),
        d => panic!("unknown device {d}"),
    };
    if device == Device::TensorRt(Precision::Fp32) {
        std::env::set_var("NVIDIA_TF32_OVERRIDE", "0");
    }
    let threads: usize = a.get(3).map_or(4, |v| v.parse().unwrap());
    let batch: usize = a.get(4).map_or(5, |v| v.parse().unwrap());
    let runs: usize = a.get(5).map_or(50, |v| v.parse().unwrap());
    let shapes: Vec<(usize, usize)> = if a.len() > 6 {
        a[6..].iter().map(|s| { let (h, w) = s.split_once('x').unwrap(); (h.parse().unwrap(), w.parse().unwrap()) }).collect()
    } else {
        vec![(640, 640), (480, 480), (416, 416), (320, 320), (384, 224), (320, 192), (256, 160), (224, 128)]
    };

    // A 108x200 crop with some texture.
    let mut crop = Image::new(108, 200);
    for (i, v) in crop.data.iter_mut().enumerate() {
        *v = ((i * 37) % 251) as u8;
    }

    println!("device {}, {threads} threads, batch {batch}, {runs} runs, crop 108x200", device.name());
    println!("{:>9} {:>11} {:>12} {:>14} {:>10} {:>10}", "shape", "prep ms", "1 crop ms", "batch ms/crop", "nms ms", "pixels");
    for (h, w) in shapes {
        // Preparing a crop.
        let t = Instant::now();
        let mut input = vec![];
        for _ in 0..runs {
            input.clear();
            let lb = letterbox_params(crop.w, crop.h, (h, w), false, 32);
            letterbox_into(&crop, &lb, &mut input);
        }
        let prep = t.elapsed().as_secs_f64() * 1e3 / runs as f64;

        // One crop at a time (fixed-shape session on CoreML / TensorRT).
        let mut single = Model::load(a[1].as_ref(), device, threads).unwrap();
        single.allow_static(1, h, w);
        let t0 = Instant::now();
        for _ in 0..3 {
            single.run(input.clone(), 1, h, w).unwrap();
        }
        let warm1 = t0.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut last = (vec![], vec![]);
        for _ in 0..runs {
            last = single.run(input.clone(), 1, h, w).unwrap();
        }
        let one = t.elapsed().as_secs_f64() * 1e3 / runs as f64;

        // A batch (one engine for 1..16 on TensorRT, a fixed batch on CoreML).
        let mut batched = Model::load(a[1].as_ref(), device, threads).unwrap();
        batched.allow_batch(h, w, 16);
        let data: Vec<f32> = (0..batch).flat_map(|_| input.iter().copied()).collect();
        for _ in 0..3 {
            batched.run(data.clone(), batch, h, w).unwrap();
        }
        let t = Instant::now();
        for _ in 0..runs {
            batched.run(data.clone(), batch, h, w).unwrap();
        }
        let per = t.elapsed().as_secs_f64() * 1e3 / (runs * batch) as f64;

        // NMS on one crop's output.
        let (shape, out) = last;
        let t = Instant::now();
        for _ in 0..runs {
            std::hint::black_box(nms(&out, shape[1] - 4, shape[2], 0.25, None));
        }
        let nms_ms = t.elapsed().as_secs_f64() * 1e3 / runs as f64;
        println!(
            "{:>9} {prep:>11.3} {one:>12.2} {per:>14.2} {nms_ms:>10.3} {:>10}   (first runs {warm1:.1}s)",
            format!("{h}x{w}"),
            h * w
        );
    }
}
