//! A YOLO model exported to ONNX (`yolo export format=onnx dynamic=True`), run with ONNX Runtime.
//!
//! On the CPU one session takes any input shape. With CoreML (Apple GPU / Neural Engine) or
//! TensorRT (NVIDIA Jetson) a model is much faster with fixed shapes, so each shape the pipeline
//! uses all the time (`allow_static`) gets its own session with the dynamic dimensions pinned; any
//! other shape runs on the dynamic session (the CPU, or CUDA with TensorRT).
//!
//! TensorRT engines are built on a session's first run and cached in `HV_TRT_CACHE` (default:
//! `trt_cache/` next to the model), so only the first run on a Jetson pays for the build.
//! `HV_ORT_PROFILE=<dir>` makes every session write an ONNX Runtime profile (per-operator times and
//! the execution provider each operator ran on) into that directory; see `end_profiling`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;

/// Numeric precision of the TensorRT engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Strict FP32: TensorRT's TF32 is turned off (NVIDIA_TF32_OVERRIDE=0, set by the CLI).
    Fp32,
    /// FP32 weights, with TensorRT free to use TF32 tensor cores (TensorRT's default).
    Tf32,
    Fp16,
}

impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Tf32 => "tf32",
            Precision::Fp16 => "fp16",
        }
    }
}

/// Which hardware runs the models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// Apple's CoreML (GPU / Neural Engine); build with --features coreml.
    CoreMl,
    /// NVIDIA TensorRT, with CUDA for shapes without an engine; build with --features tensorrt.
    TensorRt(Precision),
}

impl Device {
    /// Whether fixed-shape sessions are used (and faster) on this device.
    pub fn uses_static(self) -> bool {
        !matches!(self, Device::Cpu)
    }

    pub fn name(self) -> String {
        match self {
            Device::Cpu => "cpu".into(),
            Device::CoreMl => "coreml".into(),
            Device::TensorRt(p) => format!("tensorrt-{}", p.name()),
        }
    }
}

pub struct Model {
    path: PathBuf,
    device: Device,
    threads: usize,
    /// Takes any input shape: the CPU, or CUDA with TensorRT.
    dynamic: Session,
    static_sessions: HashMap<(usize, usize, usize), Session>,
    static_shapes: Vec<(usize, usize, usize)>,
    pub names: Vec<String>,
    pub stride: usize,
    input: String,
    /// Time to create the dynamic session.
    pub load_time: Duration,
    /// Time to create each fixed-shape session (with TensorRT, mostly building or loading the engine).
    pub static_load_times: Vec<((usize, usize, usize), Duration)>,
}

/// "{0: 'goalie_a', 1: 'goalie_b'}" (the Python dict Ultralytics writes) -> class names by index.
fn parse_names(s: &str) -> Vec<String> {
    let mut out: Vec<(usize, String)> = vec![];
    for part in s.trim().trim_start_matches('{').trim_end_matches('}').split(", ") {
        let Some((k, v)) = part.split_once(": ") else { continue };
        let v = v.trim();
        let v = v.strip_prefix(['\'', '"']).and_then(|v| v.strip_suffix(['\'', '"'])).unwrap_or(v);
        if let Ok(k) = k.trim().parse() {
            out.push((k, v.to_string()));
        }
    }
    out.sort();
    out.into_iter().map(|(_, v)| v).collect()
}

fn stem(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// A session builder, profiling into $HV_ORT_PROFILE/<tag> when that is set.
fn builder(threads: usize, profile_tag: &str) -> ort::Result<SessionBuilder> {
    let b = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?.with_intra_threads(threads)?;
    match std::env::var_os("HV_ORT_PROFILE") {
        Some(dir) if !dir.is_empty() => {
            let dir = PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            b.with_profiling(dir.join(profile_tag))
        }
        _ => Ok(b),
    }
}

#[cfg(feature = "tensorrt")]
fn cuda_ep() -> ort::execution_providers::ExecutionProviderDispatch {
    use ort::execution_providers::CUDAExecutionProvider;
    // No TF32 in the CUDA fallback either, so FP32 means FP32.
    CUDAExecutionProvider::default().with_device_id(0).with_tf32(false).build().error_on_failure()
}

impl Model {
    pub fn load(path: &Path, device: Device, threads: usize) -> Result<Self, String> {
        let err = |e: ort::Error| format!("{}: {e}", path.display());
        #[cfg(not(feature = "coreml"))]
        if device == Device::CoreMl {
            return Err("this build has no CoreML support; rebuild with --features coreml".into());
        }
        #[cfg(not(feature = "tensorrt"))]
        if matches!(device, Device::TensorRt(_)) {
            return Err("this build has no TensorRT support; rebuild with --features tensorrt (on the Jetson)".into());
        }
        let t0 = Instant::now();
        let tag = format!("{}_dynamic", stem(path));
        let b = builder(threads, &tag).map_err(err)?;
        #[cfg(feature = "tensorrt")]
        let b = if matches!(device, Device::TensorRt(_)) { b.with_execution_providers([cuda_ep()]).map_err(err)? } else { b };
        let dynamic = b.commit_from_file(path).map_err(err)?;
        let load_time = t0.elapsed();
        let (names, stride) = {
            let meta = dynamic.metadata().map_err(err)?;
            let names = meta.custom("names").map_err(err)?.map(|s| parse_names(&s)).unwrap_or_default();
            let stride = meta.custom("stride").map_err(err)?.and_then(|s| s.trim().parse().ok()).unwrap_or(32);
            (names, stride)
        };
        if names.is_empty() {
            return Err(format!("{}: no class names in the model's metadata (export it with Ultralytics)", path.display()));
        }
        let input = dynamic.inputs[0].name.clone();
        Ok(Model {
            path: path.to_path_buf(),
            device,
            threads,
            dynamic,
            static_sessions: HashMap::new(),
            static_shapes: vec![],
            names,
            stride,
            input,
            load_time,
            static_load_times: vec![],
        })
    }

    /// With CoreML or TensorRT, run inputs of (n, h, w) on a fixed-shape session.
    pub fn allow_static(&mut self, n: usize, h: usize, w: usize) {
        if self.device.uses_static() && !self.static_shapes.contains(&(n, h, w)) {
            self.static_shapes.push((n, h, w));
        }
    }

    pub fn is_static(&self, n: usize, h: usize, w: usize) -> bool {
        self.static_shapes.contains(&(n, h, w))
    }

    fn static_session(&mut self, key: (usize, usize, usize)) -> Result<&mut Session, String> {
        if !self.static_sessions.contains_key(&key) {
            let t0 = Instant::now();
            let (n, h, w) = key;
            let tag = format!("{}_{n}x{h}x{w}", stem(&self.path));
            let b = builder(self.threads, &tag)
                .and_then(|b| b.with_dimension_override("batch", n as i64))
                .and_then(|b| b.with_dimension_override("height", h as i64))
                .and_then(|b| b.with_dimension_override("width", w as i64))
                .map_err(|e| format!("{}: {e}", self.path.display()))?;
            let s = self.static_providers(b, &tag)?.commit_from_file(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
            self.static_load_times.push((key, t0.elapsed()));
            self.static_sessions.insert(key, s);
        }
        Ok(self.static_sessions.get_mut(&key).unwrap())
    }

    #[allow(unused_variables)]
    fn static_providers(&self, b: SessionBuilder, tag: &str) -> Result<SessionBuilder, String> {
        let err = |e: ort::Error| format!("{}: {e}", self.path.display());
        match self.device {
            #[cfg(feature = "coreml")]
            Device::CoreMl => {
                use ort::execution_providers::coreml::{CoreMLComputeUnits, CoreMLModelFormat};
                use ort::execution_providers::CoreMLExecutionProvider;
                let units = match std::env::var("HV_COREML_UNITS").as_deref() {
                    Ok("gpu") => CoreMLComputeUnits::CPUAndGPU,
                    Ok("ane") => CoreMLComputeUnits::CPUAndNeuralEngine,
                    Ok("cpu") => CoreMLComputeUnits::CPUOnly,
                    _ => CoreMLComputeUnits::All,
                };
                let ep = CoreMLExecutionProvider::default().with_model_format(CoreMLModelFormat::MLProgram).with_compute_units(units);
                b.with_execution_providers([ep.build().error_on_failure()]).map_err(err)
            }
            #[cfg(feature = "tensorrt")]
            Device::TensorRt(precision) => {
                use ort::execution_providers::TensorRTExecutionProvider;
                let cache = match std::env::var_os("HV_TRT_CACHE") {
                    Some(d) if !d.is_empty() => PathBuf::from(d),
                    _ => self.path.parent().unwrap_or(Path::new(".")).join("trt_cache"),
                };
                std::fs::create_dir_all(&cache).map_err(|e| format!("{}: {e}", cache.display()))?;
                let cache = cache.display().to_string();
                let ep = TensorRTExecutionProvider::default()
                    .with_device_id(0)
                    .with_fp16(precision == Precision::Fp16)
                    // The Orin Nano's GPU shares its 8 GB with the CPU: keep the build's scratch small.
                    .with_max_workspace_size(1 << 30)
                    .with_engine_cache(true)
                    .with_engine_cache_path(&cache)
                    .with_engine_cache_prefix(format!("{tag}_{}", precision.name()))
                    .with_timing_cache(true)
                    .with_timing_cache_path(&cache);
                b.with_execution_providers([ep.build().error_on_failure(), cuda_ep()]).map_err(err)
            }
            _ => Err(format!("{}: no fixed-shape sessions on {:?}", self.path.display(), self.device)),
        }
    }

    /// Run a batch of `n` images of (h, w), CHW float32 back to back. Returns the raw output and
    /// its shape ([n, 4 + classes, anchors]).
    pub fn run(&mut self, data: Vec<f32>, n: usize, h: usize, w: usize) -> Result<(Vec<usize>, Vec<f32>), String> {
        let input_name = self.input.clone();
        let session = if self.is_static(n, h, w) { self.static_session((n, h, w))? } else { &mut self.dynamic };
        let input = Tensor::from_array(([n, 3, h, w], data.into_boxed_slice())).map_err(|e| e.to_string())?;
        let outputs = session.run(ort::inputs![input_name.as_str() => input]).map_err(|e| e.to_string())?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        Ok((shape.iter().map(|&d| d as usize).collect(), values.to_vec()))
    }

    /// With HV_ORT_PROFILE set: stop profiling every session and return the profile files written.
    pub fn end_profiling(&mut self) -> Vec<String> {
        if std::env::var_os("HV_ORT_PROFILE").map_or(true, |d| d.is_empty()) {
            return vec![];
        }
        let mut files = vec![];
        for s in std::iter::once(&mut self.dynamic).chain(self.static_sessions.values_mut()) {
            if let Ok(f) = s.end_profiling() {
                files.push(f);
            }
        }
        files
    }
}

/// ONNX Runtime's build string (version, git commit, flags), once a model has loaded it.
pub fn ort_build_info() -> String {
    ort::info().to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn names_from_metadata() {
        assert_eq!(super::parse_names("{0: 'goalie_a', 1: 'b', 10: 'x'}"), vec!["goalie_a", "b", "x"]);
    }
}
