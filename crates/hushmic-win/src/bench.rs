//! Hop-time qualification benchmark (`hushmic-win bench`) and offline WAV
//! denoiser (`hushmic-win denoise <in.wav> <out.wav>`).

use crate::audio::{downmix_to_mono, Resampler48k};
use crate::config::{init_adjacent_ort_runtime, Config};
use hushmic_denoiser::{Denoiser, NpuInference, StreamDenoiser, HOP, SAMPLE_RATE};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// E16 NPU qualification threshold from the architecture plan: p99 <= 8.0 ms per 10 ms hop.
pub const NPU_P99_GATE_MS: f64 = 8.0;

#[derive(Debug, Clone, PartialEq)]
pub struct BenchOptions {
    pub inference: NpuInference,
    pub model: Option<String>,
    pub hops: usize,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            inference: NpuInference::Auto,
            model: None,
            hops: 1000,
        }
    }
}

pub fn parse_bench_args(args: &[&str]) -> Result<BenchOptions, String> {
    let mut opts = BenchOptions::default();
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--engine" | "--inference" => {
                i += 1;
                let val = args.get(i).ok_or("--engine requires auto|npu|onnx")?;
                opts.inference = NpuInference::parse(val)
                    .ok_or_else(|| format!("unknown engine '{val}' (use auto|npu|onnx)"))?;
            }
            "--model" => {
                i += 1;
                let val = args.get(i).ok_or("--model requires a model name or path")?;
                opts.model = Some((*val).to_string());
            }
            "--hops" => {
                i += 1;
                let val = args.get(i).ok_or("--hops requires a positive integer")?;
                let n: usize = val.parse().map_err(|_| format!("invalid --hops '{val}'"))?;
                if n == 0 {
                    return Err("--hops must be >= 1".to_string());
                }
                opts.hops = n;
            }
            other => {
                return Err(format!(
                    "unknown bench flag '{other}' (usage: hushmic-win bench [--engine auto|npu|onnx] [--model MODEL] [--hops N])"
                ));
            }
        }
        i += 1;
    }
    Ok(opts)
}

#[derive(Debug, Clone)]
pub struct BenchStats {
    pub min_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub rtf: f64,
    pub gate_passed: bool,
}

/// Compute percentile and real-time factor statistics from a non-empty list of hop times in ms.
pub fn compute_stats(samples_ms: &[f64]) -> BenchStats {
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let pick = |q: f64| -> f64 {
        if n == 0 {
            return 0.0;
        }
        let idx = ((n as f64 - 1.0) * q).round() as usize;
        sorted[idx.min(n - 1)]
    };
    let min_ms = sorted.first().copied().unwrap_or(0.0);
    let max_ms = sorted.last().copied().unwrap_or(0.0);
    let p50_ms = pick(0.50);
    let p95_ms = pick(0.95);
    let p99_ms = pick(0.99);
    let sum: f64 = sorted.iter().sum();
    let mean_ms = if n > 0 { sum / (n as f64) } else { 0.0 };
    let rtf = mean_ms / 10.0;
    let gate_passed = p99_ms <= NPU_P99_GATE_MS;
    BenchStats {
        min_ms,
        p50_ms,
        p95_ms,
        p99_ms,
        max_ms,
        mean_ms,
        rtf,
        gate_passed,
    }
}

pub fn run_bench(args: &[&str]) -> Result<String, String> {
    let opts = parse_bench_args(args)?;
    let cfg = Config::load();
    init_adjacent_ort_runtime();
    let model_path = cfg.resolve_model_path(opts.model.as_deref());
    let cache_dir = Config::ov_cache_dir();

    let mut denoiser = Denoiser::from_file_npu(&model_path, opts.inference, Some(&cache_dir))
        .map_err(|e| format!("load model {}: {e}", model_path.display()))?;

    let engine = denoiser.npu_engine();
    let fallback = denoiser.npu_fallback_reason().map(str::to_string);

    // Deterministic synthetic voice-like test signal so hops exercise non-zero math.
    let mut hop_in = [0.0f32; HOP];
    for (i, s) in hop_in.iter_mut().enumerate() {
        let t = (i as f32) / (SAMPLE_RATE as f32);
        *s = 0.25 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
            + 0.05 * (2.0 * std::f32::consts::PI * 1200.0 * t).cos();
    }
    let mut hop_out = [0.0f32; HOP];

    // Warm up 20 hops off the timer.
    for _ in 0..20 {
        denoiser
            .process_hop(&hop_in, &mut hop_out)
            .map_err(|e| format!("warmup hop: {e}"))?;
    }

    let mut times_ms = Vec::with_capacity(opts.hops);
    for _ in 0..opts.hops {
        let t0 = Instant::now();
        denoiser
            .process_hop(&hop_in, &mut hop_out)
            .map_err(|e| format!("bench hop: {e}"))?;
        times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let stats = compute_stats(&times_ms);
    let gate_str = if stats.gate_passed { "PASS" } else { "FAIL" };
    let mut report = format!(
        "model: {}\nrequested_engine: {}\nactive_engine: {}\nhops: {} ({:.1} s audio)\nmin: {:.3} ms\np50: {:.3} ms\np95: {:.3} ms\np99: {:.3} ms\nmax: {:.3} ms\nmean: {:.3} ms (RTF {:.4}x)\nqual_gate (p99 <= {:.1} ms): {}",
        model_path.display(),
        opts.inference.word(),
        engine,
        opts.hops,
        (opts.hops as f64) * 0.01,
        stats.min_ms,
        stats.p50_ms,
        stats.p95_ms,
        stats.p99_ms,
        stats.max_ms,
        stats.mean_ms,
        stats.rtf,
        NPU_P99_GATE_MS,
        gate_str
    );
    if let Some(reason) = fallback {
        report.push_str(&format!("\nfallback_reason: {reason}"));
    }
    Ok(report)
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenoiseWavOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub inference: NpuInference,
    pub model: Option<String>,
}

pub fn parse_denoise_args(args: &[&str]) -> Result<DenoiseWavOptions, String> {
    if args.len() < 2 {
        return Err(
            "usage: hushmic-win denoise <in.wav> <out.wav> [--engine auto|npu|onnx] [--model MODEL]"
                .to_string(),
        );
    }
    let mut opts = DenoiseWavOptions {
        input: PathBuf::from(args[0]),
        output: PathBuf::from(args[1]),
        inference: NpuInference::Auto,
        model: None,
    };
    let mut i = 2;
    while i < args.len() {
        match args[i] {
            "--engine" | "--inference" => {
                i += 1;
                let val = args.get(i).ok_or("--engine requires auto|npu|onnx")?;
                opts.inference = NpuInference::parse(val)
                    .ok_or_else(|| format!("unknown engine '{val}' (use auto|npu|onnx)"))?;
            }
            "--model" => {
                i += 1;
                let val = args.get(i).ok_or("--model requires a model name or path")?;
                opts.model = Some((*val).to_string());
            }
            other => {
                return Err(format!("unknown denoise flag '{other}'"));
            }
        }
        i += 1;
    }
    Ok(opts)
}

pub fn run_denoise_wav(args: &[&str]) -> Result<String, String> {
    let opts = parse_denoise_args(args)?;
    let cfg = Config::load();
    init_adjacent_ort_runtime();
    let model_path = cfg.resolve_model_path(opts.model.as_deref());
    let cache_dir = Config::ov_cache_dir();

    let denoiser = Denoiser::from_file_npu(&model_path, opts.inference, Some(&cache_dir))
        .map_err(|e| format!("load model {}: {e}", model_path.display()))?;
    let engine = denoiser.npu_engine();
    let fallback = denoiser.npu_fallback_reason().map(str::to_string);

    let mono_48k = read_wav_as_mono_48k(&opts.input)?;
    let mut stream = StreamDenoiser::new(denoiser);
    let mut cleaned = Vec::with_capacity(mono_48k.len() + HOP * 8);
    cleaned.extend_from_slice(stream.process(&mono_48k));
    let tail = vec![0.0f32; stream.pending()];
    cleaned.extend_from_slice(stream.process(&tail));
    if let Some(err) = stream.take_error() {
        return Err(format!("inference error: {err}"));
    }

    let latency = stream.denoiser().latency_samples();
    let end = (latency + mono_48k.len()).min(cleaned.len());
    let aligned = if latency <= end {
        &cleaned[latency..end]
    } else {
        &[]
    };

    write_mono_48k_wav(&opts.output, aligned)?;
    let mut summary = format!(
        "denoised {} -> {} ({} samples @ 48 kHz, engine: {})",
        opts.input.display(),
        opts.output.display(),
        aligned.len(),
        engine
    );
    if let Some(reason) = fallback {
        summary.push_str(&format!(" [fallback: {reason}]"));
    }
    Ok(summary)
}

fn read_wav_as_mono_48k(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let spec = reader.spec();
    let channels = (spec.channels as usize).max(1);
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|e| format!("read float samples: {e}"))?,
        hound::SampleFormat::Int => {
            let scale = ((1i64 << (spec.bits_per_sample.saturating_sub(1))) as f32).max(1.0);
            reader
                .samples::<i32>()
                .map(|r| r.map(|v| (v as f32) / scale))
                .collect::<Result<Vec<f32>, _>>()
                .map_err(|e| format!("read int samples: {e}"))?
        }
    };
    let mut mono = Vec::new();
    downmix_to_mono(&interleaved, channels, &mut mono);
    if spec.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }
    let mut resampler = Resampler48k::new(spec.sample_rate)?;
    let mut out_48k = Vec::new();
    resampler.process(&mono, &mut out_48k)?;
    Ok(out_48k)
}

fn write_mono_48k_wav(path: &Path, samples: &[f32]) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    for &s in samples {
        let clamped = (s * 32768.0)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer
            .write_sample(clamped)
            .map_err(|e| format!("write sample: {e}"))?;
    }
    writer
        .finalize()
        .map_err(|e| format!("finalize {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bench_args_handles_engine_model_and_hops() {
        let opts =
            parse_bench_args(&["--engine", "npu", "--model", "light", "--hops", "250"]).unwrap();
        assert_eq!(opts.inference, NpuInference::Npu);
        assert_eq!(opts.model.as_deref(), Some("light"));
        assert_eq!(opts.hops, 250);
    }

    #[test]
    fn compute_stats_evaluates_p50_p99_and_gate() {
        let mut samples: Vec<f64> = (1..=100).map(|x| x as f64 * 0.05).collect();
        let stats = compute_stats(&samples);
        assert!(stats.p50_ms >= 2.4 && stats.p50_ms <= 2.6);
        assert!(stats.p99_ms <= NPU_P99_GATE_MS);
        assert!(stats.gate_passed);

        samples.push(15.0);
        samples.push(16.0);
        let slow = compute_stats(&samples);
        assert!(!slow.gate_passed);
    }

    #[test]
    fn parse_denoise_args_validates_paths_and_flags() {
        let opts = parse_denoise_args(&[
            "in.wav", "out.wav", "--engine", "onnx", "--model", "quality",
        ])
        .unwrap();
        assert_eq!(opts.input, PathBuf::from("in.wav"));
        assert_eq!(opts.output, PathBuf::from("out.wav"));
        assert_eq!(opts.inference, NpuInference::Onnx);
        assert_eq!(opts.model.as_deref(), Some("quality"));
    }
}
