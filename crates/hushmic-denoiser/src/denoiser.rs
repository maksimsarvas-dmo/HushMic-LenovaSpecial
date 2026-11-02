use crate::attn::AttnLimiter;
use crate::error::Error;
use crate::mode::{GainRamp, Mode};
use crate::model::Model;
use crate::stft::{Analysis, Synthesis, HOP, SPEC_LEN};
use std::path::Path;

/// A streaming DPDFNet noise suppressor: 48 kHz mono, one 480-sample hop at a
/// time. `Send`, so the usual pattern is construct-at-init, move into the
/// audio thread. For arbitrary chunk sizes, wrap it in
/// [`StreamDenoiser`](crate::StreamDenoiser).
pub struct Denoiser {
    analysis: Analysis,
    synthesis: Synthesis,
    model: Model,
    state: Vec<f32>,
    spec: [f32; SPEC_LEN],
    spec_e: [f32; SPEC_LEN],
    state_out: Vec<f32>,
    attn: AttnLimiter,
    gain: GainRamp,
    mode: Mode,
    attn_limit_db: f32,
    /// Why a native engine was asked for but ONNX runs.
    #[cfg(feature = "native")]
    fallback_reason: Option<String>,
    /// Why an OpenVINO NPU engine was asked for but ONNX runs.
    #[cfg(feature = "npu")]
    npu_fallback_reason: Option<String>,
}

impl Denoiser {
    /// Load a DPDFNet model from a `.onnx` file.
    ///
    /// With the default `load-dynamic` feature, if no ONNX Runtime
    /// environment is committed yet one is resolved now: `ORT_DYLIB_PATH` if
    /// set and non-empty, else the platform soname (an executable-adjacent
    /// copy wins over the dlopen default search, mirroring ort). Apps that
    /// bundle their own runtime should call `init_runtime` first. Never
    /// panics on a missing/incompatible runtime — that is [`Error::Runtime`].
    /// (Static-linking builds skip all of this; ort brings its own runtime.)
    pub fn from_file(model_path: impl AsRef<Path>) -> Result<Denoiser, Error> {
        crate::runtime::ensure_runtime()?;
        let model = Model::load(model_path.as_ref()).map_err(Error::Model)?;
        Ok(Denoiser::with_model(model))
    }

    /// Load a DPDFNet model from in-memory bytes (e.g. `include_bytes!` or a
    /// download your app manages). Runtime handling as in [`Denoiser::from_file`].
    pub fn from_memory(model_bytes: &[u8]) -> Result<Denoiser, Error> {
        crate::runtime::ensure_runtime()?;
        let model = Model::from_memory(model_bytes).map_err(Error::Model)?;
        Ok(Denoiser::with_model(model))
    }

    /// Load a model file and run it on the engine `inference` selects
    /// (feature `native`). The native engine reads the packed weights
    /// `<model>.weights.f32` from the model's directory and checks them
    /// before use. Every native failure — a model file without a native
    /// build, missing or damaged weights, a CPU without AVX2/FMA —
    /// falls back to ONNX; [`Denoiser::fallback_reason`] then says why.
    /// Errors are those of [`Denoiser::from_file`] (the ONNX path).
    ///
    /// The native engine does not need ONNX Runtime, but the ONNX fallback
    /// does, so the runtime handling is the same as in `from_file`.
    ///
    /// The native engine keeps kernel scratch on the stack: the thread that
    /// calls [`Denoiser::process_hop`] needs at least 1 MiB of it (Rust's
    /// spawned threads get 2 MiB unless `RUST_MIN_STACK` says otherwise).
    #[cfg(feature = "native")]
    pub fn from_file_with(
        model_path: impl AsRef<Path>,
        inference: crate::Inference,
    ) -> Result<Denoiser, Error> {
        let model_path = model_path.as_ref();
        if inference == crate::Inference::Onnx {
            return Denoiser::from_file(model_path);
        }
        match crate::native::NativeModel::load(model_path, inference) {
            Ok((m, init_state)) => Ok(Denoiser::with_model(Model::native(m, init_state))),
            Err(why) => {
                let mut d = Denoiser::from_file(model_path)?;
                d.fallback_reason = Some(why);
                Ok(d)
            }
        }
    }

    /// The engine this instance runs on (feature `native`).
    #[cfg(feature = "native")]
    pub fn engine(&self) -> crate::Engine {
        self.model.engine()
    }

    /// The kernel set the native engine picked when it was created:
    /// `"avx-vnni"` on CPUs with AVX-VNNI (when the build has those
    /// kernels), else `"avx2"`. `None` on ONNX (feature `native`).
    #[cfg(feature = "native")]
    pub fn native_kernels(&self) -> Option<&'static str> {
        self.model.native_kernels()
    }

    /// Why this instance runs ONNX although [`Denoiser::from_file_with`]
    /// asked for a native engine (feature `native`); `None` when it got the
    /// engine it asked for.
    #[cfg(feature = "native")]
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    /// Load a model file and run it on the OpenVINO `NPU` device when
    /// `inference` selects it (`Auto` or `Npu`), falling back to ONNX Runtime
    /// CPU and recording [`Denoiser::npu_fallback_reason`] if `openvino_c` or
    /// the `NPU` device is unavailable (feature `npu`). `Auto` also falls
    /// back when a short on-device probe misses [`crate::NPU_HOP_GATE_MS`]
    /// or the NPU waveform is more than 30 dB away from ONNX Runtime.
    #[cfg(feature = "npu")]
    pub fn from_file_npu(
        model_path: impl AsRef<Path>,
        inference: crate::NpuInference,
        cache_dir: Option<&Path>,
    ) -> Result<Denoiser, Error> {
        let model_path = model_path.as_ref();
        if inference == crate::NpuInference::Onnx {
            return Denoiser::from_file(model_path);
        }
        // The NPU gets the feature block wrong in FP16, so when the split
        // sidecars exist (scripts/split-npu-model.py) the feature block runs
        // on CPU and only the body runs on the NPU.
        let loaded = if crate::hybrid::split_available(model_path) {
            crate::runtime::ensure_runtime()?;
            crate::hybrid::HybridModel::load(model_path, cache_dir)
                .map(|(m, init_state)| Model::hybrid(m, init_state))
        } else {
            crate::openvino::OpenVinoModel::load(model_path, inference, cache_dir)
                .map(|(m, init_state)| Model::npu(m, init_state))
        };
        match loaded {
            Ok(model) => {
                let mut d = Denoiser::with_model(model);
                // Explicit `npu` keeps the device even when it misses the hop
                // budget, so qualification benches can report that result.
                // `auto` must stay real-time, so a slow NPU yields the CPU.
                if inference == crate::NpuInference::Auto {
                    auto_select(d, model_path)
                } else {
                    d.reset();
                    Ok(d)
                }
            }
            Err(why) => onnx_npu_fallback(model_path, why),
        }
    }

    /// Which backend (`Npu` or `Onnx`) this instance runs on (feature `npu`).
    #[cfg(feature = "npu")]
    pub fn npu_engine(&self) -> crate::NpuEngine {
        self.model.npu_engine()
    }

    /// Why this instance runs ONNX Runtime CPU although [`Denoiser::from_file_npu`]
    /// asked for the OpenVINO `NPU` backend (feature `npu`); `None` when it
    /// got the engine it asked for.
    #[cfg(feature = "npu")]
    pub fn npu_fallback_reason(&self) -> Option<&str> {
        self.npu_fallback_reason.as_deref()
    }

    fn with_model(model: Model) -> Denoiser {
        let state = model.init_state.clone();
        let state_out = vec![0f32; model.state_size];
        Denoiser {
            analysis: Analysis::new(),
            synthesis: Synthesis::new(),
            model,
            state,
            spec: [0f32; SPEC_LEN],
            spec_e: [0f32; SPEC_LEN],
            state_out,
            attn: AttnLimiter::new(),
            gain: GainRamp::new(),
            mode: Mode::Process,
            attn_limit_db: f32::INFINITY,
            #[cfg(feature = "native")]
            fallback_reason: None,
            #[cfg(feature = "npu")]
            npu_fallback_reason: None,
        }
    }

    /// Clear all stream state (STFT rings, recurrent model state, ramps) for
    /// a fresh session. Configuration survives: the current [`Mode`] and
    /// attenuation limit stay in force — in particular, a `Denoiser` reset
    /// while muted stays silent from the very first sample.
    pub fn reset(&mut self) {
        self.analysis.reset();
        self.synthesis.reset();
        self.state.clear();
        self.state.extend_from_slice(&self.model.init_state);
        self.attn.reset();
        self.gain.reset_to(self.mode == Mode::Mute);
    }

    /// Cap how deep suppression may go by blending the latency-aligned noisy
    /// signal back in: `0.0` disables suppression entirely (you get the
    /// aligned input back), values up to ~100 allow progressively deeper
    /// suppression, and `>= 200` or non-finite means unlimited (pure model
    /// output). A fresh `Denoiser` defaults to unlimited.
    pub fn set_attenuation_limit_db(&mut self, db: f32) {
        self.attn_limit_db = db;
        self.attn.set_db(db);
    }

    /// The last value passed to [`Denoiser::set_attenuation_limit_db`]
    /// (`f32::INFINITY` — unlimited — on a fresh instance).
    pub fn attenuation_limit_db(&self) -> f32 {
        self.attn_limit_db
    }

    /// The engine keeps running in every mode (state stays warm, transitions
    /// are instant); bypass and mute only change what leaves it.
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        self.attn.set_bypass(mode == Mode::Bypass);
        self.gain.set_muted(mode == Mode::Mute);
    }

    /// The current [`Mode`].
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Algorithmic latency of the process path in samples: input at sample
    /// `n` shapes output at sample `n + latency_samples()`. Equals
    /// [`LATENCY_SAMPLES`](crate::LATENCY_SAMPLES) for the shipped DPDFNet
    /// models; an instance method so future models with different group
    /// delays stay API-compatible.
    pub fn latency_samples(&self) -> usize {
        crate::LATENCY_SAMPLES
    }

    /// Process exactly one hop. `output` is ALWAYS filled, even on `Err`: a
    /// transient model failure feeds a zero frame through the attenuation
    /// delay line and the OLA synthesis instead of skipping them, so every
    /// ring stays in lockstep with the analysis ring and the next good frame
    /// reconstructs correctly (skipping would desynchronize the overlap-add
    /// by one hop for good).
    pub fn process_hop(
        &mut self,
        input: &[f32; HOP],
        output: &mut [f32; HOP],
    ) -> Result<(), Error> {
        self.analysis.push_hop(input, &mut self.spec);
        let result = self.model.run(
            &self.spec,
            &self.state,
            &mut self.spec_e,
            &mut self.state_out,
        );
        match &result {
            Ok(()) => {
                std::mem::swap(&mut self.state, &mut self.state_out);
            }
            Err(_) => {
                // Keep the recurrent state as-is (last good frame) and emit a
                // zero spectrum; attn.apply below still blends in the delayed
                // noisy floor, so a capped limiter degrades to quiet passthrough
                // rather than a hard dropout.
                self.spec_e = [0f32; SPEC_LEN];
            }
        }
        self.attn.apply(&self.spec, &mut self.spec_e); // blend noisy floor / bypass mix
        self.synthesis.add_frame(&self.spec_e, output);
        self.gain.process(output); // mute ramp, after synthesis
        result.map_err(Error::Inference)
    }
}

#[cfg(feature = "npu")]
fn onnx_npu_fallback(model_path: &Path, why: String) -> Result<Denoiser, Error> {
    let mut d = Denoiser::from_file(model_path)?;
    d.npu_fallback_reason = Some(why);
    Ok(d)
}

/// Architecture parity floor: NPU output versus the ONNX Runtime reference.
const NPU_PARITY_SNR_DB: f64 = 30.0;

/// Keep the NPU only when it is fast enough and numerically close to ONNX.
#[cfg(feature = "npu")]
fn auto_select(mut npu: Denoiser, model_path: &Path) -> Result<Denoiser, Error> {
    let mut cpu = Denoiser::from_file(model_path)?;
    let (median_ms, snr_db) = match parity_probe(&mut npu, &mut cpu) {
        Ok(stats) => stats,
        Err(err) => {
            cpu.npu_fallback_reason = Some(format!("NPU hop probe failed: {err}"));
            return Ok(cpu);
        }
    };
    let latency_ok = median_ms <= crate::NPU_HOP_GATE_MS;
    let parity_ok = snr_db >= NPU_PARITY_SNR_DB;
    if latency_ok && parity_ok {
        npu.reset();
        return Ok(npu);
    }
    let mut why = Vec::new();
    if !latency_ok {
        why.push(format!(
            "hop median {median_ms:.2} ms exceeds {:.1} ms real-time gate",
            crate::NPU_HOP_GATE_MS
        ));
    }
    if !parity_ok {
        why.push(format!(
            "waveform SNR vs ONNX {snr_db:.1} dB is below {NPU_PARITY_SNR_DB:.0} dB"
        ));
    }
    cpu.reset();
    cpu.npu_fallback_reason = Some(format!("NPU {}", why.join("; ")));
    Ok(cpu)
}

/// Same hops through both engines. Returns `(npu median ms, SNR of NPU vs ONNX)`.
///
/// The probe signal is continuous voiced harmonics with vibrato, gated on and
/// off, over deterministic noise: the model passes the voiced part, so the
/// compared output is not near-silence (a bare sine or a repeated hop is
/// treated as noise and gated, which makes an SNR figure meaningless).
/// Latency is judged by the median: tail latency under system load hits the
/// CPU path just as much, and the median is what the device can sustain.
#[cfg(feature = "npu")]
fn parity_probe(npu: &mut Denoiser, cpu: &mut Denoiser) -> Result<(f64, f64), Error> {
    const WARMUP: usize = 16;
    const MEASURED: usize = 64;
    let sample_rate = crate::SAMPLE_RATE as f32;
    let tau = 2.0 * std::f32::consts::PI;
    let mut phase = 0.0f32;
    let mut lcg: u32 = 0x1234_5678;
    let mut hop_in = [0.0f32; HOP];
    let mut npu_out = [0.0f32; HOP];
    let mut cpu_out = [0.0f32; HOP];
    let mut signal = 0.0f64;
    let mut error = 0.0f64;
    let mut times = Vec::with_capacity(MEASURED);
    for hop in 0..(WARMUP + MEASURED) {
        for (i, sample) in hop_in.iter_mut().enumerate() {
            let t = ((hop * HOP + i) as f32) / sample_rate;
            let f0 = 150.0 + 25.0 * (tau * 2.5 * t).sin();
            phase = (phase + tau * f0 / sample_rate) % tau;
            let voice = 0.2 * phase.sin()
                + 0.14 * (2.0 * phase).sin()
                + 0.1 * (3.0 * phase).sin()
                + 0.06 * (4.0 * phase).sin()
                + 0.04 * (5.0 * phase).sin();
            let gate = ((tau * 1.2 * t).sin() * 4.0).clamp(0.0, 1.0);
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (lcg >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            *sample = voice * gate + 0.05 * noise + 0.02 * (tau * 120.0 * t).sin();
        }
        let started = std::time::Instant::now();
        npu.process_hop(&hop_in, &mut npu_out)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        cpu.process_hop(&hop_in, &mut cpu_out)?;
        if hop >= WARMUP {
            times.push(elapsed_ms);
            for i in 0..HOP {
                let reference = f64::from(cpu_out[i]);
                let delta = f64::from(npu_out[i]) - reference;
                signal += reference * reference;
                error += delta * delta;
            }
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = times[times.len() / 2];
    let snr_db = if error <= 1e-20 {
        120.0
    } else {
        10.0 * (signal / error).log10()
    };
    Ok((median, snr_db))
}