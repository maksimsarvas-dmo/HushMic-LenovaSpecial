//! Audio pipeline for `hushmic-win`:
//! - Cross-platform DSP helpers: stereo/multi-channel downmix to mono, conditional
//!   `rubato` resampler to 48 kHz, bounded capture-to-inference `HopQueue`,
//!   latency-aligned `LateHopGuard` (1st late hop held in slack, 2nd consecutive
//!   late hop crossfades toward latency-aligned bypass and logs), and device-name
//!   matcher (default render target: `"CABLE Input"`).
//! - Windows WASAPI shared event-driven capture and render threads (`#[cfg(windows)]`),
//!   with the inference thread running `StreamDenoiser` on a 2 MiB stack under
//!   MMCSS `"Pro Audio"`.
//! - An activity gate so the NPU is submitted only while a call is capturing the
//!   virtual microphone or the physical microphone rises above its noise floor.

use crate::config::{Config, NpuGate, RunMode};
use hushmic_denoiser::{Denoiser, HOP, LATENCY_SAMPLES, NPU_HOP_GATE_MS, SAMPLE_RATE};
use rubato::{FftFixedInOut, Resampler};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// One hop period at 48 kHz (480 samples = 10.0 ms).
pub const HOP_PERIOD_MS: f64 = 10.0;

/// Delay in hops of the DPDFNet model (5 hops = 50 ms).
pub const MODEL_DELAY_HOPS: usize = LATENCY_SAMPLES / HOP;

/// Maximum buffered mono samples between capture and inference before old frames drop (100 ms).
const MAX_QUEUE_SAMPLES: usize = 10 * HOP;

/// Downmix interleaved multi-channel `f32` PCM into mono `f32` frames.
pub fn downmix_to_mono(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    out.clear();
    if channels == 0 {
        return;
    }
    if channels == 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    let frames = interleaved.len() / channels;
    out.reserve(frames);
    let inv = 1.0 / (channels as f32);
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().sum();
        out.push(sum * inv);
    }
}

/// Expand mono `f32` samples to interleaved `channels` `f32` samples.
pub fn mono_to_interleaved(mono: &[f32], channels: usize, out: &mut Vec<f32>) {
    out.clear();
    if channels == 0 {
        return;
    }
    if channels == 1 {
        out.extend_from_slice(mono);
        return;
    }
    out.reserve(mono.len() * channels);
    for &s in mono {
        for _ in 0..channels {
            out.push(s);
        }
    }
}

/// Conditional mono resampler to 48 kHz. When `input_rate == 48_000`, `process`
/// is a direct pass-through and `rubato` is not invoked.
pub struct Resampler48k {
    input_rate: u32,
    inner: Option<FftFixedInOut<f32>>,
    in_pending: Vec<f32>,
}

impl Resampler48k {
    pub fn new(input_rate: u32) -> Result<Self, String> {
        if input_rate == SAMPLE_RATE {
            return Ok(Self {
                input_rate,
                inner: None,
                in_pending: Vec::new(),
            });
        }
        // ~10 ms input chunk size
        let chunk_in = ((input_rate as usize) / 100).max(64);
        let fft = FftFixedInOut::<f32>::new(input_rate as usize, SAMPLE_RATE as usize, chunk_in, 1)
            .map_err(|e| format!("create 48 kHz resampler from {input_rate} Hz: {e}"))?;
        Ok(Self {
            input_rate,
            inner: Some(fft),
            in_pending: Vec::with_capacity(chunk_in * 2),
        })
    }

    pub fn is_passthrough(&self) -> bool {
        self.inner.is_none()
    }

    pub fn input_rate(&self) -> u32 {
        self.input_rate
    }

    pub fn process(&mut self, mono_in: &[f32], out_48k: &mut Vec<f32>) -> Result<(), String> {
        out_48k.clear();
        let Some(resampler) = &mut self.inner else {
            out_48k.extend_from_slice(mono_in);
            return Ok(());
        };
        self.in_pending.extend_from_slice(mono_in);
        loop {
            let needed = resampler.input_frames_next();
            if self.in_pending.len() < needed {
                break;
            }
            let chunk: Vec<f32> = self.in_pending.drain(..needed).collect();
            let waves = resampler
                .process(&[chunk], None)
                .map_err(|e| format!("resample to 48 kHz: {e}"))?;
            if let Some(ch0) = waves.first() {
                out_48k.extend_from_slice(ch0);
            }
        }
        Ok(())
    }
}

/// Case-insensitive device-name matcher. Prefers an exact case-insensitive match
/// and otherwise returns the first candidate whose name contains `query`.
pub fn match_device_name(candidates: &[String], query: Option<&str>) -> Option<usize> {
    let q = query?.trim();
    if q.is_empty() || q.eq_ignore_ascii_case("default") {
        return None;
    }
    let q_low = q.to_ascii_lowercase();
    if let Some(idx) = candidates
        .iter()
        .position(|c| c.trim().eq_ignore_ascii_case(q))
    {
        return Some(idx);
    }
    candidates
        .iter()
        .position(|c| c.to_ascii_lowercase().contains(&q_low))
}

/// Capture-side name paired with a virtual-cable render endpoint.
/// `"CABLE Input"` becomes `"CABLE Output"`. Real speakers have no pair.
pub fn paired_capture_query(render_device: &str) -> Option<String> {
    let lower = render_device.to_ascii_lowercase();
    let idx = lower.find("input")?;
    let mut paired = String::with_capacity(render_device.len() + 1);
    paired.push_str(&render_device[..idx]);
    let word = render_device.get(idx..idx + 5).unwrap_or("input");
    if word.starts_with(|c: char| c.is_ascii_uppercase()) {
        paired.push_str("Output");
    } else {
        paired.push_str("output");
    }
    paired.push_str(&render_device[idx + 5..]);
    Some(paired)
}

/// Hops of room-noise calibration before the adaptive threshold is trusted (500 ms).
const SIGNAL_LEARN_HOPS: usize = 50;
/// Keep the model running across short pauses so words are not chopped (800 ms).
const SIGNAL_HANGOVER_HOPS: u32 = 80;
/// Open immediately on a loud hop (~ -30 dBFS), even during calibration.
const SIGNAL_LOUD_RMS: f32 = 0.03162;
/// Never open below this RMS (~ -50 dBFS).
const SIGNAL_ABS_OPEN_RMS: f32 = 0.003162;
/// Open when the hop is this many times the tracked room floor (~12 dB).
const SIGNAL_SPEECH_RATIO: f32 = 4.0;

/// Microphone energy gate. A steady room tone (fan, mic self-noise) stays closed
/// after a short calibration; speech and other bursts open it.
pub struct SignalGate {
    learn: [f32; SIGNAL_LEARN_HOPS],
    learned: usize,
    noise_rms: f32,
    hangover_left: u32,
    open: bool,
}

impl Default for SignalGate {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalGate {
    pub fn new() -> Self {
        Self {
            learn: [0.0; SIGNAL_LEARN_HOPS],
            learned: 0,
            noise_rms: 0.0,
            hangover_left: 0,
            open: false,
        }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Returns whether this hop should wake inference.
    pub fn observe(&mut self, hop: &[f32]) -> bool {
        let mut acc = 0.0f32;
        for sample in hop {
            acc += sample * sample;
        }
        let rms = if hop.is_empty() {
            0.0
        } else {
            (acc / hop.len() as f32).sqrt()
        };
        let rms = if rms.is_finite() { rms } else { 0.0 };

        let threshold = if self.learned < SIGNAL_LEARN_HOPS {
            self.learn[self.learned] = rms;
            self.learned += 1;
            if self.learned == SIGNAL_LEARN_HOPS {
                let mut sorted = self.learn;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                self.noise_rms = sorted[SIGNAL_LEARN_HOPS / 5];
            }
            SIGNAL_LOUD_RMS
        } else {
            if rms < self.noise_rms {
                self.noise_rms += (rms - self.noise_rms) * 0.08;
            } else if !self.open {
                self.noise_rms += (rms - self.noise_rms) * 0.001;
            }
            (self.noise_rms * SIGNAL_SPEECH_RATIO).max(SIGNAL_ABS_OPEN_RMS)
        };

        if rms >= threshold {
            self.open = true;
            self.hangover_left = SIGNAL_HANGOVER_HOPS;
        } else if self.hangover_left > 0 {
            self.hangover_left -= 1;
            self.open = true;
        } else {
            self.open = false;
        }
        self.open
    }
}

/// What one hop should do with the neural net.
struct HopDecision {
    run_npu: bool,
    /// When inference is skipped, copy latency-aligned raw audio instead of silence.
    passthrough: bool,
    reason: &'static str,
}

fn decide_hop(
    gate: NpuGate,
    enabled: bool,
    mode: RunMode,
    call_active: Option<bool>,
    signal_open: bool,
) -> HopDecision {
    if mode == RunMode::Mute && enabled {
        return HopDecision {
            run_npu: false,
            passthrough: false,
            reason: "mute",
        };
    }
    if !enabled {
        return HopDecision {
            run_npu: false,
            passthrough: true,
            reason: "off",
        };
    }
    if mode == RunMode::Bypass {
        return HopDecision {
            run_npu: false,
            passthrough: true,
            reason: "bypass",
        };
    }

    let call = call_active == Some(true);
    let run = match gate {
        NpuGate::Always => true,
        NpuGate::Call => call,
        NpuGate::Signal => signal_open,
        NpuGate::Auto => call || signal_open,
    };
    let reason = if !run {
        "idle"
    } else if gate == NpuGate::Always {
        "always"
    } else if call && gate != NpuGate::Signal {
        "call"
    } else {
        "signal"
    };
    HopDecision {
        run_npu: run,
        // Idle still forwards the microphone. Only Mute replaces it with silence.
        passthrough: true,
        reason,
    }
}

/// Linear fade of one hop down to silence, used on the first idle hop after inference.
fn fade_out(from: &[f32; HOP], out: &mut [f32; HOP]) {
    let inv = 1.0 / HOP as f32;
    for (i, (dst, src)) in out.iter_mut().zip(from.iter()).enumerate() {
        *dst = src * (1.0 - i as f32 * inv);
    }
}

/// Bounded queue carrying 48 kHz mono samples from the WASAPI capture callback
/// to the inference thread.
pub struct HopQueue {
    inner: Mutex<HopQueueInner>,
    cv: Condvar,
}

struct HopQueueInner {
    samples: VecDeque<f32>,
    closed: bool,
    dropped_samples: u64,
}

impl Default for HopQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl HopQueue {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HopQueueInner {
                samples: VecDeque::with_capacity(MAX_QUEUE_SAMPLES),
                closed: false,
                dropped_samples: 0,
            }),
            cv: Condvar::new(),
        }
    }

    /// Enqueue captured 48 kHz mono samples without blocking the audio callback for inference.
    pub fn push_samples(&self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        g.samples.extend(samples.iter().copied());
        if g.samples.len() > MAX_QUEUE_SAMPLES {
            let overflow = g.samples.len() - MAX_QUEUE_SAMPLES;
            g.samples.drain(..overflow);
            g.dropped_samples += overflow as u64;
        }
        self.cv.notify_one();
    }

    /// Drop samples still waiting. Used when the capture device changes so the
    /// previous microphone does not keep playing.
    pub fn clear(&self) {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        g.samples.clear();
        self.cv.notify_all();
    }

    /// Pop exactly one 480-sample hop, waiting up to `timeout` for a full hop.
    pub fn pop_hop(&self, out: &mut [f32; HOP], timeout: std::time::Duration) -> Option<bool> {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if g.samples.len() < HOP && !g.closed {
            let (guard, _) = self
                .cv
                .wait_timeout(g, timeout)
                .unwrap_or_else(|p| p.into_inner());
            g = guard;
        }
        if g.samples.len() >= HOP {
            for slot in out.iter_mut() {
                *slot = g.samples.pop_front().unwrap_or(0.0);
            }
            Some(true)
        } else if g.closed {
            None
        } else {
            Some(false)
        }
    }

    pub fn close(&self) {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        g.closed = true;
        self.cv.notify_all();
    }

    pub fn dropped_samples(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .dropped_samples
    }
}

/// Outcome of [`LateHopGuard::observe_hop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LateOutcome {
    /// Hop finished within [`NPU_HOP_GATE_MS`].
    OnTime,
    /// First consecutive hop past [`NPU_HOP_GATE_MS`]: absorbed by slack.
    HeldOnce,
    /// Second (or later) consecutive late hop: output was crossfaded toward
    /// latency-aligned bypass for this hop and should be logged.
    CrossfadedToBypass,
}

/// Tracks hop inference duration against [`NPU_HOP_GATE_MS`] and maintains a
/// 5-hop (50 ms) delay line of raw input hops so a second consecutive late hop
/// can smoothly crossfade to latency-aligned bypass.
pub struct LateHopGuard {
    raw_delay: VecDeque<[f32; HOP]>,
    consecutive_late: u32,
    total_late: u64,
    total_crossfades: u64,
}

impl Default for LateHopGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl LateHopGuard {
    pub fn new() -> Self {
        let mut raw_delay = VecDeque::with_capacity(MODEL_DELAY_HOPS + 1);
        for _ in 0..MODEL_DELAY_HOPS {
            raw_delay.push_back([0.0; HOP]);
        }
        Self {
            raw_delay,
            consecutive_late: 0,
            total_late: 0,
            total_crossfades: 0,
        }
    }

    pub fn consecutive_late(&self) -> u32 {
        self.consecutive_late
    }

    pub fn total_late(&self) -> u64 {
        self.total_late
    }

    pub fn total_crossfades(&self) -> u64 {
        self.total_crossfades
    }

    /// Push one raw hop and return the sample-aligned hop from
    /// [`MODEL_DELAY_HOPS`] ago. Used for bypass without running the model.
    pub fn delay_raw(&mut self, raw_input_hop: &[f32; HOP]) -> [f32; HOP] {
        self.raw_delay.push_back(*raw_input_hop);
        self.raw_delay.pop_front().unwrap_or(*raw_input_hop)
    }

    /// Drop the delay line back to silence. Totals are kept for status.
    pub fn reset(&mut self) {
        for frame in &mut self.raw_delay {
            *frame = [0.0; HOP];
        }
        self.consecutive_late = 0;
    }

    /// Record a completed hop, updating `denoised_hop` in place if a second
    /// consecutive late hop requires crossfading toward latency-aligned bypass.
    pub fn observe_hop(
        &mut self,
        raw_input_hop: &[f32; HOP],
        denoised_hop: &mut [f32; HOP],
        elapsed_ms: f64,
    ) -> LateOutcome {
        let aligned_raw = self.delay_raw(raw_input_hop);

        if elapsed_ms <= NPU_HOP_GATE_MS {
            self.consecutive_late = 0;
            return LateOutcome::OnTime;
        }

        self.consecutive_late += 1;
        self.total_late += 1;
        if self.consecutive_late == 1 {
            LateOutcome::HeldOnce
        } else {
            self.total_crossfades += 1;
            let inv = 1.0 / (HOP as f32);
            for i in 0..HOP {
                let alpha = (i as f32) * inv;
                denoised_hop[i] = denoised_hop[i] * (1.0 - alpha) + aligned_raw[i] * alpha;
            }
            LateOutcome::CrossfadedToBypass
        }
    }
}

/// Shared runtime telemetry read by `hushmic-win status` and the tray tooltip.
#[derive(Debug, Clone)]
pub struct RuntimeSnapshot {
    pub enabled: bool,
    pub mode: RunMode,
    pub engine: String,
    pub fallback_reason: Option<String>,
    pub model: String,
    pub attn_limit_db: f32,
    pub capture_device: String,
    pub render_device: String,
    pub last_hop_ms: f64,
    pub late_hops: u64,
    pub bypass_crossfades: u64,
    /// Configured [`NpuGate`] word (`auto`, `call`, `signal`, `always`).
    pub npu_gate: &'static str,
    /// `true` when the latest hop was submitted to the neural net.
    pub npu_active: bool,
    /// Why the latest hop did or did not run: `call`, `signal`, `always`, `idle`, `bypass`, `off`, `mute`.
    pub npu_activity: &'static str,
    /// Process holding the virtual microphone when `npu_activity` is `call`.
    pub npu_client: String,
}

impl Default for RuntimeSnapshot {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: RunMode::Suppress,
            engine: "onnx".to_string(),
            fallback_reason: None,
            model: "dpdfnet8_48khz_hr".to_string(),
            attn_limit_db: 100.0,
            capture_device: "default".to_string(),
            render_device: "CABLE Input".to_string(),
            last_hop_ms: 0.0,
            late_hops: 0,
            bypass_crossfades: 0,
            npu_gate: "auto",
            npu_active: false,
            npu_activity: "idle",
            npu_client: String::new(),
        }
    }
}

/// Live controller shared between the WASAPI inference thread, named-pipe server, and tray.
pub struct SharedEngineState {
    pub config: Mutex<Config>,
    pub snapshot: Mutex<RuntimeSnapshot>,
    pub stop: AtomicBool,
}

impl SharedEngineState {
    pub fn new(cfg: Config) -> Arc<Self> {
        let snap = RuntimeSnapshot {
            enabled: cfg.enabled,
            mode: cfg.mode,
            engine: cfg.inference.clone(),
            fallback_reason: None,
            model: cfg.model.clone(),
            attn_limit_db: cfg.attn_limit,
            capture_device: cfg.mic.clone().unwrap_or_else(|| "default".to_string()),
            render_device: cfg.render_device.clone(),
            last_hop_ms: 0.0,
            late_hops: 0,
            bypass_crossfades: 0,
            npu_gate: cfg.npu_gate.as_str(),
            npu_active: false,
            npu_activity: "idle",
            npu_client: String::new(),
        };
        Arc::new(Self {
            config: Mutex::new(cfg),
            snapshot: Mutex::new(snap),
            stop: AtomicBool::new(false),
        })
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    pub fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }
}

/// Construct a configured [`Denoiser`] using the OpenVINO NPU backend when
/// available and falling back to ONNX Runtime CPU with a logged reason.
pub fn build_denoiser(cfg: &Config) -> Result<(Denoiser, String, Option<String>), String> {
    crate::config::init_adjacent_ort_runtime();
    let model_path = cfg.resolve_model_path(None);
    let cache_dir = Config::ov_cache_dir();
    let requested = cfg.npu_inference();
    let mut denoiser = Denoiser::from_file_npu(&model_path, requested, Some(&cache_dir))
        .map_err(|e| format!("load model {}: {e}", model_path.display()))?;
    denoiser.set_attenuation_limit_db(cfg.attn_limit);
    denoiser.set_mode(if cfg.enabled {
        cfg.mode.to_denoiser_mode()
    } else {
        hushmic_denoiser::Mode::Bypass
    });
    let engine = denoiser.npu_engine().to_string();
    let fallback = denoiser.npu_fallback_reason().map(str::to_string);
    Ok((denoiser, engine, fallback))
}

#[cfg(windows)]
pub mod win_wasapi {
    use super::*;
    use hushmic_denoiser::StreamDenoiser;
    use std::time::{Duration, Instant};
    use wasapi::{Device, DeviceCollection, Direction, SampleType, StreamMode, WaveFormat};

    /// Join MMCSS `"Pro Audio"` on the current thread; returns the task handle if registered.
    struct MmcssGuard(isize);

    impl MmcssGuard {
        fn enter_pro_audio() -> Option<Self> {
            use std::os::windows::ffi::OsStrExt;
            use windows_sys::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
            let name: Vec<u16> = std::ffi::OsStr::new("Pro Audio")
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let mut task_idx: u32 = 0;
            // SAFETY: registers the calling audio worker thread with MMCSS "Pro Audio".
            let handle = unsafe { AvSetMmThreadCharacteristicsW(name.as_ptr(), &mut task_idx) };
            if handle.is_null() {
                None
            } else {
                Some(MmcssGuard(handle as isize))
            }
        }
    }

    impl Drop for MmcssGuard {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Threading::AvRevertMmThreadCharacteristics;
            // SAFETY: reverts the valid MMCSS handle acquired in `enter_pro_audio`.
            unsafe {
                AvRevertMmThreadCharacteristics(self.0 as *mut core::ffi::c_void);
            }
        }
    }

    fn pick_device(direction: Direction, query: Option<&str>) -> Result<(Device, String), String> {
        let coll = DeviceCollection::new(&direction)
            .map_err(|e| format!("enumerate {direction} devices: {e}"))?;
        let count = coll
            .get_nbr_devices()
            .map_err(|e| format!("count {direction} devices: {e}"))?;
        let mut names = Vec::with_capacity(count as usize);
        let mut devices = Vec::with_capacity(count as usize);
        for idx in 0..count {
            if let Ok(dev) = coll.get_device_at_index(idx) {
                let name = dev
                    .get_friendlyname()
                    .unwrap_or_else(|_| format!("{direction}-{idx}"));
                names.push(name);
                devices.push(dev);
            }
        }
        if let Some(matched_idx) = match_device_name(&names, query) {
            let name = names.swap_remove(matched_idx);
            let dev = devices.swap_remove(matched_idx);
            return Ok((dev, name));
        }
        let default_dev = wasapi::get_default_device(&direction)
            .map_err(|e| format!("get default {direction} device: {e}"))?;
        let default_name = default_dev
            .get_friendlyname()
            .unwrap_or_else(|_| format!("Default {direction}"));
        if let Some(q) =
            query.filter(|s| !s.trim().is_empty() && !s.eq_ignore_ascii_case("default"))
        {
            eprintln!(
                "hushmic-win: {direction} device matching '{q}' not found; using '{default_name}'"
            );
        }
        Ok((default_dev, default_name))
    }

    /// Active capture endpoints. The first value is the Windows default microphone.
    pub fn capture_device_names() -> Result<(String, Vec<String>), String> {
        let _ = wasapi::initialize_mta();
        let default = wasapi::get_default_device(&Direction::Capture)
            .map_err(|e| format!("default capture device: {e}"))?;
        let default_name = default
            .get_friendlyname()
            .unwrap_or_else(|_| "Default".to_string());
        let coll = DeviceCollection::new(&Direction::Capture)
            .map_err(|e| format!("enumerate capture devices: {e}"))?;
        let count = coll
            .get_nbr_devices()
            .map_err(|e| format!("count capture devices: {e}"))?;
        let mut names = Vec::with_capacity(count as usize);
        for idx in 0..count {
            if let Ok(dev) = coll.get_device_at_index(idx) {
                if let Ok(name) = dev.get_friendlyname() {
                    names.push(name);
                }
            }
        }
        Ok((default_name, names))
    }

    /// Start the WASAPI capture thread and the 2 MiB MMCSS Pro Audio inference/render thread.
    pub fn spawn_audio_pipeline(
        state: Arc<SharedEngineState>,
        denoiser: Denoiser,
    ) -> Result<(std::thread::JoinHandle<()>, std::thread::JoinHandle<()>), String> {
        let queue = Arc::new(HopQueue::new());
        let queue_cap = Arc::clone(&queue);
        let state_cap = Arc::clone(&state);

        let cap_handle = std::thread::Builder::new()
            .name("hushmic-wasapi-cap".into())
            .spawn(move || {
                if let Err(e) = run_capture_loop(&state_cap, &queue_cap) {
                    eprintln!("hushmic-win: capture loop stopped: {e}");
                }
                queue_cap.close();
            })
            .map_err(|e| format!("spawn capture thread: {e}"))?;

        let state_inf = Arc::clone(&state);
        // 2 MiB stack as required by the plan and denoiser stack scratch guidance.
        let inf_handle = std::thread::Builder::new()
            .name("hushmic-infer".into())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                if let Err(e) = run_infer_and_render_loop(&state_inf, &queue, denoiser) {
                    eprintln!("hushmic-win: inference/render loop stopped: {e}");
                }
            })
            .map_err(|e| format!("spawn inference thread: {e}"))?;

        Ok((cap_handle, inf_handle))
    }

    fn configured_mic(state: &SharedEngineState) -> Option<String> {
        state
            .config
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .mic
            .clone()
    }

    fn run_capture_loop(state: &SharedEngineState, queue: &HopQueue) -> Result<(), String> {
        let _ = wasapi::initialize_mta();
        while !state.is_stopped() {
            let mic_query = configured_mic(state);
            if let Err(err) = capture_one_device(state, queue, mic_query.as_deref()) {
                eprintln!("hushmic-win: capture device: {err}");
                std::thread::sleep(Duration::from_millis(400));
            }
        }
        wasapi::deinitialize();
        Ok(())
    }

    /// Capture from one endpoint until the configured microphone changes or the engine stops.
    fn capture_one_device(
        state: &SharedEngineState,
        queue: &HopQueue,
        mic_query: Option<&str>,
    ) -> Result<(), String> {
        let (device, dev_name) = pick_device(Direction::Capture, mic_query)?;
        state
            .snapshot
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .capture_device = dev_name;
        queue.clear();

        let mut client = device
            .get_iaudioclient()
            .map_err(|e| format!("capture get_iaudioclient: {e}"))?;
        let mix_fmt = client
            .get_mixformat()
            .map_err(|e| format!("capture get_mixformat: {e}"))?;
        let rate = mix_fmt.get_samplespersec();
        let channels = mix_fmt.get_nchannels() as usize;
        let desired_fmt =
            WaveFormat::new(32, 32, &SampleType::Float, rate as usize, channels, None);
        // 20 ms shared buffer in 100-ns units; event-driven 10 ms periods.
        let mode = StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns: 200_000,
        };
        client
            .initialize_client(&desired_fmt, &Direction::Capture, &mode)
            .map_err(|e| format!("capture initialize_client: {e}"))?;
        let event = client
            .set_get_eventhandle()
            .map_err(|e| format!("capture set_get_eventhandle: {e}"))?;
        let capture = client
            .get_audiocaptureclient()
            .map_err(|e| format!("get_audiocaptureclient: {e}"))?;

        let mut resampler = Resampler48k::new(rate)?;
        let mut raw_bytes = VecDeque::<u8>::with_capacity(rate as usize * channels * 4 / 10);
        let mut interleaved = Vec::<f32>::new();
        let mut mono = Vec::<f32>::new();
        let mut mono_48k = Vec::<f32>::new();

        client
            .start_stream()
            .map_err(|e| format!("capture start_stream: {e}"))?;

        while !state.is_stopped() && configured_mic(state).as_deref() == mic_query {
            let _ = event.wait_for_event(100);
            if state.is_stopped() || configured_mic(state).as_deref() != mic_query {
                break;
            }
            raw_bytes.clear();
            if capture.read_from_device_to_deque(&mut raw_bytes).is_err() {
                continue;
            }
            if raw_bytes.len() < 4 * channels {
                continue;
            }
            let sample_count = raw_bytes.len() / 4;
            interleaved.clear();
            interleaved.reserve(sample_count);
            while raw_bytes.len() >= 4 {
                let b0 = raw_bytes.pop_front().unwrap_or(0);
                let b1 = raw_bytes.pop_front().unwrap_or(0);
                let b2 = raw_bytes.pop_front().unwrap_or(0);
                let b3 = raw_bytes.pop_front().unwrap_or(0);
                interleaved.push(f32::from_le_bytes([b0, b1, b2, b3]));
            }
            downmix_to_mono(&interleaved, channels, &mut mono);
            if resampler.process(&mono, &mut mono_48k).is_ok() && !mono_48k.is_empty() {
                queue.push_samples(&mono_48k);
            }
        }

        let _ = client.stop_stream();
        Ok(())
    }

    fn run_infer_and_render_loop(
        state: &SharedEngineState,
        queue: &HopQueue,
        denoiser: Denoiser,
    ) -> Result<(), String> {
        let _ = wasapi::initialize_mta();
        let _mmcss = MmcssGuard::enter_pro_audio();

        let render_query = state
            .config
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .render_device
            .clone();
        let (device, dev_name) = pick_device(Direction::Render, Some(&render_query))?;
        state
            .snapshot
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .render_device = dev_name;

        let mut client = device
            .get_iaudioclient()
            .map_err(|e| format!("render get_iaudioclient: {e}"))?;
        let mix_fmt = client
            .get_mixformat()
            .map_err(|e| format!("render get_mixformat: {e}"))?;
        let render_channels = (mix_fmt.get_nchannels() as usize).max(1);
        let desired_fmt = WaveFormat::new(
            32,
            32,
            &SampleType::Float,
            SAMPLE_RATE as usize,
            render_channels,
            None,
        );
        let mode = StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns: 200_000,
        };
        client
            .initialize_client(&desired_fmt, &Direction::Render, &mode)
            .map_err(|e| format!("render initialize_client: {e}"))?;
        let _event = client
            .set_get_eventhandle()
            .map_err(|e| format!("render set_get_eventhandle: {e}"))?;
        let render = client
            .get_audiorenderclient()
            .map_err(|e| format!("get_audiorenderclient: {e}"))?;

        let mut stream = StreamDenoiser::new(denoiser);
        let mut late_guard = LateHopGuard::new();
        let mut signal_gate = SignalGate::new();
        let mut call_watch = CallWatch::new();
        let mut hop_in = [0.0f32; HOP];
        let mut hop_out = [0.0f32; HOP];
        let mut last_out = [0.0f32; HOP];
        let mut interleaved = Vec::<f32>::with_capacity(HOP * render_channels);
        let mut byte_deque = VecDeque::<u8>::with_capacity(HOP * render_channels * 8);
        let mut arm_reset = false;
        let mut was_running = false;
        let mut last_log: Option<(bool, &'static str)> = None;

        client
            .start_stream()
            .map_err(|e| format!("render start_stream: {e}"))?;

        while !state.is_stopped() {
            let Some(got_hop) = queue.pop_hop(&mut hop_in, Duration::from_millis(50)) else {
                break;
            };
            if !got_hop {
                continue;
            }

            let (enabled, run_mode, attn_limit, gate_mode, render_device) = {
                let cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
                (
                    cfg.enabled,
                    cfg.mode,
                    cfg.attn_limit,
                    cfg.npu_gate,
                    cfg.render_device.clone(),
                )
            };

            let signal_open = matches!(gate_mode, NpuGate::Auto | NpuGate::Signal)
                && signal_gate.observe(&hop_in);
            let call_active = if matches!(gate_mode, NpuGate::Auto | NpuGate::Call) {
                call_watch.poll(&render_device, gate_mode)
            } else {
                None
            };
            let decision = decide_hop(gate_mode, enabled, run_mode, call_active, signal_open);

            if last_log != Some((decision.run_npu, decision.reason)) {
                last_log = Some((decision.run_npu, decision.reason));
                if decision.run_npu && decision.reason == "call" {
                    match call_watch.client.as_deref() {
                        Some(name) => eprintln!("hushmic-win: NPU running (call: {name})"),
                        None => eprintln!("hushmic-win: NPU running (call)"),
                    }
                } else if decision.run_npu {
                    eprintln!("hushmic-win: NPU running ({})", decision.reason);
                } else if decision.reason == "idle" {
                    let waiting = match gate_mode {
                        NpuGate::Call => "a call",
                        NpuGate::Signal => "microphone signal",
                        _ => "a call or microphone signal",
                    };
                    eprintln!("hushmic-win: NPU idle — waiting for {waiting}");
                } else {
                    eprintln!("hushmic-win: NPU off ({})", decision.reason);
                }
            }

            let mut elapsed_ms = 0.0;
            if decision.run_npu {
                let target_mode = run_mode.to_denoiser_mode();
                if stream.denoiser().mode() != target_mode {
                    stream.denoiser_mut().set_mode(target_mode);
                }
                if (stream.denoiser().attenuation_limit_db() - attn_limit).abs() > 0.01 {
                    stream.denoiser_mut().set_attenuation_limit_db(attn_limit);
                }
                if arm_reset {
                    // Recurrent state from the previous session would color the new one.
                    stream.reset();
                    late_guard.reset();
                    arm_reset = false;
                }
                let t0 = Instant::now();
                let cleaned = stream.process(&hop_in);
                elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if cleaned.len() >= HOP {
                    hop_out.copy_from_slice(&cleaned[..HOP]);
                } else {
                    hop_out = [0.0; HOP];
                }
                let outcome = late_guard.observe_hop(&hop_in, &mut hop_out, elapsed_ms);
                if outcome == LateOutcome::CrossfadedToBypass {
                    eprintln!(
                        "hushmic-win: consecutive late hop ({elapsed_ms:.2} ms > {NPU_HOP_GATE_MS:.1} ms); crossfaded hop to bypass"
                    );
                }
            } else {
                arm_reset = true;
                if decision.passthrough {
                    let raw = late_guard.delay_raw(&hop_in);
                    if was_running {
                        let inv = 1.0 / HOP as f32;
                        for i in 0..HOP {
                            let alpha = i as f32 * inv;
                            hop_out[i] = last_out[i] * (1.0 - alpha) + raw[i] * alpha;
                        }
                    } else {
                        hop_out = raw;
                    }
                } else if was_running {
                    fade_out(&last_out, &mut hop_out);
                    let _ = late_guard.delay_raw(&hop_in);
                } else {
                    hop_out = [0.0; HOP];
                    let _ = late_guard.delay_raw(&hop_in);
                }
            }
            was_running = decision.run_npu;
            last_out = hop_out;

            {
                let mut snap = state.snapshot.lock().unwrap_or_else(|p| p.into_inner());
                snap.last_hop_ms = elapsed_ms;
                snap.late_hops = late_guard.total_late();
                snap.bypass_crossfades = late_guard.total_crossfades();
                snap.npu_gate = gate_mode.as_str();
                snap.npu_active = decision.run_npu;
                snap.npu_activity = decision.reason;
                if decision.reason == "call" {
                    let client = call_watch.client.as_deref().unwrap_or("");
                    if snap.npu_client != client {
                        snap.npu_client = client.to_string();
                    }
                } else if !snap.npu_client.is_empty() {
                    snap.npu_client.clear();
                }
                snap.enabled = enabled;
                snap.mode = run_mode;
                snap.attn_limit_db = attn_limit;
            }

            mono_to_interleaved(&hop_out, render_channels, &mut interleaved);
            for &sample in &interleaved {
                byte_deque.extend(sample.to_le_bytes());
            }

            if let Ok(avail_frames) = client.get_available_space_in_frames() {
                let queued_frames = byte_deque.len() / (render_channels * 4);
                let to_write = (avail_frames as usize).min(queued_frames);
                if to_write > 0 {
                    let _ = render.write_to_device_from_deque(to_write, &mut byte_deque, None);
                }
            }
        }

        let _ = client.stop_stream();
        wasapi::deinitialize();
        Ok(())
    }

    /// Polls the virtual-microphone capture endpoint a few times a second.
    struct CallWatch {
        last_poll: Option<Instant>,
        cached: Option<bool>,
        client: Option<String>,
        unknown_logged: bool,
    }

    impl CallWatch {
        fn new() -> Self {
            Self {
                last_poll: None,
                cached: None,
                client: None,
                unknown_logged: false,
            }
        }

        /// `Some(true)` when another process is recording the paired capture endpoint.
        /// `None` when that endpoint cannot be seen; the signal gate still applies in `auto`.
        fn poll(&mut self, render_device: &str, gate: NpuGate) -> Option<bool> {
            let due = match self.last_poll {
                Some(t) => t.elapsed() >= Duration::from_millis(200),
                None => true,
            };
            if !due {
                return self.cached;
            }
            self.last_poll = Some(Instant::now());
            match capture_clients(render_device) {
                Ok(names) if names.is_empty() => {
                    self.cached = Some(false);
                    self.client = None;
                    self.unknown_logged = false;
                    Some(false)
                }
                Ok(names) => {
                    self.cached = Some(true);
                    self.client = Some(names.join(", "));
                    self.unknown_logged = false;
                    Some(true)
                }
                Err(err) => {
                    if !self.unknown_logged {
                        if gate == NpuGate::Call {
                            eprintln!(
                                "hushmic-win: call detection unavailable ({err}); NPU stays idle"
                            );
                        } else {
                            eprintln!(
                                "hushmic-win: call detection unavailable ({err}); NPU wakes on microphone signal"
                            );
                        }
                        self.unknown_logged = true;
                    }
                    self.cached = None;
                    self.client = None;
                    None
                }
            }
        }
    }

    /// Names of other processes with an active capture session on the endpoint
    /// paired with `render_device` (`CABLE Input` -> `CABLE Output`).
    fn capture_clients(render_device: &str) -> Result<Vec<String>, String> {
        let query = super::paired_capture_query(render_device).ok_or_else(|| {
            format!("'{render_device}' has no Input/Output pair to watch for a call")
        })?;
        let id = capture_device_id(&query)?;
        active_capture_clients(&id)
    }

    fn capture_device_id(query: &str) -> Result<String, String> {
        let coll = DeviceCollection::new(&Direction::Capture)
            .map_err(|e| format!("enumerate capture devices: {e}"))?;
        let count = coll
            .get_nbr_devices()
            .map_err(|e| format!("count capture devices: {e}"))?;
        let mut names = Vec::with_capacity(count as usize);
        let mut devices = Vec::with_capacity(count as usize);
        for idx in 0..count {
            if let Ok(dev) = coll.get_device_at_index(idx) {
                let name = dev.get_friendlyname().unwrap_or_default();
                names.push(name);
                devices.push(dev);
            }
        }
        let idx = super::match_device_name(&names, Some(query))
            .ok_or_else(|| format!("no capture device matching '{query}'"))?;
        devices[idx]
            .get_id()
            .map_err(|e| format!("capture device id: {e}"))
    }

    fn active_capture_clients(device_id: &str) -> Result<Vec<String>, String> {
        use windows::core::{Interface, PCWSTR};
        use windows::Win32::Foundation::S_OK;
        use windows::Win32::Media::Audio::{
            AudioSessionStateActive, IAudioSessionControl2, IAudioSessionManager2,
            IMMDeviceEnumerator, MMDeviceEnumerator,
        };
        use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_ALL};

        let enumerator: IMMDeviceEnumerator =
            unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
                .map_err(|e| format!("audio enumerator: {e}"))?;
        let wide: Vec<u16> = device_id.encode_utf16().chain(std::iter::once(0)).collect();
        let device = unsafe { enumerator.GetDevice(PCWSTR(wide.as_ptr())) }
            .map_err(|e| format!("open capture endpoint: {e}"))?;
        let manager: IAudioSessionManager2 = unsafe { device.Activate(CLSCTX_ALL, None) }
            .map_err(|e| format!("session manager: {e}"))?;
        let sessions = unsafe { manager.GetSessionEnumerator() }
            .map_err(|e| format!("session enumerator: {e}"))?;
        let count = unsafe { sessions.GetCount() }.map_err(|e| format!("session count: {e}"))?;
        let self_pid = std::process::id();
        let mut names = Vec::new();
        for index in 0..count {
            let Ok(control) = (unsafe { sessions.GetSession(index) }) else {
                continue;
            };
            let Ok(state) = (unsafe { control.GetState() }) else {
                continue;
            };
            if state != AudioSessionStateActive {
                continue;
            }
            let Ok(control2) = control.cast::<IAudioSessionControl2>() else {
                continue;
            };
            if unsafe { control2.IsSystemSoundsSession() } == S_OK {
                continue;
            }
            let Ok(pid) = (unsafe { control2.GetProcessId() }) else {
                continue;
            };
            if pid == 0 || pid == self_pid {
                continue;
            }
            let name = process_image_name(pid);
            if !names.contains(&name) {
                names.push(name);
            }
            if names.len() == 3 {
                break;
            }
        }
        Ok(names)
    }

    fn process_image_name(pid: u32) -> String {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{
            OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
            PROCESS_QUERY_LIMITED_INFORMATION,
        };
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                return format!("pid {pid}");
            }
            let mut buf = [0u16; 1024];
            let mut len = buf.len() as u32;
            let ok =
                QueryFullProcessImageNameW(handle, PROCESS_NAME_WIN32, buf.as_mut_ptr(), &mut len);
            let _ = CloseHandle(handle);
            if ok == 0 || len == 0 {
                return format!("pid {pid}");
            }
            let path = String::from_utf16_lossy(&buf[..len as usize]);
            path.rsplit(['\\', '/']).next().unwrap_or(&path).to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_stereo_and_mono_round_trip() {
        let stereo = [0.5f32, -0.5, 1.0, 0.0, 0.25, 0.75];
        let mut mono = Vec::new();
        downmix_to_mono(&stereo, 2, &mut mono);
        assert_eq!(mono, vec![0.0, 0.5, 0.5]);

        let mut expanded = Vec::new();
        mono_to_interleaved(&mono, 2, &mut expanded);
        assert_eq!(expanded, vec![0.0, 0.0, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn resampler_48k_is_passthrough_at_48khz_and_resamples_44100() {
        let mut r48 = Resampler48k::new(48_000).unwrap();
        assert!(r48.is_passthrough());
        let mut out = Vec::new();
        r48.process(&[0.1, 0.2, 0.3], &mut out).unwrap();
        assert_eq!(out, vec![0.1, 0.2, 0.3]);

        let mut r44 = Resampler48k::new(44_100).unwrap();
        assert!(!r44.is_passthrough());
        let input = vec![0.0f32; 4410]; // 100 ms at 44.1 kHz
        r44.process(&input, &mut out).unwrap();
        assert!(!out.is_empty());
    }

    #[test]
    fn device_name_matcher_prefers_exact_then_substring() {
        let devs = vec![
            "Microphone Array (Realtek(R) Audio)".to_string(),
            "CABLE Input (VB-Audio Virtual Cable)".to_string(),
            "Headphones".to_string(),
        ];
        assert_eq!(match_device_name(&devs, Some("CABLE Input")), Some(1));
        assert_eq!(match_device_name(&devs, Some("realtek")), Some(0));
        assert_eq!(match_device_name(&devs, Some("default")), None);
        assert_eq!(match_device_name(&devs, Some("Nonexistent")), None);
    }

    #[test]
    fn hop_queue_slices_and_bounds_samples() {
        let q = HopQueue::new();
        q.push_samples(&vec![1.0f32; HOP + 10]);
        let mut hop = [0.0f32; HOP];
        assert_eq!(
            q.pop_hop(&mut hop, std::time::Duration::from_millis(5)),
            Some(true)
        );
        assert!(hop.iter().all(|&x| x == 1.0));
        q.clear();
        assert_eq!(
            q.pop_hop(&mut hop, std::time::Duration::from_millis(5)),
            Some(false)
        );
    }

    #[test]
    fn late_hop_guard_holds_first_and_crossfades_second() {
        let mut g = LateHopGuard::new();
        let raw = [1.0f32; HOP];
        // Prime the 5-hop delay line so aligned_raw becomes 1.0.
        for _ in 0..MODEL_DELAY_HOPS {
            let mut den = [0.0f32; HOP];
            assert_eq!(g.observe_hop(&raw, &mut den, 2.5), LateOutcome::OnTime);
        }

        // 20 ms is inside the ceiling: denoised output stays.
        let mut den_ok = [0.0f32; HOP];
        assert_eq!(g.observe_hop(&raw, &mut den_ok, 20.0), LateOutcome::OnTime);

        // 1st hop past 28 ms: held in slack, output untouched.
        let mut den1 = [0.0f32; HOP];
        assert_eq!(g.observe_hop(&raw, &mut den1, 29.0), LateOutcome::HeldOnce);
        assert!(den1.iter().all(|&x| x == 0.0));

        // 2nd consecutive late hop: crossfades from denoised (0.0) toward aligned raw (1.0).
        let mut den2 = [0.0f32; HOP];
        assert_eq!(
            g.observe_hop(&raw, &mut den2, 30.0),
            LateOutcome::CrossfadedToBypass
        );
        assert_eq!(den2[0], 0.0);
        assert!(den2[HOP - 1] > 0.99);
        assert_eq!(g.total_crossfades(), 1);
    }

    #[test]
    fn paired_capture_query_rewrites_virtual_cable_input() {
        assert_eq!(
            paired_capture_query("CABLE Input").as_deref(),
            Some("CABLE Output")
        );
        assert_eq!(
            paired_capture_query("CABLE Input (VB-Audio Virtual Cable)").as_deref(),
            Some("CABLE Output (VB-Audio Virtual Cable)")
        );
        assert_eq!(
            paired_capture_query("cable input").as_deref(),
            Some("cable output")
        );
        assert_eq!(paired_capture_query("Speakers (Realtek)"), None);
    }

    #[test]
    fn signal_gate_ignores_room_tone_and_opens_on_speech() {
        let mut gate = SignalGate::new();
        let room = [0.01f32; HOP];
        for _ in 0..SIGNAL_LEARN_HOPS {
            assert!(
                !gate.observe(&room),
                "calibration must not treat a steady room tone as speech"
            );
        }
        for _ in 0..30 {
            assert!(!gate.observe(&room), "steady room tone stays idle");
        }

        let mut quiet = SignalGate::new();
        let zeros = [0.0f32; HOP];
        for _ in 0..SIGNAL_LEARN_HOPS {
            assert!(!quiet.observe(&zeros));
        }
        let speech = [0.08f32; HOP];
        assert!(
            quiet.observe(&speech),
            "speech above the floor opens the gate"
        );
        let mut still_open = true;
        for _ in 0..SIGNAL_HANGOVER_HOPS {
            still_open &= quiet.observe(&zeros);
        }
        assert!(
            still_open,
            "hangover keeps the gate open across a short pause"
        );
        assert!(!quiet.observe(&zeros), "the gate closes after the hangover");
    }

    #[test]
    fn signal_gate_opens_immediately_on_a_loud_hop() {
        let mut gate = SignalGate::new();
        assert!(gate.observe(&[0.2f32; HOP]));
    }

    #[test]
    fn decide_hop_runs_for_call_or_signal_and_skips_when_idle() {
        let idle = decide_hop(NpuGate::Auto, true, RunMode::Suppress, Some(false), false);
        assert!(!idle.run_npu);
        assert!(idle.passthrough);
        assert_eq!(idle.reason, "idle");

        let call = decide_hop(NpuGate::Auto, true, RunMode::Suppress, Some(true), false);
        assert!(call.run_npu);
        assert_eq!(call.reason, "call");

        let signal = decide_hop(NpuGate::Auto, true, RunMode::Suppress, Some(false), true);
        assert!(signal.run_npu);
        assert_eq!(signal.reason, "signal");

        let call_only = decide_hop(NpuGate::Call, true, RunMode::Suppress, Some(false), true);
        assert!(!call_only.run_npu);

        let signal_only = decide_hop(NpuGate::Signal, true, RunMode::Suppress, Some(true), false);
        assert!(!signal_only.run_npu);

        let always = decide_hop(NpuGate::Always, true, RunMode::Suppress, Some(false), false);
        assert!(always.run_npu);
        assert_eq!(always.reason, "always");

        let bypass = decide_hop(NpuGate::Always, true, RunMode::Bypass, Some(true), true);
        assert!(!bypass.run_npu);
        assert!(bypass.passthrough);

        let off = decide_hop(NpuGate::Always, false, RunMode::Suppress, Some(true), true);
        assert!(!off.run_npu);
        assert!(off.passthrough);

        let mute = decide_hop(NpuGate::Always, true, RunMode::Mute, Some(true), true);
        assert!(!mute.run_npu);
        assert!(!mute.passthrough);
    }
}
