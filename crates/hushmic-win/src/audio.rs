//! Audio pipeline for `hushmic-win`:
//! - Cross-platform DSP helpers: stereo/multi-channel downmix to mono, conditional
//!   `rubato` resampler to 48 kHz, bounded capture-to-inference `HopQueue`,
//!   latency-aligned `LateHopGuard` (1st late hop held in slack, 2nd consecutive
//!   late hop crossfades toward latency-aligned bypass and logs), and device-name
//!   matcher (default render target: `"CABLE Input"`).
//! - Windows WASAPI shared event-driven capture and render threads (`#[cfg(windows)]`),
//!   with the inference thread running `StreamDenoiser` on a 2 MiB stack under
//!   MMCSS `"Pro Audio"`.

use crate::config::{Config, RunMode};
use hushmic_denoiser::{Denoiser, HOP, LATENCY_SAMPLES, SAMPLE_RATE};
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
    /// Hop finished within the 10 ms period.
    OnTime,
    /// First consecutive late hop (> 10 ms): absorbed by the 10 ms slack buffer.
    HeldOnce,
    /// Second (or later) consecutive late hop: output was crossfaded toward
    /// latency-aligned bypass for this hop and should be logged.
    CrossfadedToBypass,
}

/// Tracks hop inference duration against the 10 ms hop budget and maintains a
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

    /// Record a completed hop, updating `denoised_hop` in place if a second
    /// consecutive late hop requires crossfading toward latency-aligned bypass.
    pub fn observe_hop(
        &mut self,
        raw_input_hop: &[f32; HOP],
        denoised_hop: &mut [f32; HOP],
        elapsed_ms: f64,
    ) -> LateOutcome {
        self.raw_delay.push_back(*raw_input_hop);
        let aligned_raw = self.raw_delay.pop_front().unwrap_or(*raw_input_hop);

        if elapsed_ms <= HOP_PERIOD_MS {
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

    fn run_capture_loop(state: &SharedEngineState, queue: &HopQueue) -> Result<(), String> {
        let _ = wasapi::initialize_mta();
        let mic_query = state
            .config
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .mic
            .clone();
        let (device, dev_name) = pick_device(Direction::Capture, mic_query.as_deref())?;
        state
            .snapshot
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .capture_device = dev_name;

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

        while !state.is_stopped() {
            let _ = event.wait_for_event(100);
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
        wasapi::deinitialize();
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
        let mut hop_in = [0.0f32; HOP];
        let mut hop_out = [0.0f32; HOP];
        let mut interleaved = Vec::<f32>::with_capacity(HOP * render_channels);
        let mut byte_deque = VecDeque::<u8>::with_capacity(HOP * render_channels * 8);

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

            // Synchronize mode and attenuation limit with shared config.
            {
                let cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
                let target_mode = if cfg.enabled {
                    cfg.mode.to_denoiser_mode()
                } else {
                    hushmic_denoiser::Mode::Bypass
                };
                if stream.denoiser().mode() != target_mode {
                    stream.denoiser_mut().set_mode(target_mode);
                }
                if (stream.denoiser().attenuation_limit_db() - cfg.attn_limit).abs() > 0.01 {
                    stream
                        .denoiser_mut()
                        .set_attenuation_limit_db(cfg.attn_limit);
                }
            }

            let t0 = Instant::now();
            let cleaned = stream.process(&hop_in);
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if cleaned.len() >= HOP {
                hop_out.copy_from_slice(&cleaned[..HOP]);
            } else {
                hop_out = [0.0; HOP];
            }

            let outcome = late_guard.observe_hop(&hop_in, &mut hop_out, elapsed_ms);
            if outcome == LateOutcome::CrossfadedToBypass {
                eprintln!(
                    "hushmic-win: consecutive late hop ({elapsed_ms:.2} ms > {HOP_PERIOD_MS:.1} ms); crossfaded hop to bypass"
                );
            }

            {
                let mut snap = state.snapshot.lock().unwrap_or_else(|p| p.into_inner());
                snap.last_hop_ms = elapsed_ms;
                snap.late_hops = late_guard.total_late();
                snap.bypass_crossfades = late_guard.total_crossfades();
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

        // 1st late hop (> 10 ms): held in slack, output untouched.
        let mut den1 = [0.0f32; HOP];
        assert_eq!(g.observe_hop(&raw, &mut den1, 11.2), LateOutcome::HeldOnce);
        assert!(den1.iter().all(|&x| x == 0.0));

        // 2nd consecutive late hop: crossfades from denoised (0.0) toward aligned raw (1.0).
        let mut den2 = [0.0f32; HOP];
        assert_eq!(
            g.observe_hop(&raw, &mut den2, 12.0),
            LateOutcome::CrossfadedToBypass
        );
        assert_eq!(den2[0], 0.0);
        assert!(den2[HOP - 1] > 0.99);
        assert_eq!(g.total_crossfades(), 1);
    }
}
