# Lenovo ThinkPad E16 (Intel NPU) — Project Context & Handoff Guide

This document summarizes everything implemented so far and lists the exact steps required after moving this repository and IDE to the **Lenovo ThinkPad E16 Gen 3 (Intel)** laptop.

---

## 1. Target Hardware & NPU Architecture

| Component | Specification on Lenovo ThinkPad E16 Gen 3 (Intel) |
| --- | --- |
| **Processor** | Intel Core Ultra Series 2 (e.g., Core Ultra 7 255H / 200H, Arrow Lake-H); also compatible with E16 Gen 2 (Meteor Lake) |
| **NPU Hardware** | Integrated **Intel(R) AI Boost** (up to **13 TOPS** INT8 with sparsity; internal hardware compute is **FP16**) |
| **Windows Device Node** | `Device Manager -> Neural processors -> Intel(R) AI Boost` (or `Intel(R) NPU Accelerator`) |
| **Inference Runtime** | OpenVINO C API (`openvino_c.dll` + `openvino_intel_npu_plugin.dll`) targeting device `"NPU"` |
| **CPU Fallback** | ONNX Runtime (`onnxruntime.dll`, API v18+) on single-thread CPU |
| **Model Contract** | DPDFNet 48 kHz mono, 480-sample hop (10 ms), 960-pt STFT (`962` floats `[1, 1, 481, 2]`), model delay 5 hops (50 ms) |
| **Model Sizes** | `dpdfnet8_48khz_hr.onnx` (quality: `state_size = 90,228` floats) and `dpdfnet2_48khz_hr.onnx` (light: `state_size = 56,436` floats) |

### Why Hybrid CPU + NPU Execution Is Used
- **On CPU:** WASAPI capture, stereo-to-mono downmix, conditional resampling to 48 kHz (`rubato`), 960-point STFT/iSTFT (`rustfft`), deep-filter attenuation limiting (`attn_limit`), and mute/bypass gain ramps.
- **On Intel NPU:** Only the recurrent spectral network (`spec` + `state_in` -> `spec_e` + `state_out`) is dispatched to the NPU, using a single synchronous `ov_infer_request_t` with `PERFORMANCE_HINT = LATENCY` and `NUM_STREAMS = 1`.

---

## 2. What Was Implemented in This Repository

### A. OpenVINO NPU Backend in `crates/hushmic-denoiser`
- **[`crates/hushmic-denoiser/Cargo.toml`](crates/hushmic-denoiser/Cargo.toml)**: Added optional feature `npu = ["dep:libloading"]`.
- **[`crates/hushmic-denoiser/src/openvino.rs`](crates/hushmic-denoiser/src/openvino.rs)**:
  - Defines `NpuInference` (`Auto`, `Npu`, `Onnx`) and `NpuEngine` (`Npu`, `Onnx`).
  - Dynamically loads `openvino_c.dll` (checked in order: `OPENVINO_C_PATH` env var, directory next to the executable, then system `PATH`) via `libloading`. No OpenVINO SDK is needed at compile time.
  - Parses ONNX protobuf `metadata_props` (`state_size`, `erb_norm_state_size`, `spec_norm_state_size`, `erb_norm_init`, `spec_norm_init`) directly from the `.onnx` file so `init_state` is seeded without starting an ONNX Runtime session first.
  - Configures `CACHE_DIR` (`%LOCALAPPDATA%\HushMic\ov-cache`) on the OpenVINO core and compiles the model for `"NPU"` with `PERFORMANCE_HINT = LATENCY` and `NUM_STREAMS = 1`.
  - Pre-binds `spec`, `state_in`, `spec_e`, and `state_out` tensors on one reusable `ov_infer_request_t` and runs synchronous per-hop inference.
- **[`crates/hushmic-denoiser/src/model.rs`](crates/hushmic-denoiser/src/model.rs)** & **[`crates/hushmic-denoiser/src/denoiser.rs`](crates/hushmic-denoiser/src/denoiser.rs)**:
  - Added `Denoiser::from_file_npu(model_path, inference, cache_dir)`, `Denoiser::npu_engine()`, and `Denoiser::npu_fallback_reason()`.
  - If `openvino_c.dll` is missing, the `NPU` device is absent, or compilation fails, `Denoiser::from_file_npu` automatically falls back to the CPU ONNX Runtime session and stores the reason in `npu_fallback_reason()`.

### B. Windows Engine & CLI in `crates/hushmic-win`
- **[`crates/hushmic-win/Cargo.toml`](crates/hushmic-win/Cargo.toml)**: New workspace binary crate `hushmic-win`. Compiles cleanly on both Windows (`x86_64-pc-windows-msvc`) and Linux (so Ubuntu CI `cargo clippy --workspace` and `cargo test --workspace` continue to pass).
- **[`crates/hushmic-win/src/audio.rs`](crates/hushmic-win/src/audio.rs)**:
  - `win_wasapi`: Event-driven shared-mode WASAPI capture (`StreamMode::EventsShared { autoconvert: true, buffer_duration_hns: 200_000 }`).
  - `downmix_to_mono` + `Resampler48k`: Converts capture audio to 48 kHz mono (`Resampler48k` is a zero-copy pass-through when the endpoint mix format is already 48 kHz).
  - `HopQueue`: Bounded lock-free-style queue between the capture thread and the inference thread.
  - Inference thread: Spawned with a **2 MiB stack** and registered with Windows MMCSS **`"Pro Audio"`** (`AvSetMmThreadCharacteristicsW`).
  - `LateHopGuard`: Enforces the 10 ms hop budget. One late hop (> 10.0 ms) is absorbed by the 10 ms slack buffer; a second consecutive late hop smoothly crossfades across the 480 samples toward the 5-hop (50 ms) latency-aligned raw input and logs a warning.
  - WASAPI render: Outputs cleaned 48 kHz audio to the configured playback endpoint (default substring match: `"CABLE Input"` for VB-Audio Virtual Cable).
- **[`crates/hushmic-win/src/control.rs`](crates/hushmic-win/src/control.rs)**:
  - Named-pipe server and CLI client on `\\.\pipe\hushmic-win`.
  - Supports the same command grammar as Linux `hushmic`: `status [--json]`, `mode [suppress|bypass|mute|off]`, `toggle mute|bypass`, `config [get KEY|set KEY VALUE] [--json]`, and `quit`.
- **[`crates/hushmic-win/src/config.rs`](crates/hushmic-win/src/config.rs)**:
  - Reads/writes `%APPDATA%\HushMic\config.toml`.
  - Resolves model files (`dpdfnet8_48khz_hr.onnx`, `dpdfnet2_48khz_hr.onnx`) from `HUSHMIC_MODEL_PATH`, `./models/`, `../assets/models/`, or `%APPDATA%\HushMic\models\`.
  - Automatically initializes `onnxruntime.dll` when placed next to `hushmic-win.exe`.
- **[`crates/hushmic-win/src/tray.rs`](crates/hushmic-win/src/tray.rs)** & **[`crates/hushmic-win/src/autostart.rs`](crates/hushmic-win/src/autostart.rs)**:
  - System tray icon and menu (`Suppress noise`, `Bypass (raw mic)`, `Mute`, `Off`, `Start on login`, `Quit HushMic`).
  - Per-user logon autostart in `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` (`HushMic`).
- **[`crates/hushmic-win/src/bench.rs`](crates/hushmic-win/src/bench.rs)**:
  - `hushmic-win bench [--engine auto|npu|onnx] [--model quality|light] [--hops N]`: Runs 20 warmup hops + `N` timed hops, prints `min`, `p50`, `p95`, `p99`, `max`, `mean`, `RTF`, active engine, fallback reason (if any), and checks `qual_gate (p99 <= 8.0 ms): PASS|FAIL`.
  - `hushmic-win denoise <in.wav> <out.wav> [--engine auto|npu|onnx] [--model MODEL]`: Offline WAV denoiser with latency-tail draining and sample-exact alignment.

---

## 3. What Needs to Be Done on the Lenovo ThinkPad E16 Gen 3

### Step 1: Install Rust & Build `hushmic-win.exe` on Windows
On the E16 laptop, ensure Rust (`x86_64-pc-windows-msvc`) and Visual Studio C++ Build Tools are installed, then run in PowerShell:

```powershell
cargo test -p hushmic-denoiser --features npu --lib
cargo test -p hushmic-win
cargo build --release -p hushmic-win
```

Output binary: `target\release\hushmic-win.exe`.

### Step 2: Provision the Models, ONNX Runtime DLL, and OpenVINO NPU Runtime

1. **DPDFNet ONNX Models** (place in `assets\models\` or `target\release\models\`):
   - Either run `bash scripts/setup-assets.sh` in WSL, or use Python (`>= 3.11`) on Windows:
     ```powershell
     pip install dpdfnet==0.5.1
     dpdfnet download dpdfnet8_48khz_hr
     dpdfnet download dpdfnet2_48khz_hr
     ```
   - Expected SHA-256 checksums:
     - `dpdfnet8_48khz_hr.onnx`: `7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631`
     - `dpdfnet2_48khz_hr.onnx`: `7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b`

2. **ONNX Runtime Shared Library (`onnxruntime.dll`)**:
   - Download `onnxruntime-win-x64-1.27.0.zip` from [Microsoft ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.27.0) and copy `lib\onnxruntime.dll` next to `target\release\hushmic-win.exe`.

3. **Intel NPU Driver & OpenVINO Runtime (`openvino_c.dll`)**:
   - In Windows **Device Manager** -> **Neural processors**, confirm **Intel(R) AI Boost** is present and updated via Windows Update / Lenovo Vantage.
   - Download the **OpenVINO 2025/2026 Windows x86_64 archive** from Intel.
   - Add `<openvino_dir>\runtime\bin\intel64\Release` to `PATH` (or set `$env:OPENVINO_C_PATH = "<openvino_dir>\runtime\bin\intel64\Release\openvino_c.dll"`). Ensure `openvino_intel_npu_plugin.dll` is in the same OpenVINO bin folder.

### Step 3: Run On-Device NPU Qualification Benchmark
Run the benchmark first on CPU (`onnx`) and then on the Intel NPU (`npu`):

```powershell
# 1. Baseline on CPU (ONNX Runtime)
.\target\release\hushmic-win.exe bench --engine onnx --model quality --hops 1000

# 2. Target on Intel AI Boost NPU (OpenVINO)
.\target\release\hushmic-win.exe bench --engine npu --model quality --hops 1000

# 3. Light model on Intel NPU
.\target\release\hushmic-win.exe bench --engine npu --model light --hops 1000
```

**What to verify on the E16:**
- `active_engine: openvino npu` (and no `fallback_reason` line).
- `qual_gate (p99 <= 8.0 ms): PASS`.
- Confirm `%LOCALAPPDATA%\HushMic\ov-cache` contains the compiled NPU blob and that a second run of `bench --engine npu` starts without recompilation delay.

### Step 4: Verify Audio Quality & NPU Numerical Parity
Run an offline WAV test through both engines and compare:

```powershell
.\target\release\hushmic-win.exe denoise input_noisy.wav out_onnx.wav --engine onnx
.\target\release\hushmic-win.exe denoise input_noisy.wav out_npu.wav --engine npu
```

**Pass criteria (from the architecture design):**
- Waveform SNR between `out_npu.wav` and `out_onnx.wav` **>= 30 dB**.
- DNSMOS overall score drop **<= 0.05** across the keyboard, fan, and cafe test clips.

### Step 5: Live Microphone + Virtual Cable Setup
1. Install **VB-Audio Virtual Cable** on the E16 (`CABLE Input` playback endpoint and `CABLE Output` recording endpoint).
2. In Windows **Settings -> System -> Sound -> Input -> Microphone Array**:
   - Turn **OFF** **Windows Studio Effects -> Voice Focus**.
   - Turn **OFF** **Lenovo Vantage Smart Noise Cancellation** (so two neural suppressors do not stack in series).
3. Start the engine:
   ```powershell
   .\target\release\hushmic-win.exe
   ```
4. In another terminal, check live telemetry:
   ```powershell
   .\target\release\hushmic-win.exe status
   ```
5. Select **`CABLE Output (VB-Audio Virtual Cable)`** as the microphone in Teams, Zoom, Discord, or browser calls.

---

## 4. Potential Adjustments / Next Steps on the E16 Hardware

1. **OpenVINO NPU Plugin Property Tuning (`crates/hushmic-denoiser/src/openvino.rs`)**:
   - `OpenVinoModel::load` currently passes 2 property pairs (`PERFORMANCE_HINT = LATENCY`, `NUM_STREAMS = 1`) to `ov_core_compile_model_from_file`.
   - If a specific Intel NPU driver revision on the E16 rejects `NUM_STREAMS` on the `"NPU"` plugin (returning an OpenVINO status error in `fallback_reason`), change `num_args` from `4` to `2` in `crates/hushmic-denoiser/src/openvino.rs` (passing only `PERFORMANCE_HINT = LATENCY`) or `0` (default plugin config).
2. **NNCF INT8 Quantization (If Needed for Latency or Power)**:
   - OpenVINO compiles the FP32 `.onnx` graph to FP16 on the NPU by default.
   - If `dpdfnet8_48khz_hr` in FP16 exceeds `p99 <= 8.0 ms` or uses more package power than desired, quantize the ONNX weights to INT8 using OpenVINO `nncf.quantize()` with speech+noise calibration frames so the E16's 13 TOPS INT8 NPU MAC array is used.
3. **Signed WaveRT Virtual Microphone Driver (Optional Production Step)**:
   - To expose a native **`HushMic`** recording endpoint in Windows without VB-Audio Virtual Cable, pair `hushmic-win.exe` with a signed WaveRT ring-buffer virtual audio driver.
