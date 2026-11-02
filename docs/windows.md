# HushMic NPU Fork — Windows & Lenovo ThinkPad E16 (Intel NPU)

This document belongs to the **HushMic NPU Fork** of [Fovty/hushmic](https://github.com/Fovty/hushmic). Upstream HushMic is the Linux, CPU-only app. The Windows engine, the OpenVINO NPU backend, the model split, and the call/signal idle gate are fork-only.

`hushmic-win` brings the fork's DPDFNet 48 kHz real-time noise suppression to Windows 11, targeting the **Lenovo ThinkPad E16 Gen 3 (Intel Core Ultra 200H)** and **Gen 2 (Meteor Lake)** NPU via OpenVINO with automatic fallback to ONNX Runtime on the CPU.

## Architecture Overview

1. **WASAPI Capture (CPU):** Captures the selected microphone in shared event-driven mode (10 ms period), downmixes to mono, and resamples to 48 kHz with `rubato` only when the endpoint's mix format is not already 48 kHz.
2. **STFT & Framing (CPU):** `StreamDenoiser` slices incoming audio into 480-sample (10 ms) hops and runs a 960-point analysis STFT (`[1, 1, 481, 2]`, 962 floats).
3. **Spectral Neural Network (Intel NPU or ONNX CPU):**
   - When `openvino_c.dll` and the Intel NPU driver are present, `hushmic-denoiser` (with feature `npu`) compiles the DPDFNet `.onnx` graph for the `NPU` device (`PERFORMANCE_HINT = LATENCY`, `NUM_STREAMS = 1`) and caches the compiled blob in `%LOCALAPPDATA%\HushMic\ov-cache`.
   - When OpenVINO or the `NPU` device is absent, `Denoiser::from_file_npu` records the fallback reason and runs the exact same model on ONNX Runtime CPU (`onnxruntime.dll`).
4. **Masking, iSTFT & Guardrails (CPU):** Applies the deep-filter mask, attenuation limit (`attn_limit`), and mute/bypass ramps on an MMCSS `"Pro Audio"` thread (2 MiB stack). One late hop (> 10 ms) is absorbed by slack; a second consecutive late hop crossfades toward latency-aligned bypass and logs a warning.
5. **WASAPI Render (`CABLE Input`):** Writes cleaned 48 kHz audio to `CABLE Input` (VB-Audio Virtual Cable) so communication apps (Teams, Zoom, Discord, browsers) can select **`CABLE Output`** as their microphone.

---

## Building and Testing

From the repository root:

```powershell
cargo build --release -p hushmic-win
cargo test -p hushmic-win
cargo test -p hushmic-denoiser --features npu
```

This produces `target\release\hushmic-win.exe`.

A setup program with that binary, ONNX Runtime, the OpenVINO NPU runtime, and the DPDFNet models (including the NPU split files) is built by:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build-installer.ps1
```

The script writes `dist\HushMic-Setup-<version>.exe`. It needs [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`winget install JRSoftware.InnoSetup`) plus the local `.build-tools` copies of ONNX Runtime and OpenVINO. The Intel AI Boost driver and VB-Audio Virtual Cable are not inside the setup.

---

## Runtime Files Layout

Place the runtime files beside `hushmic-win.exe` (or configure paths in `%APPDATA%\HushMic\config.toml`):

```text
hushmic-win.exe
onnxruntime.dll                  # ONNX Runtime 1.18+ x64 shared library (CPU fallback)
models\
  dpdfnet8_48khz_hr.onnx         # Quality model (default)
  dpdfnet2_48khz_hr.onnx         # Light model
```

- You can obtain `dpdfnet8_48khz_hr.onnx` and `dpdfnet2_48khz_hr.onnx` from the [HushMic release assets](https://github.com/Fovty/HushMic/releases) or by running `bash scripts/setup-assets.sh` in WSL.
- On the ThinkPad E16, add the OpenVINO Windows runtime `bin\intel64\Release` directory (containing `openvino_c.dll`, `openvino.dll`, and `openvino_intel_npu_plugin.dll`) to `PATH` or copy those DLLs next to `hushmic-win.exe`, or set `OPENVINO_C_PATH` to the full path of `openvino_c.dll`.

---

## CLI & Configuration

`hushmic-win` uses the same command words as the Linux `hushmic` binary over the named pipe `\\.\pipe\hushmic-win`:

```powershell
hushmic-win                         # Start the engine + system tray
hushmic-win --headless              # Start the engine without the tray icon
hushmic-win status                  # Inspect active engine (openvino npu vs onnx), devices, and hop latency
hushmic-win status --json           # JSON status output
hushmic-win mode suppress           # suppress | bypass | mute | off
hushmic-win toggle mute             # Toggle mute on/off
hushmic-win config                  # Print %APPDATA%\HushMic\config.toml
hushmic-win config set model light  # Switch between quality (dpdfnet8) and light (dpdfnet2)
hushmic-win config set attn_limit strong
hushmic-win config set inference npu
hushmic-win config set mic "Realtek"
hushmic-win config set render_device "CABLE Input"
hushmic-win config set npu_gate auto # auto | call | signal | always
hushmic-win config set autostart true
hushmic-win quit                    # Stop the running daemon
```

`status` includes an `npu:` line. `npu: idle (gate auto)` means inference is not being submitted. `npu: call (Teams.exe, gate auto)` means a capture session on the virtual microphone is keeping the NPU busy. `npu: signal (gate auto)` means the physical microphone rose above its noise floor.

### When the NPU runs

Default `npu_gate` is `auto`. The compiled model stays loaded, but a hop is submitted only when another process is actively capturing the recording endpoint paired with the render device (`CABLE Input` is watched as `CABLE Output`), or when the physical microphone is above the room-noise floor. A steady fan does not open that floor after a 500 ms calibration. A loud hop (about −30 dBFS) opens it at once, and inference then holds for 800 ms after the level drops.

While the gate is closed, the neural net is not called and the virtual cable still receives latency-aligned raw microphone audio. Bypass and Off do the same. Mute writes silence. None of those paths submit hops.

During an active call session every hop is submitted, so the recurrent state keeps tracking the noise. `npu_gate = signal` also sleeps through quiet gaps inside a call. The next phrase starts from a reset state, so its first syllables are weaker. `npu_gate = call` ignores the microphone level. `npu_gate = always` is the previous behavior: every hop while the mode is Suppress.

The tray tooltip shows the same state. A browser tab or Discord voice channel that keeps `CABLE Output` open counts as a call until that session goes inactive.

### Offline WAV Denoising (No Virtual Cable Needed)

To verify audio quality on any machine without VB-Cable or a live call:

```powershell
hushmic-win denoise noisy.wav cleaned.wav --engine auto
```

---

## Lenovo ThinkPad E16 Gen 3 (Intel) Qualification Checklist

When moving this repository to the **ThinkPad E16 Gen 3 (Intel Core Ultra)**:

1. **Verify Intel NPU Driver:**
   - Open Windows **Device Manager** -> **Neural processors** and confirm **Intel(R) AI Boost** (or **Intel(R) NPU Accelerator**) is enabled and updated via Windows Update / Lenovo Vantage.
2. **Install OpenVINO Runtime for Windows:**
   - Download the OpenVINO 2025/2026 C/C++ runtime archive for Windows x86_64 and add its `runtime\bin\intel64\Release` folder to `PATH` (or place `openvino_c.dll` next to `hushmic-win.exe`).
3. **Split the model for the NPU (required for correct output):**
   The Intel NPU runs the model's input feature block (`Pow`, `ReduceSum`, `Sqrt`, `Log`, `Div`) wrongly in FP16, so the whole graph on the NPU produces garbage after a few hops (3-8 dB SNR versus ONNX). Cut the model at its first `Conv`: the feature block stays on the CPU (about 0.13 ms per hop) and the convolution/GRU body runs on the NPU.
   ```powershell
   pip install onnx
   python scripts\split-npu-model.py assets\models\dpdfnet2_48khz_hr.onnx
   ```
   Copy the generated `*.front.onnx`, `*.body.onnx` and `*.split.txt` files next to the model. Without them, `--engine npu` compiles the whole graph (wrong output) and `--engine auto` falls back to ONNX CPU.
   Measured on a Core Ultra 5 125U: the `light` model reaches 38-44 dB waveform SNR versus ONNX and uses roughly 4-5x less CPU time per hop than ONNX CPU. The `quality` model body is about 14 ms per hop and stays on the NPU: `auto` leaves the NPU only when the hop median reaches 28 ms.
4. **Run the Hop-Time Qualification Benchmark:**
   ```powershell
   hushmic-win bench --engine npu --model light --hops 1000
   ```
   - Confirm `active_engine: openvino npu` is reported (no `fallback_reason`).
   - Confirm `qual_gate (p99 <= 28.0 ms): PASS`. First run populates `%LOCALAPPDATA%\HushMic\ov-cache`; subsequent starts load the cached NPU blob immediately.
5. **Install VB-Audio Virtual Cable:**
   - Install VB-Cable so Windows has `CABLE Input` (playback) and `CABLE Output` (recording).
   - Start `hushmic-win` and select **`CABLE Output`** as the microphone inside Microsoft Teams, Zoom, Discord, or your browser.
6. **Disable Competing Built-in Suppressors on the Physical Mic:**
   - Turn off **Windows Studio Effects -> Voice Focus** and **Lenovo Vantage Smart Noise Cancellation** on the raw microphone endpoint so two neural suppressors are not stacked in series.
7. **Production Follow-ups on Hardware:**
   - **Signed WaveRT Virtual Microphone Driver:** Replace VB-Cable with an EV-signed WaveRT virtual audio driver so **HushMic** appears directly as a native Windows microphone endpoint.
   - **NNCF INT8 Calibration:** If `dpdfnet8_48khz_hr` in FP16 on a given NPU driver revision exceeds the 28 ms hop ceiling or you want lower package power, quantize the spectral ONNX weights to INT8 with OpenVINO NNCF against speech+noise calibration clips and verify waveform SNR >= 30 dB and DNSMOS drop <= 0.05.
