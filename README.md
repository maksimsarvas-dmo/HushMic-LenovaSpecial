<h1 align="center">
  <img src="docs/img/hushmic-logo.png" alt="HushMic" width="120"><br>
  HushMic NPU Fork
</h1>

<p align="center">
  <b>This repository is the HushMic NPU Fork.</b><br>
  A fork of <a href="https://github.com/Fovty/hushmic">Fovty/hushmic</a> that runs DPDFNet on the Intel NPU of the Lenovo ThinkPad E16.<br>
  Upstream HushMic stays a Linux, CPU-only virtual microphone. The NPU engine, the Windows build, and the idle gate live only in this fork.
</p>

<p align="center">
  <a href="https://github.com/Fovty/hushmic"><img src="https://img.shields.io/badge/fork%20of-Fovty%2Fhushmic-orange" alt="Fork of Fovty/hushmic"></a>
  <img src="https://img.shields.io/badge/target-Intel%20NPU%20%28ThinkPad%20E16%29-blue" alt="Intel NPU on ThinkPad E16">
</p>

<p align="center">
  <a href="https://github.com/Fovty/hushmic/releases"><img src="https://img.shields.io/github/v/release/Fovty/hushmic?sort=semver" alt="Release"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue" alt="License"></a>
  <a href="https://github.com/Fovty/hushmic/actions/workflows/release.yml"><img src="https://github.com/Fovty/hushmic/actions/workflows/release.yml/badge.svg" alt="Build"></a>
  <a href="https://ko-fi.com/fovty"><img src="https://img.shields.io/badge/Ko--fi-support-ff5e5b?logo=ko-fi&logoColor=white" alt="Support on Ko-fi"></a>
</p>

The **HushMic NPU Fork** keeps upstream HushMic's virtual microphone, which removes keyboard noise, fans and background chatter in real time, and adds a Windows engine that runs the spectral network on the Intel AI Boost NPU. On Linux the microphone is still a PipeWire source named **HushMic**. On Windows it is the recording side of a VB-Audio virtual cable, fed by this fork's `hushmic-win`. Pick it as the input in Discord, TeamSpeak, Teams, Zoom, OBS or your browser. The model is DPDFNet, which scored above Krisp, DeepFilterNet and RNNoise on the demo clips (see [Why DPDFNet](#why-dpdfnet)). The Linux path, inherited from upstream, adds 80 ms of latency. The fork's Windows path uses the same 48 kHz model at a 10 ms hop (50 ms of model delay) and runs the spectral network on the ThinkPad E16 NPU.

## Demo

Each clip plays the noisy input, then the same audio cleaned by HushMic.

<table>
<tr>
<td align="center"><b>Keyboard</b></td>
<td align="center"><b>Fan / AC hum</b></td>
<td align="center"><b>Café chatter</b></td>
</tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/9cc7abe4-26fc-440f-88da-b86ef33df142" controls width="280"></video></td>
<td><video src="https://github.com/user-attachments/assets/d4b396a7-d980-4e0d-bac2-2a119debd768" controls width="280"></video></td>
<td><video src="https://github.com/user-attachments/assets/f4cd4504-d49a-4a46-95c2-01e56cd2d268" controls width="280"></video></td>
</tr>
</table>

Audio sources and licenses: [docs/demo/ASSETS.md](docs/demo/ASSETS.md). Measurements: [docs/comparison.md](docs/comparison.md).

## Install

The packages and the install script below are **upstream HushMic**, not this fork. They install the Linux CPU build from [Fovty/hushmic](https://github.com/Fovty/hushmic). They do not contain `hushmic-win` or the Intel NPU backend. To run the HushMic NPU Fork on a ThinkPad E16, build `hushmic-win` from this tree. See [Windows on the Lenovo ThinkPad E16](#windows-on-the-lenovo-thinkpad-e16) and [Build from source](#build-from-source).

Requirements for the upstream Linux app: x86-64 with PipeWire and WirePlumber. PulseAudio apps need `pipewire-pulse`. A tray icon needs StatusNotifierItem support, which GNOME gets from the *AppIndicator and KStatusNotifierItem Support* extension. Without a tray, HushMic still runs and the command line controls it.

**Install script** (any distro, system-wide):

```bash
curl -fsSL https://raw.githubusercontent.com/Fovty/hushmic/main/scripts/install.sh | sudo sh
```

**Debian / Ubuntu**:

```bash
curl -fsSLO https://github.com/Fovty/hushmic/releases/latest/download/hushmic_0.11.0-1_amd64.deb
sudo apt install ./hushmic_0.11.0-1_amd64.deb
```

**Arch Linux** (AUR):

```bash
yay -S hushmic-bin
```

**Fedora** (COPR):

```bash
sudo dnf copr enable fovty/hushmic
sudo dnf install hushmic
```

**NixOS** (flake, builds from source):

```bash
nix run github:Fovty/hushmic-nix -- --tray
```

**AppImage**:

```bash
curl -fsSLO https://github.com/Fovty/hushmic/releases/latest/download/hushmic-x86_64.AppImage
chmod +x hushmic-x86_64.AppImage
./hushmic-x86_64.AppImage --tray
```

Ubuntu 22.04, custom prefixes, Nix flags, Flatpak and uninstalling: [docs/install.md](docs/install.md).

## Usage

Start HushMic from your application menu. A tray icon appears and noise suppression is on. Pick your microphone in the tray menu, then choose **HushMic** as the input in your app, or turn on **Set as default microphone** so apps that follow the system default use it.

The menu switches between noise suppression, **bypass** (your raw voice) and **mute** (silence on the virtual mic). Switching is instant and does not reconnect your call. It also holds the model choice (quality or light), the suppression strength, start on login and global shortcuts.

**Test my mic** opens a window with the raw microphone and the cleaned output side by side, and records a 10-second sample you can play back both ways.

<p align="center">
  <img src="docs/img/hushmic-ab-window.png" alt="Live A/B mic test" width="720">
</p>

The same controls work from a terminal, so any hotkey tool can drive them:

```bash
hushmic status              # what the running instance is doing
hushmic mode mute           # suppress | bypass | mute | off
hushmic toggle mute         # one key on, same key off
hushmic config set attn_limit strong
hushmic quit
```

Menu screenshots, keyboard shortcuts, every command and every config key: [docs/usage.md](docs/usage.md).

## Starting at login

Turn on **Start on login** in the tray menu, or run `hushmic config set autostart true`. For a daemon without a tray icon, `hushmic --headless` and the systemd user unit are described in [docs/headless.md](docs/headless.md). Use one of the two, not both.

## Why DPDFNet

Upstream HushMic picked DPDFNet after a three-clip DNSMOS comparison against Krisp, DeepFilterNet 3 and RNNoise. That scoring run is not in this fork. The write-up and the public clips are in [docs/comparison.md](docs/comparison.md). DPDFNet is a DeepFilterNet-lineage model by Ceva ([paper](https://arxiv.org/abs/2512.16420)).

The numbers below were measured in this fork on 2026-11-02. Machine: Lenovo `21MAS04600`, Intel Core Ultra 5 125U (Meteor Lake), Intel AI Boost driver `32.0.100.5540`, OpenVINO 2025.4.0, ONNX Runtime 1.27.0. Warm cache, 1000 hops unless noted. A hop is 10 ms of audio, so a hop that takes longer than 10 ms is not real time. `auto` still keeps the NPU unless the probe median reaches 28 ms.

| Engine | Model | p50 | p99 | Inside a 10 ms hop |
| --- | --- | ---: | ---: | --- |
| ONNX CPU | quality (`dpdfnet8`) | 5.92 ms | 11.15 ms | no |
| OpenVINO NPU | quality | 11.68 ms | 12.68 ms | no |
| OpenVINO NPU | light (`dpdfnet2`) | 4.32 ms | 4.71 ms | yes |
| ONNX CPU (50 hops, auto fallback) | light | 2.31 ms | 3.73 ms | yes |

The whole graph on the NPU is the wrong path. On a 3 s clip its waveform SNR against ONNX was **8.2 dB** (quality) and **3.4 dB** (light). The input feature block (`Pow`, `ReduceSum`, `Sqrt`, `Log`, `Div`) is wrong in FP16. With that block on the CPU and the convolution/GRU body on the NPU, the same light clip scored **44.0 dB** against ONNX (the whole-graph figure on that file was 3.4 dB).

Hybrid path, 200 hops against a full ONNX reference:

| Model | Spectrum SNR | Final state SNR | NPU body p50 | Total p50 / p99 |
| --- | ---: | ---: | ---: | ---: |
| light | 38.9 dB | 28.4 dB | 5.1 ms | 5.3 / 7.3 ms |
| quality | 41.4 dB | 25.0 dB | 13.8 ms | 14.0 / 16.2 ms |

Quiet machine, light model: total p50 5.3 ms, p99 7.3 ms. Under about 50% background load, 1500 hops of the light model cost 7.8–10.1 s of CPU time on ONNX (about 5–6.7 ms/hop) and 1.75–2.2 s on the hybrid path (about 1.2–1.4 ms/hop). Quality's body stays near 14 ms per hop, under the 28 ms ceiling and over the 10 ms hop, so `auto` leaves it on the NPU. Use the light model when the hop itself has to finish in 10 ms.

## How it works

- [`hushmic-denoiser`](crates/hushmic-denoiser) runs the DPDFNet model on 48 kHz mono audio, on ONNX Runtime, the native engine, or (Windows, feature `npu`) OpenVINO on the Intel NPU. It is a plain Rust library you can embed in your own app.
- `dpdfnet-ladspa` wraps it as a LADSPA plugin.
- `hushmic` writes a PipeWire `module-filter-chain` config and runs it as a child process. PipeWire owns the real-time scheduling, so no `setcap` is needed. A watchdog recreates the virtual mic after a PipeWire restart or suspend, and quitting restores your previous default input.
- `hushmic-win` captures the chosen microphone with WASAPI, denoises it, and plays the result into a virtual-cable playback endpoint (default name `CABLE Input`). Communication apps record `CABLE Output`. Full setup: [docs/windows.md](docs/windows.md).

## Windows on the Lenovo ThinkPad E16

This is the part that exists only in the **HushMic NPU Fork**. `hushmic-win` targets Windows 11 on the **Lenovo ThinkPad E16 Gen 3 (Intel Core Ultra 200H)** and **Gen 2 (Meteor Lake)**. The NPU is Intel AI Boost, reached through the OpenVINO C API (`openvino_c.dll` and `openvino_intel_npu_plugin.dll`). If the DLL, the driver, or a timing and accuracy check fails, the same model runs on ONNX Runtime on the CPU and `hushmic-win status` prints the reason.

### Pipeline

1. WASAPI captures the selected microphone in shared event-driven mode, downmixes to mono, and resamples to 48 kHz only when the endpoint is not already 48 kHz.
2. The stream is cut into 480-sample hops (10 ms). A 960-point STFT, the attenuation limit, and mute or bypass ramps stay on the CPU.
3. The recurrent spectral network runs on the NPU when OpenVINO can use it. One late hop is absorbed by slack. A second late hop in a row crossfades that hop toward the latency-aligned raw input.
4. Cleaned 48 kHz audio is written to the render endpoint (default `CABLE Input`). Teams, Zoom, Discord, and browsers use **`CABLE Output`** as their microphone.

The Intel NPU evaluates the model's input feature block (`Pow`, `ReduceSum`, `Sqrt`, `Log`, `Div`) incorrectly in FP16, so a whole-graph compile sounds wrong after a few hops. `scripts/split-npu-model.py` cuts the ONNX file at the first `Conv`. The feature block stays on the CPU (about 0.13 ms per hop) and the convolution and GRU body goes to the NPU. `from_file_npu` uses that hybrid whenever `<model>.front.onnx`, `<model>.body.onnx`, and `<model>.split.txt` sit next to the model. Without those files, `--engine npu` compiles the whole graph and `--engine auto` falls back to the CPU.

On the Core Ultra 5 125U the split light model is the real-time NPU path (p99 7.3 ms quiet, 44.0 dB waveform SNR against ONNX on a 3 s clip). The quality body is about 14 ms per hop and stays on the NPU because that is still under the 28 ms `auto` ceiling. `auto` leaves the NPU when the probe median reaches 28 ms or the waveform SNR against ONNX drops below 30 dB. The table is in [Why DPDFNet](#why-dpdfnet). The first NPU run compiles into `%LOCALAPPDATA%\HushMic\ov-cache`. The next start loads that blob.

Compiled graphs use `PERFORMANCE_HINT = LATENCY` and `NUM_STREAMS = 1`.

### The NPU runs only while someone needs it

A loaded model that is asked for a result on every 10 ms hop keeps the NPU busy for the whole time the app is open, including silence. `hushmic-win` now skips inference until one of these is true:

- **A call.** Some other process has an active capture session on the recording endpoint paired with the render device. `CABLE Input` is watched as `CABLE Output`. That is how Windows reports that Teams, Zoom, Discord, or a browser has opened the virtual microphone. The process name is shown in `hushmic-win status` and on the tray tooltip.
- **Microphone signal.** The physical microphone rises above the room noise. The first 500 ms calibrate that floor, so a steady fan or the mic's own hiss does not wake the network. A loud hop (about −30 dBFS) wakes it immediately. After the sound drops, inference stays on for 800 ms so a short pause between words is not cut. Speech about 12 dB above the tracked floor also opens the gate, and nothing quieter than about −50 dBFS does.

While both are false, the neural net is not called. The virtual cable still receives the microphone, latency-aligned and unprocessed, so a call is not muted if session detection is a moment late. **Bypass** and **Off** do the same. **Mute** writes silence. In every one of those cases the NPU is idle.

While a call session is active, every hop is submitted. The network is recurrent: it has to hear the pauses too, or the start of the next phrase is poorly cleaned. The state is reset only when inference has actually been stopped and then starts again.

The tray tooltip follows the same state (`NPU спит`, `NPU: звонок`, `NPU: сигнал с микрофона`).

`npu_gate` in `%APPDATA%\HushMic\config.toml` selects the policy. The default is `auto`. Changing it on a running engine takes effect on the next hop:

| Value | When the NPU runs |
| --- | --- |
| `auto` | A call is capturing the virtual mic, or the physical mic is above the noise floor. |
| `call` | Only while another process is capturing the virtual mic. |
| `signal` | Only while the physical mic is above the noise floor, including the quiet gaps inside a call. The first syllables after a long pause are weaker, because the recurrent state was reset. |
| `always` | Every hop for as long as the mode is **Suppress**. |

An app that keeps `CABLE Output` open (a Discord voice channel, a browser tab that already called `getUserMedia`) counts as a call, so the NPU stays on until that session goes inactive. `status` names the process.

### Windows commands

```powershell
cargo build --release -p hushmic-win
.\target\release\hushmic-win.exe            # tray + engine
.\target\release\hushmic-win.exe --headless
.\target\release\hushmic-win.exe status     # engine, devices, hop time, npu: idle|call|signal
.\target\release\hushmic-win.exe mode suppress   # suppress | bypass | mute | off
.\target\release\hushmic-win.exe config set model light
.\target\release\hushmic-win.exe config set inference auto   # auto | npu | onnx
.\target\release\hushmic-win.exe config set npu_gate auto    # auto | call | signal | always
.\target\release\hushmic-win.exe config set mic "Realtek"
.\target\release\hushmic-win.exe config set render_device "CABLE Input"
.\target\release\hushmic-win.exe bench --engine npu --model light --hops 1000
.\target\release\hushmic-win.exe denoise noisy.wav cleaned.wav --engine auto
.\target\release\hushmic-win.exe quit
```

The tray menu chooses the microphone, Suppress, Bypass, Mute, Off, and start on login. Place `onnxruntime.dll` and the OpenVINO NPU DLLs next to the exe (or on `PATH`). Put the split model files next to the `.onnx`. Turn off Windows Studio Effects Voice Focus and Lenovo Vantage Smart Noise Cancellation on the physical mic so two neural suppressors are not stacked. Installer, device checklist, and benchmark gates: [docs/windows.md](docs/windows.md).

## FAQ

**Does my audio go anywhere?** No. Nothing is uploaded. On Linux the model runs on the CPU. On Windows the spectral network can run on the Intel NPU, and the CPU path is the fallback. Capture, resampling, the STFT, and the virtual-cable output stay on the machine either way.

**How much latency and CPU?** On Linux, 80 ms of processing plus PipeWire's own buffering; on PipeWire 1.6 and later apps like OBS can compensate for it. On CPUs with AVX2 and FMA (most Intel Core and AMD Ryzen CPUs since 2015) a native engine runs the models in under half the CPU ONNX Runtime needs, and in about a quarter to a third on CPUs with AVX-VNNI. Other CPUs, including many older Pentium, Celeron and Atom chips, use ONNX Runtime, and on slow ones HushMic runs the light model on its own. Details: [docs/troubleshooting.md](docs/troubleshooting.md#latency-and-cpu). On the ThinkPad E16 the Windows engine uses a 10 ms hop and 50 ms of model delay. With `npu_gate` left at `auto`, the NPU is busy during a call or while the microphone is above the room noise, and idle the rest of the time. Hop-time numbers are in [Windows on the Lenovo ThinkPad E16](#windows-on-the-lenovo-thinkpad-e16).

**No tray icon on GNOME?** Install the *AppIndicator and KStatusNotifierItem Support* extension. HushMic keeps running without it.

**Can I use it in EasyEffects?** Yes. EasyEffects can load HushMic's plugin as the DPDFNet model of its Deep Noise Remover, on your microphone and also on what you hear. Install HushMic from a package, then pick the model in EasyEffects. Details: [docs/usage.md](docs/usage.md#using-the-plugin-in-easyeffects).

More: [docs/troubleshooting.md](docs/troubleshooting.md).

## Alternatives

- [NoiseTorch-ng](https://github.com/noisetorch/NoiseTorch): RNNoise-based virtual mic, needs `setcap`.
- [EasyEffects](https://github.com/wwmm/easyeffects): full PipeWire effects suite with an RNNoise denoiser. Its Deep Noise Remover can also run DPDFNet through HushMic's plugin.
- [noise-suppression-for-voice](https://github.com/werman/noise-suppression-for-voice): RNNoise LADSPA/VST plugin, wired by hand.
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet): the model family HushMic builds on, with its own LADSPA plugin.

## Build from source

Clone **this** repository, the HushMic NPU Fork. Cloning [Fovty/hushmic](https://github.com/Fovty/hushmic) gives upstream Linux only, without `hushmic-win` or the NPU backend.

```bash
git clone https://github.com/maksimsarvas-dmo/HushMic-LenovaSpecial
cd HushMic-LenovaSpecial
./scripts/setup-assets.sh        # fetch the DPDFNet models, native weights and ONNX Runtime
cargo build --release            # upstream Linux binary, still built from this fork
cargo build --release -p hushmic-win   # this fork's Windows NPU engine
```

`cargo build --release` produces `target/release/hushmic` and `target/release/libdpdfnet_ladspa.so`. `scripts/install.sh` shows the install layout; `crates/dpdfnet-ladspa/examples/run-filter-chain.md` shows how to load the plugin by hand.

On Windows, from the same checkout of the fork:

```powershell
cargo build --release -p hushmic-win
python scripts\split-npu-model.py assets\models\dpdfnet2_48khz_hr.onnx
python scripts\split-npu-model.py assets\models\dpdfnet8_48khz_hr.onnx
```

That produces `target\release\hushmic-win.exe`. Copy the `.onnx` files, the generated `.front.onnx`, `.body.onnx`, and `.split.txt` files, `onnxruntime.dll`, and the OpenVINO NPU runtime next to it. The rest of the Windows layout is in [docs/windows.md](docs/windows.md).

## Contributing

Translations happen on [Weblate](https://hosted.weblate.org/projects/hushmic/), see [TRANSLATING.md](TRANSLATING.md). If HushMic is useful to you, a coffee on [Ko-fi](https://ko-fi.com/fovty) helps keep it maintained.

## License

MIT ([LICENSE-MIT](LICENSE-MIT)) or Apache-2.0 ([LICENSE-APACHE](LICENSE-APACHE)), at your option.

## Credits

[DPDFNet](https://github.com/ceva-ip/DPDFNet) by Ceva (Apache-2.0) is the speech-enhancement model; the native engine is its authors' C runtime, included unmodified in [crates/hushmic-denoiser/native](crates/hushmic-denoiser/native) under the same license. The real-time plugin follows the architecture of [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet). Built with [PipeWire](https://pipewire.org), [ort](https://github.com/pykeio/ort), [rustfft](https://github.com/ejmahler/RustFFT) and [ksni](https://github.com/iovxw/ksni). The Windows engine uses WASAPI and [OpenVINO](https://github.com/openvinotoolkit/openvino) for the Intel NPU. Demo voice: "After Love" by Sara Teasdale, read by a LibriVox volunteer (public domain); keyboard by C40115 and fan hum by Gravity Sound (CC BY 4.0); café ambience public domain. Details in [docs/demo/ASSETS.md](docs/demo/ASSETS.md).
