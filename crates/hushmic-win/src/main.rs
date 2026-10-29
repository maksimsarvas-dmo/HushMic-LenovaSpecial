//! `hushmic-win`: Windows noise-suppression engine and CLI for ThinkPad E16 Intel NPU
//! (OpenVINO) with automatic ONNX Runtime CPU fallback.

pub mod audio;
pub mod autostart;
pub mod bench;
pub mod config;
pub mod control;
pub mod tray;

use std::process::ExitCode;

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = raw_args.iter().map(String::as_str).collect();

    match dispatch(&words) {
        Ok(Some(output)) => {
            if !output.is_empty() {
                println!("{output}");
            }
            ExitCode::SUCCESS
        }
        Ok(None) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn dispatch(words: &[&str]) -> Result<Option<String>, String> {
    match words {
        [] | ["run"] => {
            run_daemon(false)?;
            Ok(None)
        }
        ["--headless"] | ["run", "--headless"] => {
            run_daemon(true)?;
            Ok(None)
        }
        ["--help"] | ["-h"] | ["help"] => Ok(Some(control::USAGE.to_string())),
        ["bench", rest @ ..] => bench::run_bench(rest).map(Some),
        ["denoise", rest @ ..] => bench::run_denoise_wav(rest).map(Some),
        _ => control::send_cli_request(words).map(Some),
    }
}

#[cfg(windows)]
fn run_daemon(headless_flag: bool) -> Result<(), String> {
    let cfg = config::Config::load();
    let _ = autostart::apply_autostart(cfg.autostart);
    let (denoiser, engine, fallback_reason) = audio::build_denoiser(&cfg)?;

    if let Some(reason) = &fallback_reason {
        eprintln!(
            "hushmic-win: running on {engine} (NPU fallback reason: {reason}); model={}",
            cfg.model
        );
    } else {
        eprintln!("hushmic-win: running on {engine}; model={}", cfg.model);
    }

    let state = audio::SharedEngineState::new(cfg.clone());
    {
        let mut snap = state.snapshot.lock().unwrap_or_else(|p| p.into_inner());
        snap.engine = engine;
        snap.fallback_reason = fallback_reason;
    }

    let _pipe_thread = control::spawn_pipe_server(std::sync::Arc::clone(&state));
    let (cap_handle, inf_handle) =
        audio::win_wasapi::spawn_audio_pipeline(std::sync::Arc::clone(&state), denoiser)?;

    if !headless_flag && cfg.tray {
        let _ = tray::win_tray::run_tray_loop(std::sync::Arc::clone(&state));
    } else {
        while !state.is_stopped() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    state.request_stop();
    let _ = cap_handle.join();
    let _ = inf_handle.join();
    Ok(())
}

#[cfg(not(windows))]
fn run_daemon(_headless_flag: bool) -> Result<(), String> {
    Err(
        "hushmic-win live WASAPI daemon requires Windows; use `bench`, `denoise`, or `config` on this host"
            .to_string(),
    )
}
