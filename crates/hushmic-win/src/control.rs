//! Named-pipe control protocol (`\\.\pipe\hushmic-win`) and CLI request parser,
//! using the same command words as `crates/hushmic/src/control.rs`:
//! `status`, `mode`, `toggle`, `quit`, and `config get/set`.

use crate::audio::SharedEngineState;
use crate::autostart::apply_autostart;
use crate::config::{Config, RunMode};
use serde::Serialize;
#[cfg(windows)]
use std::sync::Arc;

pub const PIPE_NAME: &str = r"\\.\pipe\hushmic-win";

pub const USAGE: &str =
    "usage: hushmic-win [run] | status [--json] | mode [suppress|bypass|mute|off] | \
toggle mute|bypass | quit | config [get KEY|set KEY VALUE] [--json] | \
bench [--engine auto|npu|onnx] [--model MODEL] [--hops N] | \
denoise <in.wav> <out.wav> [--engine auto|npu|onnx] [--model MODEL]";

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Request {
    Status { json: bool },
    GetMode,
    SetMode(Option<RunMode>),
    Toggle(RunMode),
    Quit,
    ConfigGet { key: Option<String>, json: bool },
    ConfigSet { key: String, value: String },
}

/// Parse CLI/pipe words into a [`Request`], matching `crates/hushmic/src/control.rs`.
pub fn parse_request(words: &[&str]) -> Result<Request, String> {
    match words {
        ["status"] => Ok(Request::Status { json: false }),
        ["status", "--json"] => Ok(Request::Status { json: true }),
        ["quit"] => Ok(Request::Quit),
        ["config"] => Ok(Request::ConfigGet {
            key: None,
            json: false,
        }),
        ["config", "--json"] => Ok(Request::ConfigGet {
            key: None,
            json: true,
        }),
        ["config", "get", key] => Ok(Request::ConfigGet {
            key: Some((*key).to_string()),
            json: false,
        }),
        ["config", "get", key, "--json"] => Ok(Request::ConfigGet {
            key: Some((*key).to_string()),
            json: true,
        }),
        ["config", "set", key, rest @ ..] if !rest.is_empty() => Ok(Request::ConfigSet {
            key: (*key).to_string(),
            value: rest.join(" "),
        }),
        ["mode"] => Ok(Request::GetMode),
        ["mode", state] => match *state {
            "suppress" => Ok(Request::SetMode(Some(RunMode::Suppress))),
            "bypass" => Ok(Request::SetMode(Some(RunMode::Bypass))),
            "mute" => Ok(Request::SetMode(Some(RunMode::Mute))),
            "off" => Ok(Request::SetMode(None)),
            other => Err(format!("unknown mode '{other}'\n{USAGE}")),
        },
        ["toggle", which] => match *which {
            "mute" => Ok(Request::Toggle(RunMode::Mute)),
            "bypass" => Ok(Request::Toggle(RunMode::Bypass)),
            other => Err(format!("cannot toggle '{other}' (mute|bypass)\n{USAGE}")),
        },
        _ => Err(USAGE.to_string()),
    }
}

#[derive(Serialize)]
struct StatusJson<'a> {
    mode: &'a str,
    enabled: bool,
    engine: &'a str,
    fallback_reason: Option<&'a str>,
    model: &'a str,
    attn_limit_db: f32,
    capture_device: &'a str,
    render_device: &'a str,
    last_hop_ms: f64,
    late_hops: u64,
    bypass_crossfades: u64,
}

/// Execute a [`Request`] against the running engine state.
pub fn handle_request(state: &SharedEngineState, req: Request) -> Result<String, String> {
    match req {
        Request::Status { json } => {
            let snap = state.snapshot.lock().unwrap_or_else(|p| p.into_inner());
            let mode_str = if snap.enabled {
                snap.mode.as_str()
            } else {
                "off"
            };
            if json {
                let payload = StatusJson {
                    mode: mode_str,
                    enabled: snap.enabled,
                    engine: &snap.engine,
                    fallback_reason: snap.fallback_reason.as_deref(),
                    model: &snap.model,
                    attn_limit_db: snap.attn_limit_db,
                    capture_device: &snap.capture_device,
                    render_device: &snap.render_device,
                    last_hop_ms: snap.last_hop_ms,
                    late_hops: snap.late_hops,
                    bypass_crossfades: snap.bypass_crossfades,
                };
                serde_json::to_string(&payload).map_err(|e| e.to_string())
            } else {
                let mut out = format!(
                    "mode: {mode_str}\nengine: {}\nmodel: {}\nattn_limit: {} dB\ncapture: {}\nrender: {}\nlast_hop: {:.2} ms (late: {}, crossfades: {})",
                    snap.engine,
                    snap.model,
                    snap.attn_limit_db,
                    snap.capture_device,
                    snap.render_device,
                    snap.last_hop_ms,
                    snap.late_hops,
                    snap.bypass_crossfades
                );
                if let Some(reason) = &snap.fallback_reason {
                    out.push_str(&format!("\nfallback_reason: {reason}"));
                }
                Ok(out)
            }
        }
        Request::GetMode => {
            let cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
            Ok(if cfg.enabled {
                cfg.mode.as_str().to_string()
            } else {
                "off".to_string()
            })
        }
        Request::SetMode(target) => {
            let mut cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
            let mut snap = state.snapshot.lock().unwrap_or_else(|p| p.into_inner());
            match target {
                Some(m) => {
                    cfg.enabled = true;
                    cfg.mode = m;
                    snap.enabled = true;
                    snap.mode = m;
                }
                None => {
                    cfg.enabled = false;
                    snap.enabled = false;
                }
            }
            let _ = cfg.save();
            Ok(if cfg.enabled {
                cfg.mode.as_str().to_string()
            } else {
                "off".to_string()
            })
        }
        Request::Toggle(which) => {
            let mut cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
            let mut snap = state.snapshot.lock().unwrap_or_else(|p| p.into_inner());
            let next = if cfg.enabled && cfg.mode == which {
                RunMode::Suppress
            } else {
                which
            };
            cfg.enabled = true;
            cfg.mode = next;
            snap.enabled = true;
            snap.mode = next;
            let _ = cfg.save();
            Ok(next.as_str().to_string())
        }
        Request::Quit => {
            state.request_stop();
            Ok("stopping".to_string())
        }
        Request::ConfigGet { key, json } => {
            let cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
            format_config_get(&cfg, key.as_deref(), json)
        }
        Request::ConfigSet { key, value } => {
            let mut cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
            cfg.set_key(&key, &value)?;
            if key == "autostart" {
                let _ = apply_autostart(cfg.autostart);
            }
            {
                let mut snap = state.snapshot.lock().unwrap_or_else(|p| p.into_inner());
                snap.enabled = cfg.enabled;
                snap.mode = cfg.mode;
                snap.model = cfg.model.clone();
                snap.attn_limit_db = cfg.attn_limit;
            }
            cfg.save()?;
            cfg.get_key(&key)
        }
    }
}

pub fn format_config_get(cfg: &Config, key: Option<&str>, json: bool) -> Result<String, String> {
    match key {
        Some(k) => {
            let val = cfg.get_key(k)?;
            if json {
                let mut map = serde_json::Map::new();
                map.insert(k.to_string(), serde_json::Value::String(val));
                serde_json::to_string(&map).map_err(|e| e.to_string())
            } else {
                Ok(val)
            }
        }
        None => {
            if json {
                serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())
            } else {
                toml::to_string_pretty(cfg).map_err(|e| e.to_string())
            }
        }
    }
}

/// Handle `config` subcommands directly against `config.toml` when the daemon is not running.
pub fn handle_offline_config(req: Request) -> Result<String, String> {
    let mut cfg = Config::load();
    match req {
        Request::ConfigGet { key, json } => format_config_get(&cfg, key.as_deref(), json),
        Request::ConfigSet { key, value } => {
            cfg.set_key(&key, &value)?;
            if key == "autostart" {
                let _ = apply_autostart(cfg.autostart);
            }
            cfg.save()?;
            cfg.get_key(&key)
        }
        _ => Err("hushmic-win daemon is not running".to_string()),
    }
}

/// Encode a single-line request for the named pipe.
pub fn serialize_line(words: &[&str]) -> String {
    format!("{}\n", words.join(" "))
}

/// Execute a single line received over the named pipe.
pub fn process_pipe_line(state: &SharedEngineState, line: &str) -> String {
    let words: Vec<&str> = line.split_whitespace().collect();
    match parse_request(&words) {
        Ok(req) => match handle_request(state, req) {
            Ok(resp) => format!("OK\n{resp}\n"),
            Err(e) => format!("ERR\n{e}\n"),
        },
        Err(e) => format!("ERR\n{e}\n"),
    }
}

#[cfg(windows)]
pub fn spawn_pipe_server(state: Arc<SharedEngineState>) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("hushmic-pipe".into())
        .spawn(move || run_pipe_server(state))
        .expect("spawn named-pipe server thread")
}

#[cfg(windows)]
fn run_pipe_server(state: Arc<SharedEngineState>) {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{
        CloseHandle, GetLastError, ERROR_PIPE_CONNECTED, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        FlushFileBuffers, ReadFile, WriteFile, PIPE_ACCESS_DUPLEX,
    };
    use windows_sys::Win32::System::Pipes::{
        ConnectNamedPipe, CreateNamedPipeW, DisconnectNamedPipe, PIPE_READMODE_MESSAGE,
        PIPE_TYPE_MESSAGE, PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
    };

    let pipe_w: Vec<u16> = std::ffi::OsStr::new(PIPE_NAME)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    while !state.is_stopped() {
        // SAFETY: creates a message-mode duplex named pipe instance for local CLI control.
        let handle = unsafe {
            CreateNamedPipeW(
                pipe_w.as_ptr(),
                PIPE_ACCESS_DUPLEX,
                PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
                PIPE_UNLIMITED_INSTANCES,
                4096,
                4096,
                200,
                std::ptr::null(),
            )
        };
        if handle == INVALID_HANDLE_VALUE || handle.is_null() {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        // SAFETY: waits for a client connection on `handle`.
        let connected = unsafe {
            ConnectNamedPipe(handle, std::ptr::null_mut()) != 0
                || GetLastError() == ERROR_PIPE_CONNECTED
        };
        if connected {
            let mut buf = [0u8; 2048];
            let mut read: u32 = 0;
            // SAFETY: reads one request message from the connected client.
            let ok = unsafe {
                ReadFile(
                    handle,
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                    &mut read,
                    std::ptr::null_mut(),
                ) != 0
            };
            if ok && read > 0 {
                let text = String::from_utf8_lossy(&buf[..read as usize]);
                let reply = process_pipe_line(&state, text.trim());
                let mut written: u32 = 0;
                // SAFETY: writes the response payload back to the client and flushes.
                unsafe {
                    let _ = WriteFile(
                        handle,
                        reply.as_ptr(),
                        reply.len() as u32,
                        &mut written,
                        std::ptr::null_mut(),
                    );
                    let _ = FlushFileBuffers(handle);
                }
            }
            // SAFETY: disconnects and closes this pipe instance.
            unsafe {
                let _ = DisconnectNamedPipe(handle);
            }
        }
        unsafe {
            let _ = CloseHandle(handle);
        }
    }
}

/// Send a CLI command to `\\.\pipe\hushmic-win`, or fall back to offline config file handling.
#[cfg(windows)]
pub fn send_cli_request(words: &[&str]) -> Result<String, String> {
    use std::io::{Read, Write};
    let req = parse_request(words)?;
    let line = serialize_line(words);

    match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(PIPE_NAME)
    {
        Ok(mut pipe) => {
            pipe.write_all(line.as_bytes())
                .map_err(|e| format!("write pipe: {e}"))?;
            let _ = pipe.flush();
            let mut resp = String::new();
            pipe.read_to_string(&mut resp)
                .map_err(|e| format!("read pipe: {e}"))?;
            if let Some(rest) = resp.strip_prefix("OK\n") {
                Ok(rest.trim_end().to_string())
            } else if let Some(rest) = resp.strip_prefix("ERR\n") {
                Err(rest.trim_end().to_string())
            } else {
                Ok(resp.trim_end().to_string())
            }
        }
        Err(_) => handle_offline_config(req),
    }
}

#[cfg(not(windows))]
pub fn send_cli_request(words: &[&str]) -> Result<String, String> {
    let req = parse_request(words)?;
    handle_offline_config(req)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_request_matches_linux_grammar() {
        assert_eq!(
            parse_request(&["status"]),
            Ok(Request::Status { json: false })
        );
        assert_eq!(
            parse_request(&["status", "--json"]),
            Ok(Request::Status { json: true })
        );
        assert_eq!(
            parse_request(&["mode", "suppress"]),
            Ok(Request::SetMode(Some(RunMode::Suppress)))
        );
        assert_eq!(
            parse_request(&["mode", "bypass"]),
            Ok(Request::SetMode(Some(RunMode::Bypass)))
        );
        assert_eq!(
            parse_request(&["mode", "mute"]),
            Ok(Request::SetMode(Some(RunMode::Mute)))
        );
        assert_eq!(parse_request(&["mode", "off"]), Ok(Request::SetMode(None)));
        assert_eq!(
            parse_request(&["toggle", "mute"]),
            Ok(Request::Toggle(RunMode::Mute))
        );
        assert_eq!(
            parse_request(&["config", "set", "attn_limit", "strong"]),
            Ok(Request::ConfigSet {
                key: "attn_limit".into(),
                value: "strong".into()
            })
        );
        assert_eq!(parse_request(&["quit"]), Ok(Request::Quit));
    }

    #[test]
    fn handle_request_updates_mode_and_toggles() {
        let state = SharedEngineState::new(Config::default());
        assert_eq!(
            handle_request(&state, Request::GetMode).unwrap(),
            "suppress"
        );
        assert_eq!(
            handle_request(&state, Request::Toggle(RunMode::Mute)).unwrap(),
            "mute"
        );
        assert_eq!(
            handle_request(&state, Request::Toggle(RunMode::Mute)).unwrap(),
            "suppress"
        );
        let status_json = handle_request(&state, Request::Status { json: true }).unwrap();
        assert!(status_json.contains("\"mode\":\"suppress\""));
    }
}
