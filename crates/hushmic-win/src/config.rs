//! Persistent configuration for `hushmic-win` (`%APPDATA%\HushMic\config.toml` on Windows).

use directories::ProjectDirs;
use hushmic_denoiser::{Mode, NpuInference};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Operating mode of the Windows noise-suppression engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RunMode {
    #[default]
    Suppress,
    Bypass,
    Mute,
}

impl RunMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "suppress" | "process" | "on" => Some(Self::Suppress),
            "bypass" => Some(Self::Bypass),
            "mute" => Some(Self::Mute),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Suppress => "suppress",
            Self::Bypass => "bypass",
            Self::Mute => "mute",
        }
    }

    pub fn to_denoiser_mode(self) -> Mode {
        match self {
            Self::Suppress => Mode::Process,
            Self::Bypass => Mode::Bypass,
            Self::Mute => Mode::Mute,
        }
    }
}

/// Named attenuation limit presets matching the Linux CLI (`crates/hushmic/src/config_cli.rs`).
pub const ATTN_PRESETS: [(&str, f32); 4] = [
    ("maximum", 100.0),
    ("strong", 24.0),
    ("medium", 12.0),
    ("light", 6.0),
];

pub fn parse_attn_limit(raw: &str) -> Result<f32, String> {
    let trimmed = raw.trim().to_ascii_lowercase();
    for (name, db) in ATTN_PRESETS {
        if trimmed == name {
            return Ok(db);
        }
    }
    if trimmed == "unlimited" || trimmed == "off" {
        return Ok(100.0);
    }
    let val: f32 = trimmed.parse().map_err(|_| {
        format!("invalid attn_limit '{raw}' (use maximum|strong|medium|light or 0..100)")
    })?;
    if !(0.0..=100.0).contains(&val) {
        return Err(format!(
            "attn_limit {val} out of range (expected 0..100 dB)"
        ));
    }
    Ok(val)
}

pub fn normalize_model_name(raw: &str) -> String {
    match raw.trim() {
        "quality" | "dpdfnet8" | "dpdfnet8_48khz_hr" => "dpdfnet8_48khz_hr".to_string(),
        "light" | "dpdfnet2" | "dpdfnet2_48khz_hr" => "dpdfnet2_48khz_hr".to_string(),
        other => other.to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub enabled: bool,
    pub mode: RunMode,
    /// Optional capture endpoint name substring; `None` uses the Windows default microphone.
    pub mic: Option<String>,
    /// Playback endpoint name substring receiving cleaned 48 kHz audio (default: `"CABLE Input"`).
    pub render_device: String,
    /// Model stem (`dpdfnet8_48khz_hr` or `dpdfnet2_48khz_hr`) or explicit path to an `.onnx` file.
    pub model: String,
    /// Attenuation limit in dB (`100.0` = maximum).
    pub attn_limit: f32,
    /// Inference backend: `"auto"` (try OpenVINO NPU, then ONNX CPU), `"npu"`, or `"onnx"`.
    pub inference: String,
    /// Start `hushmic-win` at Windows logon via `HKCU\...\Run`.
    pub autostart: bool,
    /// Show the system tray icon while running.
    pub tray: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: RunMode::Suppress,
            mic: None,
            render_device: "CABLE Input".to_string(),
            model: "dpdfnet8_48khz_hr".to_string(),
            attn_limit: 100.0,
            inference: "auto".to_string(),
            autostart: false,
            tray: true,
        }
    }
}

impl Config {
    /// `%APPDATA%\HushMic\config.toml` on Windows, or standard user config dir elsewhere.
    pub fn path() -> PathBuf {
        if let Ok(appdata) = std::env::var("APPDATA") {
            let trimmed = appdata.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed).join("HushMic").join("config.toml");
            }
        }
        if let Some(proj) = ProjectDirs::from("", "", "HushMic") {
            return proj.config_dir().join("config.toml");
        }
        PathBuf::from("config.toml")
    }

    /// `%LOCALAPPDATA%\HushMic\ov-cache` on Windows, used for compiled OpenVINO NPU blobs.
    pub fn ov_cache_dir() -> PathBuf {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            let trimmed = local.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed).join("HushMic").join("ov-cache");
            }
        }
        if let Some(proj) = ProjectDirs::from("", "", "HushMic") {
            return proj.cache_dir().join("ov-cache");
        }
        PathBuf::from("ov-cache")
    }

    pub fn load() -> Self {
        Self::load_from(&Self::path())
    }

    pub fn load_from(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> Result<(), String> {
        self.save_to(&Self::path())
    }

    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        let text = toml::to_string_pretty(self).map_err(|e| format!("serialize config: {e}"))?;
        std::fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))
    }

    pub fn npu_inference(&self) -> NpuInference {
        NpuInference::parse(&self.inference).unwrap_or(NpuInference::Auto)
    }

    /// Resolve the `.onnx` file path from `self.model` or an explicit override.
    pub fn resolve_model_path(&self, override_model: Option<&str>) -> PathBuf {
        let raw = override_model.unwrap_or(&self.model);
        let direct = PathBuf::from(raw);
        if direct
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
            && direct.exists()
        {
            return direct;
        }
        let stem = normalize_model_name(raw);
        let filename = if stem.ends_with(".onnx") {
            stem
        } else {
            format!("{stem}.onnx")
        };

        let mut candidates = Vec::new();
        if let Ok(env_model) = std::env::var("HUSHMIC_MODEL_PATH") {
            let trimmed = env_model.trim();
            if !trimmed.is_empty() {
                candidates.push(PathBuf::from(trimmed));
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join(&filename));
                candidates.push(dir.join("models").join(&filename));
                candidates.push(dir.join("../assets/models").join(&filename));
            }
        }
        candidates.push(PathBuf::from("assets/models").join(&filename));
        candidates.push(PathBuf::from("models").join(&filename));
        if let Some(cfg_dir) = Self::path().parent() {
            candidates.push(cfg_dir.join("models").join(&filename));
        }

        for cand in &candidates {
            if cand.exists() {
                return cand.clone();
            }
        }
        candidates
            .into_iter()
            .next()
            .unwrap_or_else(|| PathBuf::from(filename))
    }

    pub fn get_key(&self, key: &str) -> Result<String, String> {
        match key {
            "enabled" => Ok(self.enabled.to_string()),
            "mode" => Ok(if self.enabled {
                self.mode.as_str().to_string()
            } else {
                "off".to_string()
            }),
            "mic" => Ok(self.mic.clone().unwrap_or_else(|| "default".to_string())),
            "render_device" => Ok(self.render_device.clone()),
            "model" => Ok(self.model.clone()),
            "attn_limit" => Ok(format!("{}", self.attn_limit)),
            "inference" | "engine" => Ok(self.inference.clone()),
            "autostart" => Ok(self.autostart.to_string()),
            "tray" => Ok(self.tray.to_string()),
            other => Err(format!(
                "unknown key '{other}' (keys: enabled mode mic render_device model attn_limit inference autostart tray)"
            )),
        }
    }

    pub fn set_key(&mut self, key: &str, value: &str) -> Result<(), String> {
        let val = value.trim();
        if val.is_empty() {
            return Err(format!("'{key}' requires a value"));
        }
        match key {
            "enabled" => {
                return Err(
                    "'enabled' is set with: hushmic-win mode suppress|bypass|mute|off".into(),
                );
            }
            "mode" => {
                if val.eq_ignore_ascii_case("off") {
                    self.enabled = false;
                } else if let Some(m) = RunMode::parse(val) {
                    self.enabled = true;
                    self.mode = m;
                } else {
                    return Err(format!(
                        "invalid mode '{val}' (use suppress|bypass|mute|off)"
                    ));
                }
            }
            "mic" => {
                if val.eq_ignore_ascii_case("default") || val.eq_ignore_ascii_case("none") {
                    self.mic = None;
                } else {
                    self.mic = Some(val.to_string());
                }
            }
            "render_device" => {
                self.render_device = val.to_string();
            }
            "model" => {
                self.model = normalize_model_name(val);
            }
            "attn_limit" => {
                self.attn_limit = parse_attn_limit(val)?;
            }
            "inference" | "engine" => {
                let parsed = NpuInference::parse(val).ok_or_else(|| {
                    format!("invalid inference engine '{val}' (use auto|npu|onnx)")
                })?;
                self.inference = parsed.word().to_string();
            }
            "autostart" => {
                self.autostart = parse_bool(val, "autostart")?;
            }
            "tray" => {
                self.tray = parse_bool(val, "tray")?;
            }
            other => {
                return Err(format!(
                    "unknown key '{other}' (settable keys: mode mic render_device model attn_limit inference autostart tray)"
                ));
            }
        }
        Ok(())
    }
}

fn parse_bool(raw: &str, key: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "on" | "yes" => Ok(true),
        "false" | "0" | "off" | "no" => Ok(false),
        _ => Err(format!(
            "invalid boolean '{raw}' for {key} (use true|false)"
        )),
    }
}

/// Initialize `onnxruntime.dll` from the executable's directory when present.
pub fn init_adjacent_ort_runtime() {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let dll = dir.join("onnxruntime.dll");
            if dll.exists() {
                let _ = hushmic_denoiser::init_runtime(&dll);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_get_and_set_round_trip() {
        let mut cfg = Config::default();
        assert_eq!(cfg.get_key("render_device").unwrap(), "CABLE Input");
        assert_eq!(cfg.get_key("inference").unwrap(), "auto");

        cfg.set_key("attn_limit", "strong").unwrap();
        assert_eq!(cfg.attn_limit, 24.0);

        cfg.set_key("model", "light").unwrap();
        assert_eq!(cfg.model, "dpdfnet2_48khz_hr");

        cfg.set_key("inference", "npu").unwrap();
        assert_eq!(cfg.npu_inference(), NpuInference::Npu);

        cfg.set_key("mic", "ThinkPad Mic").unwrap();
        assert_eq!(cfg.mic.as_deref(), Some("ThinkPad Mic"));
        cfg.set_key("mic", "default").unwrap();
        assert_eq!(cfg.mic, None);

        cfg.set_key("autostart", "true").unwrap();
        assert!(cfg.autostart);
    }

    #[test]
    fn toml_serialization_round_trips() {
        let mut cfg = Config::default();
        cfg.set_key("attn_limit", "18.5").unwrap();
        cfg.set_key("render_device", "CABLE Input (VB-Audio Virtual Cable)")
            .unwrap();
        let text = toml::to_string_pretty(&cfg).unwrap();
        let loaded: Config = toml::from_str(&text).unwrap();
        assert_eq!(loaded, cfg);
    }
}
