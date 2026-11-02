//! Windows system tray menu (`#[cfg(windows)]`) using `tray-icon`.
//! Exposes Suppress, Bypass, Mute, Off, microphone picker, Autostart, and Quit.

#[cfg(windows)]
pub mod win_tray {
    use crate::audio::SharedEngineState;
    use crate::config::RunMode;
    use crate::control::{handle_request, Request};
    use std::sync::Arc;
    use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem};
    use tray_icon::{Icon, TrayIconBuilder};

    fn make_solid_icon(r: u8, g: u8, b: u8) -> Icon {
        let width = 16u32;
        let height = 16u32;
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let border = x == 0 || y == 0 || x == width - 1 || y == height - 1;
                if border {
                    rgba.extend_from_slice(&[24, 24, 24, 255]);
                } else {
                    rgba.extend_from_slice(&[r, g, b, 255]);
                }
            }
        }
        Icon::from_rgba(rgba, width, height).expect("valid 16x16 RGBA tray icon")
    }

    /// Run the Windows tray message loop on the main thread until `state.is_stopped()` becomes true.
    pub fn run_tray_loop(state: Arc<SharedEngineState>) -> Result<(), String> {
        let menu = Menu::new();
        let snap = state
            .snapshot
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();

        let header = MenuItem::new(format!("HushMic ({})", snap.engine), false, None);
        let item_suppress = CheckMenuItem::new(
            "Suppress noise",
            true,
            snap.enabled && snap.mode == RunMode::Suppress,
            None,
        );
        let item_bypass = CheckMenuItem::new(
            "Bypass (raw mic)",
            true,
            snap.enabled && snap.mode == RunMode::Bypass,
            None,
        );
        let item_mute = CheckMenuItem::new(
            "Mute",
            true,
            snap.enabled && snap.mode == RunMode::Mute,
            None,
        );
        let item_off = CheckMenuItem::new("Off", true, !snap.enabled, None);
        let item_mic = {
            let cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
            MenuItem::new(mic_menu_label(cfg.mic.as_deref()), true, None)
        };
        let item_autostart = {
            let cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
            CheckMenuItem::new("Start on login", true, cfg.autostart, None)
        };
        let item_quit = MenuItem::new("Quit HushMic", true, None);

        let _ = menu.append(&header);
        let _ = menu.append(&PredefinedMenuItem::separator());
        let _ = menu.append(&item_suppress);
        let _ = menu.append(&item_bypass);
        let _ = menu.append(&item_mute);
        let _ = menu.append(&item_off);
        let _ = menu.append(&PredefinedMenuItem::separator());
        let _ = menu.append(&item_mic);
        let _ = menu.append(&PredefinedMenuItem::separator());
        let _ = menu.append(&item_autostart);
        let _ = menu.append(&PredefinedMenuItem::separator());
        let _ = menu.append(&item_quit);

        let icon = make_solid_icon(46, 160, 67);
        let _tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip(format!(
                "HushMic - {} ({})",
                snap.mode.as_str(),
                snap.engine
            ))
            .with_icon(icon)
            .build()
            .map_err(|e| format!("create tray icon: {e}"))?;

        let id_suppress = item_suppress.id().clone();
        let id_bypass = item_bypass.id().clone();
        let id_mute = item_mute.id().clone();
        let id_off = item_off.id().clone();
        let id_mic = item_mic.id().clone();
        let id_autostart = item_autostart.id().clone();
        let id_quit = item_quit.id().clone();

        let sync_checks = |s: &SharedEngineState| {
            let cfg = s.config.lock().unwrap_or_else(|p| p.into_inner());
            item_suppress.set_checked(cfg.enabled && cfg.mode == RunMode::Suppress);
            item_bypass.set_checked(cfg.enabled && cfg.mode == RunMode::Bypass);
            item_mute.set_checked(cfg.enabled && cfg.mode == RunMode::Mute);
            item_off.set_checked(!cfg.enabled);
            item_autostart.set_checked(cfg.autostart);
        };

        let mut tick: u32 = 0;
        while !state.is_stopped() {
            tick = tick.wrapping_add(1);
            if tick.is_multiple_of(40) {
                let (active, activity, client) = {
                    let snap = state.snapshot.lock().unwrap_or_else(|p| p.into_inner());
                    (snap.npu_active, snap.npu_activity, snap.npu_client.clone())
                };
                let _ = _tray.set_tooltip(Some(npu_tooltip(active, activity, &client)));
            }
            pump_win32_messages();
            while let Ok(event) = MenuEvent::receiver().try_recv() {
                if event.id == id_suppress {
                    let _ = handle_request(&state, Request::SetMode(Some(RunMode::Suppress)));
                } else if event.id == id_bypass {
                    let _ = handle_request(&state, Request::SetMode(Some(RunMode::Bypass)));
                } else if event.id == id_mute {
                    let _ = handle_request(&state, Request::SetMode(Some(RunMode::Mute)));
                } else if event.id == id_off {
                    let _ = handle_request(&state, Request::SetMode(None));
                } else if event.id == id_mic {
                    let current = {
                        let cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
                        cfg.mic.clone()
                    };
                    match crate::mic_picker::choose_microphone(current.as_deref()) {
                        Ok(Some(choice)) => {
                            let value = choice.unwrap_or_else(|| "default".to_string());
                            let _ = handle_request(
                                &state,
                                Request::ConfigSet {
                                    key: "mic".into(),
                                    value,
                                },
                            );
                            let cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
                            item_mic.set_text(mic_menu_label(cfg.mic.as_deref()));
                        }
                        Ok(None) => {}
                        Err(err) => crate::mic_picker::report_picker_error(&err),
                    }
                } else if event.id == id_autostart {
                    let next = {
                        let cfg = state.config.lock().unwrap_or_else(|p| p.into_inner());
                        !cfg.autostart
                    };
                    let _ = handle_request(
                        &state,
                        Request::ConfigSet {
                            key: "autostart".into(),
                            value: next.to_string(),
                        },
                    );
                } else if event.id == id_quit {
                    let _ = handle_request(&state, Request::Quit);
                }
                sync_checks(&state);
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        Ok(())
    }

    fn npu_tooltip(active: bool, activity: &str, client: &str) -> String {
        if active && activity == "call" && !client.is_empty() {
            format!("HushMic — NPU: звонок ({client})")
        } else if active && activity == "call" {
            "HushMic — NPU: звонок".to_string()
        } else if active && activity == "signal" {
            "HushMic — NPU: сигнал с микрофона".to_string()
        } else if active {
            "HushMic — NPU включен".to_string()
        } else if activity == "idle" {
            "HushMic — NPU спит".to_string()
        } else {
            "HushMic — NPU выключен".to_string()
        }
    }

    fn mic_menu_label(mic: Option<&str>) -> String {
        match mic.map(str::trim).filter(|s| !s.is_empty()) {
            None => "Микрофон: по умолчанию".to_string(),
            Some(name) => {
                let short: String = name.chars().take(42).collect();
                if name.chars().count() > 42 {
                    format!("Микрофон: {short}…")
                } else {
                    format!("Микрофон: {short}")
                }
            }
        }
    }

    fn pump_win32_messages() {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE,
        };
        unsafe {
            let mut msg: MSG = std::mem::zeroed();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }
}
