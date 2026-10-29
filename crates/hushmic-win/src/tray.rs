//! Windows system tray menu (`#[cfg(windows)]`) using `tray-icon`.
//! Exposes Suppress, Bypass, Mute, Off, Autostart, and Quit.

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

        while !state.is_stopped() {
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
