//! Windows logon autostart via `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`.

const RUN_SUBKEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
const VALUE_NAME: &str = "HushMic";

/// Synchronize the HKCU Run value with `enabled`.
#[cfg(windows)]
pub fn apply_autostart(enabled: bool) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::ERROR_FILE_NOT_FOUND;
    use windows_sys::Win32::System::Registry::{
        RegCloseKey, RegCreateKeyExW, RegDeleteValueW, RegSetValueExW, HKEY, HKEY_CURRENT_USER,
        KEY_SET_VALUE, REG_OPTION_NON_VOLATILE, REG_SZ,
    };

    fn wide_nul(s: &str) -> Vec<u16> {
        std::ffi::OsStr::new(s)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    let subkey = wide_nul(RUN_SUBKEY);
    let value_name = wide_nul(VALUE_NAME);
    let mut hkey: HKEY = std::ptr::null_mut();

    // SAFETY: opens/creates the standard per-user Run key under HKCU.
    let status = unsafe {
        RegCreateKeyExW(
            HKEY_CURRENT_USER,
            subkey.as_ptr(),
            0,
            std::ptr::null(),
            REG_OPTION_NON_VOLATILE,
            KEY_SET_VALUE,
            std::ptr::null(),
            &mut hkey,
            std::ptr::null_mut(),
        )
    };
    if status != 0 {
        return Err(format!("RegCreateKeyExW failed with Win32 status {status}"));
    }

    let res = if enabled {
        let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
        let cmd = format!("\"{}\"", exe.display());
        let data = wide_nul(&cmd);
        let byte_len = (data.len() * 2) as u32;
        // SAFETY: writes a NUL-terminated UTF-16 REG_SZ command line.
        let st = unsafe {
            RegSetValueExW(
                hkey,
                value_name.as_ptr(),
                0,
                REG_SZ,
                data.as_ptr().cast::<u8>(),
                byte_len,
            )
        };
        if st == 0 {
            Ok(())
        } else {
            Err(format!("RegSetValueExW failed with Win32 status {st}"))
        }
    } else {
        // SAFETY: removes the HushMic value if present.
        let st = unsafe { RegDeleteValueW(hkey, value_name.as_ptr()) };
        if st == 0 || st == ERROR_FILE_NOT_FOUND {
            Ok(())
        } else {
            Err(format!("RegDeleteValueW failed with Win32 status {st}"))
        }
    };

    // SAFETY: closes the key handle opened above.
    unsafe {
        RegCloseKey(hkey);
    }
    res
}

#[cfg(not(windows))]
pub fn apply_autostart(_enabled: bool) -> Result<(), String> {
    let _ = (RUN_SUBKEY, VALUE_NAME);
    Ok(())
}
