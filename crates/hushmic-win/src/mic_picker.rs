//! Small window for choosing the capture microphone.

use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;

use windows_sys::Win32::Foundation::{GetLastError, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{GetStockObject, DEFAULT_GUI_FONT};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRect, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
    GetMessageW, GetSystemMetrics, GetWindowLongPtrW, IsDialogMessageW, LoadCursorW, MessageBoxW,
    PostQuitMessage, RegisterClassW, SendMessageW, SetForegroundWindow, SetWindowLongPtrW,
    ShowWindow, TranslateMessage, BS_DEFPUSHBUTTON, CS_HREDRAW, CS_VREDRAW, GWLP_USERDATA, HMENU,
    IDC_ARROW, LBS_HASSTRINGS, LBS_NOINTEGRALHEIGHT, LBS_NOTIFY, MB_ICONERROR, MB_OK, MSG,
    SM_CXSCREEN, SM_CYSCREEN, SW_SHOW, WM_CLOSE, WM_COMMAND, WM_CREATE, WM_DESTROY, WM_NCCREATE,
    WM_SETFONT, WNDCLASSW, WS_BORDER, WS_CAPTION, WS_CHILD, WS_EX_DLGMODALFRAME, WS_OVERLAPPED,
    WS_SYSMENU, WS_TABSTOP, WS_VISIBLE, WS_VSCROLL,
};

const LIST_ID: usize = 100;
const OK_ID: usize = 1;
const CANCEL_ID: usize = 2;
const LB_ADDSTRING: u32 = 0x0180;
const LB_SETCURSEL: u32 = 0x0186;
const LB_GETCURSEL: u32 = 0x0188;
const LB_ERR: isize = -1;
const LBN_DBLCLK: u16 = 2;
const ERROR_CLASS_ALREADY_EXISTS: u32 = 1410;

struct PickerState {
    labels: Vec<String>,
    selected: i32,
    done: bool,
    accepted: bool,
}

/// Row 0 is "Windows default". Device names occupy the following rows.
pub fn selection_index(current: Option<&str>, device_names: &[String]) -> usize {
    let Some(current) = current.map(str::trim).filter(|s| !s.is_empty()) else {
        return 0;
    };
    device_names
        .iter()
        .position(|name| name.eq_ignore_ascii_case(current))
        .map(|idx| idx + 1)
        .unwrap_or(0)
}

/// `Ok(None)` is Cancel. `Ok(Some(None))` is the Windows default microphone.
/// `Ok(Some(Some(name)))` is a chosen capture endpoint.
pub fn choose_microphone(current: Option<&str>) -> Result<Option<Option<String>>, String> {
    let (default_name, devices) = crate::audio::win_wasapi::capture_device_names()?;
    let mut labels = Vec::with_capacity(devices.len() + 1);
    labels.push(format!("По умолчанию Windows — {default_name}"));
    labels.extend(devices.iter().cloned());
    let selected = selection_index(current, &devices) as i32;
    let device_count = devices.len();

    let mut state = Box::new(PickerState {
        labels,
        selected,
        done: false,
        accepted: false,
    });
    let state_ptr = state.as_mut() as *mut PickerState;

    let hwnd = create_picker_window(state_ptr, &state.labels, selected)?;
    // SAFETY: `hwnd` is the window just created on this thread.
    unsafe {
        ShowWindow(hwnd, SW_SHOW);
        SetForegroundWindow(hwnd);
    }

    // SAFETY: the window procedure only reads `state` while this loop owns it.
    unsafe {
        let mut msg: MSG = std::mem::zeroed();
        loop {
            let got = GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0);
            if got == 0 {
                PostQuitMessage(msg.wParam as i32);
                break;
            }
            if got < 0 {
                break;
            }
            if IsDialogMessageW(hwnd, &msg) == 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if (*state_ptr).done {
                break;
            }
        }
    }

    if !state.accepted {
        return Ok(None);
    }
    let idx = state.selected.max(0) as usize;
    if idx == 0 || idx > device_count {
        Ok(Some(None))
    } else {
        Ok(Some(Some(devices[idx - 1].clone())))
    }
}

fn create_picker_window(
    state: *mut PickerState,
    labels: &[String],
    selected: i32,
) -> Result<HWND, String> {
    let class_name = wide("HushMicMicPicker");
    let title = wide("Микрофон HushMic");
    // SAFETY: registers a private window class and creates one overlapped window.
    unsafe {
        let instance = GetModuleHandleW(std::ptr::null());
        let mut wc: WNDCLASSW = std::mem::zeroed();
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = Some(wnd_proc);
        wc.hInstance = instance;
        wc.hCursor = LoadCursorW(std::ptr::null_mut(), IDC_ARROW as _);
        wc.hbrBackground = (5isize + 1) as _; // COLOR_WINDOW + 1
        wc.lpszClassName = class_name.as_ptr();
        let atom = RegisterClassW(&wc);
        if atom == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS {
            return Err(format!("RegisterClassW failed ({})", GetLastError()));
        }

        let style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: 460,
            bottom: 320,
        };
        AdjustWindowRect(&mut rect, style, 0);
        let width = rect.right - rect.left;
        let height = rect.bottom - rect.top;
        let screen_w = GetSystemMetrics(SM_CXSCREEN);
        let screen_h = GetSystemMetrics(SM_CYSCREEN);
        let x = (screen_w - width).max(0) / 2;
        let y = (screen_h - height).max(0) / 2;

        let hwnd = CreateWindowExW(
            WS_EX_DLGMODALFRAME,
            class_name.as_ptr(),
            title.as_ptr(),
            style,
            x,
            y,
            width,
            height,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            instance,
            state.cast(),
        );
        if hwnd.is_null() {
            return Err(format!("CreateWindowExW failed ({})", GetLastError()));
        }
        build_controls(hwnd, labels, selected);
        Ok(hwnd)
    }
}

fn build_controls(parent: HWND, labels: &[String], selected: i32) {
    let list_style = WS_CHILD
        | WS_VISIBLE
        | WS_TABSTOP
        | WS_VSCROLL
        | WS_BORDER
        | (LBS_NOTIFY as u32)
        | (LBS_HASSTRINGS as u32)
        | (LBS_NOINTEGRALHEIGHT as u32);
    let ok_style = WS_CHILD | WS_VISIBLE | WS_TABSTOP | (BS_DEFPUSHBUTTON as u32);
    let cancel_style = WS_CHILD | WS_VISIBLE | WS_TABSTOP;
    // SAFETY: parent is a live window on this thread; control ids are local to it.
    unsafe {
        let instance = GetModuleHandleW(std::ptr::null());
        let font = GetStockObject(DEFAULT_GUI_FONT);
        let list = make_child(
            parent, instance, "LISTBOX", "", list_style, 12, 12, 436, 248, LIST_ID,
        );
        let ok = make_child(
            parent, instance, "BUTTON", "ОК", ok_style, 248, 272, 92, 28, OK_ID,
        );
        let cancel = make_child(
            parent,
            instance,
            "BUTTON",
            "Отмена",
            cancel_style,
            350,
            272,
            98,
            28,
            CANCEL_ID,
        );
        for (hwnd, _) in [(list, 0), (ok, 0), (cancel, 0)] {
            if !hwnd.is_null() {
                SendMessageW(hwnd, WM_SETFONT, font as WPARAM, 1);
            }
        }
        for label in labels {
            let wide_label = wide(label);
            SendMessageW(list, LB_ADDSTRING, 0, wide_label.as_ptr() as LPARAM);
        }
        SendMessageW(list, LB_SETCURSEL, selected as WPARAM, 0);
    }
}

fn make_child(
    parent: HWND,
    instance: windows_sys::Win32::Foundation::HINSTANCE,
    class_name: &str,
    text: &str,
    style: u32,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    id: usize,
) -> HWND {
    let class_name = wide(class_name);
    let text = wide(text);
    // SAFETY: caller passes a window created on this thread.
    unsafe {
        CreateWindowExW(
            0,
            class_name.as_ptr(),
            text.as_ptr(),
            style,
            x,
            y,
            w,
            h,
            parent,
            id as HMENU,
            instance,
            std::ptr::null(),
        )
    }
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_NCCREATE {
        let create =
            &*(lparam as *const windows_sys::Win32::UI::WindowsAndMessaging::CREATESTRUCTW);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, create.lpCreateParams as isize);
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
    let state = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut PickerState;
    if state.is_null() {
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
    match msg {
        WM_CREATE => 0,
        WM_COMMAND => {
            let id = (wparam & 0xFFFF) as u16;
            let code = ((wparam >> 16) & 0xFFFF) as u16;
            if id as usize == OK_ID || (id as usize == LIST_ID && code == LBN_DBLCLK) {
                accept(hwnd, state);
            } else if id as usize == CANCEL_ID {
                (*state).accepted = false;
                DestroyWindow(hwnd);
            }
            0
        }
        WM_CLOSE => {
            (*state).accepted = false;
            DestroyWindow(hwnd);
            0
        }
        WM_DESTROY => {
            (*state).done = true;
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

unsafe fn accept(hwnd: HWND, state: *mut PickerState) {
    let list = windows_sys::Win32::UI::WindowsAndMessaging::GetDlgItem(hwnd, LIST_ID as i32);
    let sel = SendMessageW(list, LB_GETCURSEL, 0, 0);
    if sel == LB_ERR {
        (*state).accepted = false;
    } else {
        (*state).selected = sel as i32;
        (*state).accepted = true;
    }
    DestroyWindow(hwnd);
}

fn wide(text: &str) -> Vec<u16> {
    OsStr::new(text)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

pub fn report_picker_error(message: &str) {
    let text = wide(message);
    let title = wide("HushMic");
    // SAFETY: both strings are NUL-terminated and live until MessageBoxW returns.
    unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            text.as_ptr(),
            title.as_ptr(),
            MB_OK | MB_ICONERROR,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::selection_index;

    #[test]
    fn default_row_is_selected_when_mic_is_unset_or_unknown() {
        let devices = vec!["Microphone Array".to_string(), "Headset".to_string()];
        assert_eq!(selection_index(None, &devices), 0);
        assert_eq!(selection_index(Some("default"), &devices), 0);
        assert_eq!(selection_index(Some("missing"), &devices), 0);
        assert_eq!(selection_index(Some("headset"), &devices), 2);
    }
}
