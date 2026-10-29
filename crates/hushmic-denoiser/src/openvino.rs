//! OpenVINO NPU inference engine (feature `npu`): loads `openvino_c.dll` (or
//! `libopenvino_c.so`) at runtime via `libloading` and compiles the DPDFNet
//! ONNX model for the `NPU` device.
//!
//! No OpenVINO SDK is required at build time. When the shared library is
//! missing, no `NPU` device is present, or compilation fails, `load` returns
//! an `Err(String)` explaining why, and [`Denoiser::from_file_npu`] falls back
//! to the CPU ONNX Runtime session while recording the reason.
//!
//! The compiled graph serves the exact same one-hop contract as [`Model`]:
//! - inputs : `spec` `[1, 1, 481, 2]` (`f32`) and `state_in` `[state_size]` (`f32`)
//! - outputs: `spec_e` `[1, 1, 481, 2]` (`f32`) and `state_out` `[state_size]` (`f32`)

use crate::stft::SPEC_LEN;
use libloading::Library;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

/// Which inference backend [`Denoiser::from_file_npu`](crate::Denoiser::from_file_npu) selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NpuInference {
    /// Try OpenVINO `NPU` first; fall back to ONNX Runtime CPU with a reason.
    #[default]
    Auto,
    /// Require OpenVINO `NPU` if available, falling back to ONNX Runtime CPU
    /// with a recorded fallback reason if compilation or device lookup fails.
    Npu,
    /// Use ONNX Runtime CPU directly without probing OpenVINO.
    Onnx,
}

impl NpuInference {
    /// Parse `"auto"`, `"npu"`, or `"onnx"`.
    pub fn parse(word: &str) -> Option<NpuInference> {
        match word.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(NpuInference::Auto),
            "npu" | "openvino" | "openvino-npu" => Some(NpuInference::Npu),
            "onnx" | "cpu" => Some(NpuInference::Onnx),
            _ => None,
        }
    }

    /// Canonical configuration word for this mode.
    pub fn word(self) -> &'static str {
        match self {
            NpuInference::Auto => "auto",
            NpuInference::Npu => "npu",
            NpuInference::Onnx => "onnx",
        }
    }
}

/// Which backend a [`Denoiser`](crate::Denoiser) created with `feature = "npu"` runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuEngine {
    /// OpenVINO `NPU` plugin.
    Npu,
    /// ONNX Runtime CPU session.
    Onnx,
}

impl NpuEngine {
    pub fn is_npu(self) -> bool {
        self == NpuEngine::Npu
    }
}

impl std::fmt::Display for NpuEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            NpuEngine::Npu => "openvino npu",
            NpuEngine::Onnx => "onnx",
        })
    }
}

#[repr(C)]
struct OvCore {
    _opaque: [u8; 0],
}

#[repr(C)]
struct OvCompiledModel {
    _opaque: [u8; 0],
}

#[repr(C)]
struct OvInferRequest {
    _opaque: [u8; 0],
}

#[repr(C)]
struct OvTensor {
    _opaque: [u8; 0],
}

type OvStatus = c_int;
const OV_OK: OvStatus = 0;

type FnCoreCreate = unsafe extern "C" fn(*mut *mut OvCore) -> OvStatus;
type FnCoreFree = unsafe extern "C" fn(*mut OvCore);
type FnCoreSetProperty = unsafe extern "C" fn(
    *const OvCore,
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
) -> OvStatus;
type FnCoreCompileModelFromFile = unsafe extern "C" fn(
    *const OvCore,
    *const c_char,
    *const c_char,
    usize,
    *mut *mut OvCompiledModel,
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
) -> OvStatus;
type FnCompiledModelCreateInferRequest =
    unsafe extern "C" fn(*const OvCompiledModel, *mut *mut OvInferRequest) -> OvStatus;
type FnCompiledModelFree = unsafe extern "C" fn(*mut OvCompiledModel);
type FnInferRequestGetTensor =
    unsafe extern "C" fn(*const OvInferRequest, *const c_char, *mut *mut OvTensor) -> OvStatus;
type FnInferRequestInfer = unsafe extern "C" fn(*mut OvInferRequest) -> OvStatus;
type FnInferRequestFree = unsafe extern "C" fn(*mut OvInferRequest);
type FnTensorData = unsafe extern "C" fn(*const OvTensor, *mut *mut c_void) -> OvStatus;
type FnTensorGetSize = unsafe extern "C" fn(*const OvTensor, *mut usize) -> OvStatus;
type FnTensorFree = unsafe extern "C" fn(*mut OvTensor);
type FnGetErrorInfo = unsafe extern "C" fn(OvStatus) -> *const c_char;

/// Dynamically loaded function table for `openvino_c`.
struct OvApi {
    _lib: Library,
    core_create: FnCoreCreate,
    core_free: FnCoreFree,
    core_set_property: FnCoreSetProperty,
    core_compile_model_from_file: FnCoreCompileModelFromFile,
    compiled_model_create_infer_request: FnCompiledModelCreateInferRequest,
    compiled_model_free: FnCompiledModelFree,
    infer_request_get_tensor: FnInferRequestGetTensor,
    infer_request_infer: FnInferRequestInfer,
    infer_request_free: FnInferRequestFree,
    tensor_data: FnTensorData,
    tensor_get_size: FnTensorGetSize,
    tensor_free: FnTensorFree,
    get_error_info: Option<FnGetErrorInfo>,
}

impl OvApi {
    fn load() -> Result<Self, String> {
        let candidates = candidate_library_paths();
        let mut last_err = String::from("no OpenVINO C library candidates");
        for cand in candidates {
            // SAFETY: loading the OpenVINO C shared library and resolving its
            // documented C ABI entry points; no symbol is called during load.
            unsafe {
                match Library::new(&cand) {
                    Ok(lib) => return Self::from_library(lib, &cand),
                    Err(e) => {
                        last_err = format!("{}: {e}", cand.display());
                    }
                }
            }
        }
        Err(format!("openvino_c library not found ({last_err})"))
    }

    unsafe fn from_library(lib: Library, shown: &Path) -> Result<Self, String> {
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let s = lib.get::<$ty>($name).map_err(|e| {
                    format!(
                        "{}: missing symbol {}: {e}",
                        shown.display(),
                        String::from_utf8_lossy($name)
                    )
                })?;
                *s
            }};
        }
        let core_create = sym!(b"ov_core_create\0", FnCoreCreate);
        let core_free = sym!(b"ov_core_free\0", FnCoreFree);
        let core_set_property = sym!(b"ov_core_set_property\0", FnCoreSetProperty);
        let core_compile_model_from_file = sym!(
            b"ov_core_compile_model_from_file\0",
            FnCoreCompileModelFromFile
        );
        let compiled_model_create_infer_request = sym!(
            b"ov_compiled_model_create_infer_request\0",
            FnCompiledModelCreateInferRequest
        );
        let compiled_model_free = sym!(b"ov_compiled_model_free\0", FnCompiledModelFree);
        let infer_request_get_tensor =
            sym!(b"ov_infer_request_get_tensor\0", FnInferRequestGetTensor);
        let infer_request_infer = sym!(b"ov_infer_request_infer\0", FnInferRequestInfer);
        let infer_request_free = sym!(b"ov_infer_request_free\0", FnInferRequestFree);
        let tensor_data = sym!(b"ov_tensor_data\0", FnTensorData);
        let tensor_get_size = sym!(b"ov_tensor_get_size\0", FnTensorGetSize);
        let tensor_free = sym!(b"ov_tensor_free\0", FnTensorFree);
        let get_error_info = lib
            .get::<FnGetErrorInfo>(b"ov_get_error_info\0")
            .ok()
            .map(|s| *s);

        Ok(OvApi {
            _lib: lib,
            core_create,
            core_free,
            core_set_property,
            core_compile_model_from_file,
            compiled_model_create_infer_request,
            compiled_model_free,
            infer_request_get_tensor,
            infer_request_infer,
            infer_request_free,
            tensor_data,
            tensor_get_size,
            tensor_free,
            get_error_info,
        })
    }

    fn status_message(&self, status: OvStatus) -> String {
        if let Some(f) = self.get_error_info {
            // SAFETY: ov_get_error_info returns a static C string for status codes.
            let ptr = unsafe { f(status) };
            if !ptr.is_null() {
                if let Ok(s) = unsafe { CStr::from_ptr(ptr) }.to_str() {
                    return format!("{s} (status {status})");
                }
            }
        }
        format!("OpenVINO status {status}")
    }

    fn check(&self, status: OvStatus, context: &str) -> Result<(), String> {
        if status == OV_OK {
            Ok(())
        } else {
            Err(format!("{context}: {}", self.status_message(status)))
        }
    }
}

fn candidate_library_paths() -> Vec<PathBuf> {
    let soname = if cfg!(target_os = "windows") {
        "openvino_c.dll"
    } else if cfg!(any(target_os = "macos", target_os = "ios")) {
        "libopenvino_c.dylib"
    } else {
        "libopenvino_c.so"
    };

    let mut out = Vec::new();
    if let Ok(env_path) = std::env::var("OPENVINO_C_PATH") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            out.push(PathBuf::from(trimmed));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let adjacent = dir.join(soname);
            if adjacent.exists() {
                out.push(adjacent);
            }
        }
    }
    out.push(PathBuf::from(soname));
    out
}

/// One compiled OpenVINO NPU session and its pre-bound synchronous infer request.
pub(crate) struct OpenVinoModel {
    api: OvApi,
    core: NonNull<OvCore>,
    compiled: NonNull<OvCompiledModel>,
    request: NonNull<OvInferRequest>,
    spec_tensor: NonNull<OvTensor>,
    state_in_tensor: NonNull<OvTensor>,
    spec_e_tensor: NonNull<OvTensor>,
    state_out_tensor: NonNull<OvTensor>,
    state_size: usize,
}

// SAFETY: OpenVinoModel owns its OpenVINO handles exclusively; only `run`
// accesses the infer request or tensors and it requires `&mut self`.
unsafe impl Send for OpenVinoModel {}
unsafe impl Sync for OpenVinoModel {}

impl Drop for OpenVinoModel {
    fn drop(&mut self) {
        // SAFETY: handles were created by this `api` instance and are freed in
        // reverse dependency order (tensors -> infer_request -> compiled_model -> core).
        unsafe {
            (self.api.tensor_free)(self.spec_tensor.as_ptr());
            (self.api.tensor_free)(self.state_in_tensor.as_ptr());
            (self.api.tensor_free)(self.spec_e_tensor.as_ptr());
            (self.api.tensor_free)(self.state_out_tensor.as_ptr());
            (self.api.infer_request_free)(self.request.as_ptr());
            (self.api.compiled_model_free)(self.compiled.as_ptr());
            (self.api.core_free)(self.core.as_ptr());
        }
    }
}

impl OpenVinoModel {
    pub(crate) fn state_size(&self) -> usize {
        self.state_size
    }

    /// Compile `onnx_path` on the OpenVINO `NPU` device with `LATENCY` hint and
    /// a single stream, returning `(OpenVinoModel, init_state)`.
    pub(crate) fn load(
        onnx_path: &Path,
        inference: NpuInference,
        cache_dir: Option<&Path>,
    ) -> Result<(OpenVinoModel, Vec<f32>), String> {
        if inference == NpuInference::Onnx {
            return Err("ONNX requested".into());
        }
        let onnx_bytes =
            std::fs::read(onnx_path).map_err(|e| format!("read {}: {e}", onnx_path.display()))?;
        let meta = parse_onnx_custom_metadata(&onnx_bytes);

        let api = OvApi::load()?;
        let mut raw_core: *mut OvCore = std::ptr::null_mut();
        // SAFETY: creates a fresh ov_core_t handle.
        let status = unsafe { (api.core_create)(&mut raw_core) };
        api.check(status, "ov_core_create")?;
        let core = NonNull::new(raw_core).ok_or("ov_core_create returned null")?;

        let device = CString::new("NPU").unwrap();
        if let Some(dir) = cache_dir {
            let _ = std::fs::create_dir_all(dir);
            if let Ok(c_dir) = CString::new(dir.to_string_lossy().as_bytes()) {
                let key = CString::new("CACHE_DIR").unwrap();
                // SAFETY: sets CACHE_DIR on the NPU plugin before compilation;
                // trailing null terminates the variadic property list.
                let _ = unsafe {
                    (api.core_set_property)(
                        core.as_ptr(),
                        device.as_ptr(),
                        key.as_ptr(),
                        c_dir.as_ptr(),
                        std::ptr::null(),
                    )
                };
            }
        }

        let c_model_path = CString::new(onnx_path.to_string_lossy().as_bytes())
            .map_err(|_| "model path contains interior NUL byte".to_string())?;
        let k_hint = CString::new("PERFORMANCE_HINT").unwrap();
        let v_hint = CString::new("LATENCY").unwrap();
        let k_streams = CString::new("NUM_STREAMS").unwrap();
        let v_streams = CString::new("1").unwrap();

        let mut raw_compiled: *mut OvCompiledModel = std::ptr::null_mut();
        // SAFETY: compiles the model on NPU with 4 property string arguments
        // (2 key-value pairs: PERFORMANCE_HINT=LATENCY, NUM_STREAMS=1).
        let status = unsafe {
            (api.core_compile_model_from_file)(
                core.as_ptr(),
                c_model_path.as_ptr(),
                device.as_ptr(),
                4,
                &mut raw_compiled,
                k_hint.as_ptr(),
                v_hint.as_ptr(),
                k_streams.as_ptr(),
                v_streams.as_ptr(),
            )
        };
        if let Err(e) = api.check(status, "ov_core_compile_model_from_file(NPU)") {
            unsafe { (api.core_free)(core.as_ptr()) };
            return Err(e);
        }
        let Some(compiled) = NonNull::new(raw_compiled) else {
            unsafe { (api.core_free)(core.as_ptr()) };
            return Err("ov_core_compile_model_from_file(NPU) returned null".into());
        };

        let mut raw_req: *mut OvInferRequest = std::ptr::null_mut();
        // SAFETY: creates one synchronous infer request from the compiled model.
        let status =
            unsafe { (api.compiled_model_create_infer_request)(compiled.as_ptr(), &mut raw_req) };
        if let Err(e) = api.check(status, "ov_compiled_model_create_infer_request") {
            unsafe {
                (api.compiled_model_free)(compiled.as_ptr());
                (api.core_free)(core.as_ptr());
            }
            return Err(e);
        }
        let Some(request) = NonNull::new(raw_req) else {
            unsafe {
                (api.compiled_model_free)(compiled.as_ptr());
                (api.core_free)(core.as_ptr());
            }
            return Err("ov_compiled_model_create_infer_request returned null".into());
        };

        let get_tensor = |name: &str| -> Result<(NonNull<OvTensor>, usize), String> {
            let c_name = CString::new(name).unwrap();
            let mut raw_t: *mut OvTensor = std::ptr::null_mut();
            // SAFETY: retrieves the named input/output tensor from the request.
            let st = unsafe {
                (api.infer_request_get_tensor)(request.as_ptr(), c_name.as_ptr(), &mut raw_t)
            };
            api.check(st, &format!("ov_infer_request_get_tensor({name})"))?;
            let t = NonNull::new(raw_t).ok_or_else(|| format!("tensor '{name}' is null"))?;
            let mut count: usize = 0;
            // SAFETY: queries element count of the valid tensor.
            let st_sz = unsafe { (api.tensor_get_size)(t.as_ptr(), &mut count) };
            if let Err(e) = api.check(st_sz, &format!("ov_tensor_get_size({name})")) {
                unsafe { (api.tensor_free)(t.as_ptr()) };
                return Err(e);
            }
            Ok((t, count))
        };

        let cleanup_req = || unsafe {
            (api.infer_request_free)(request.as_ptr());
            (api.compiled_model_free)(compiled.as_ptr());
            (api.core_free)(core.as_ptr());
        };

        let (spec_tensor, spec_len) = match get_tensor("spec") {
            Ok(v) => v,
            Err(e) => {
                cleanup_req();
                return Err(e);
            }
        };
        let (state_in_tensor, state_in_len) = match get_tensor("state_in") {
            Ok(v) => v,
            Err(e) => {
                unsafe { (api.tensor_free)(spec_tensor.as_ptr()) };
                cleanup_req();
                return Err(e);
            }
        };
        let (spec_e_tensor, spec_e_len) = match get_tensor("spec_e") {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    (api.tensor_free)(spec_tensor.as_ptr());
                    (api.tensor_free)(state_in_tensor.as_ptr());
                }
                cleanup_req();
                return Err(e);
            }
        };
        let (state_out_tensor, state_out_len) = match get_tensor("state_out") {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    (api.tensor_free)(spec_tensor.as_ptr());
                    (api.tensor_free)(state_in_tensor.as_ptr());
                    (api.tensor_free)(spec_e_tensor.as_ptr());
                }
                cleanup_req();
                return Err(e);
            }
        };

        let model = OpenVinoModel {
            api,
            core,
            compiled,
            request,
            spec_tensor,
            state_in_tensor,
            spec_e_tensor,
            state_out_tensor,
            state_size: state_in_len,
        };

        if spec_len != SPEC_LEN || spec_e_len != SPEC_LEN {
            return Err(format!(
                "OpenVINO spectrum tensor size mismatch (spec={spec_len}, spec_e={spec_e_len}, expected {SPEC_LEN})"
            ));
        }
        if state_in_len == 0 || state_in_len != state_out_len {
            return Err(format!(
                "OpenVINO state tensor size mismatch (state_in={state_in_len}, state_out={state_out_len})"
            ));
        }
        if let Some(meta_sz) = meta
            .get("state_size")
            .and_then(|s| s.trim().parse::<usize>().ok())
        {
            if meta_sz != state_in_len {
                return Err(format!(
                    "OpenVINO state_in size {state_in_len} does not match ONNX metadata state_size {meta_sz}"
                ));
            }
        }

        let init_state = build_init_state_from_map(&meta, state_in_len);
        Ok((model, init_state))
    }

    pub(crate) fn run(
        &mut self,
        spec: &[f32; SPEC_LEN],
        state_in: &[f32],
        spec_e: &mut [f32; SPEC_LEN],
        state_out: &mut Vec<f32>,
    ) -> Result<(), String> {
        if state_in.len() != self.state_size {
            return Err(format!(
                "state_in length {} does not match model state_size {}",
                state_in.len(),
                self.state_size
            ));
        }

        unsafe {
            let mut spec_ptr: *mut c_void = std::ptr::null_mut();
            self.api.check(
                (self.api.tensor_data)(self.spec_tensor.as_ptr(), &mut spec_ptr),
                "ov_tensor_data(spec)",
            )?;
            if spec_ptr.is_null() {
                return Err("ov_tensor_data(spec) returned null".into());
            }
            std::ptr::copy_nonoverlapping(spec.as_ptr(), spec_ptr.cast::<f32>(), SPEC_LEN);

            let mut state_in_ptr: *mut c_void = std::ptr::null_mut();
            self.api.check(
                (self.api.tensor_data)(self.state_in_tensor.as_ptr(), &mut state_in_ptr),
                "ov_tensor_data(state_in)",
            )?;
            if state_in_ptr.is_null() {
                return Err("ov_tensor_data(state_in) returned null".into());
            }
            std::ptr::copy_nonoverlapping(
                state_in.as_ptr(),
                state_in_ptr.cast::<f32>(),
                self.state_size,
            );

            self.api.check(
                (self.api.infer_request_infer)(self.request.as_ptr()),
                "ov_infer_request_infer",
            )?;

            let mut spec_e_ptr: *mut c_void = std::ptr::null_mut();
            self.api.check(
                (self.api.tensor_data)(self.spec_e_tensor.as_ptr(), &mut spec_e_ptr),
                "ov_tensor_data(spec_e)",
            )?;
            if spec_e_ptr.is_null() {
                return Err("ov_tensor_data(spec_e) returned null".into());
            }
            std::ptr::copy_nonoverlapping(spec_e_ptr.cast::<f32>(), spec_e.as_mut_ptr(), SPEC_LEN);

            let mut state_out_ptr: *mut c_void = std::ptr::null_mut();
            self.api.check(
                (self.api.tensor_data)(self.state_out_tensor.as_ptr(), &mut state_out_ptr),
                "ov_tensor_data(state_out)",
            )?;
            if state_out_ptr.is_null() {
                return Err("ov_tensor_data(state_out) returned null".into());
            }
            let out_slice =
                std::slice::from_raw_parts(state_out_ptr.cast::<f32>(), self.state_size);
            state_out.clear();
            state_out.extend_from_slice(out_slice);
        }
        Ok(())
    }
}

fn parse_csv_f32(s: &str) -> Vec<f32> {
    s.split(',')
        .filter_map(|t| t.trim().parse::<f32>().ok())
        .collect()
}

pub(crate) fn build_init_state_from_map(
    meta: &BTreeMap<String, String>,
    state_size: usize,
) -> Vec<f32> {
    let erb_sz: usize = meta
        .get("erb_norm_state_size")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(481);
    let spec_sz: usize = meta
        .get("spec_norm_state_size")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(96);
    let erb_init = meta
        .get("erb_norm_init")
        .map(|s| parse_csv_f32(s))
        .unwrap_or_default();
    let spec_init = meta
        .get("spec_norm_init")
        .map(|s| parse_csv_f32(s))
        .unwrap_or_default();

    let mut init_state = vec![0f32; state_size];
    if erb_init.len() == erb_sz && erb_sz <= state_size {
        init_state[0..erb_sz].copy_from_slice(&erb_init);
    }
    if spec_init.len() == spec_sz && erb_sz + spec_sz <= state_size {
        init_state[erb_sz..erb_sz + spec_sz].copy_from_slice(&spec_init);
    }
    init_state
}

/// Extract `ModelProto.metadata_props` (field 14, repeated `StringStringEntryProto`)
/// directly from serialized ONNX protobuf bytes so OpenVINO NPU initialization can
/// seed `init_state` without spinning up an ONNX Runtime session first.
pub(crate) fn parse_onnx_custom_metadata(bytes: &[u8]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut pos = 0usize;
    while let Some((tag, next)) = read_varint(bytes, pos) {
        pos = next;
        let field_num = tag >> 3;
        let wire_type = (tag & 0x07) as u8;
        match wire_type {
            0 => {
                let Some((_, n)) = read_varint(bytes, pos) else {
                    break;
                };
                pos = n;
            }
            1 => {
                let Some(n) = pos.checked_add(8).filter(|&n| n <= bytes.len()) else {
                    break;
                };
                pos = n;
            }
            2 => {
                let Some((len, n)) = read_varint(bytes, pos) else {
                    break;
                };
                let len = len as usize;
                let Some(end) = n.checked_add(len).filter(|&e| e <= bytes.len()) else {
                    break;
                };
                if field_num == 14 {
                    if let Some((k, v)) = parse_string_entry(&bytes[n..end]) {
                        map.insert(k, v);
                    }
                }
                pos = end;
            }
            5 => {
                let Some(n) = pos.checked_add(4).filter(|&n| n <= bytes.len()) else {
                    break;
                };
                pos = n;
            }
            _ => break,
        }
    }
    map
}

fn parse_string_entry(sub: &[u8]) -> Option<(String, String)> {
    let mut key = None;
    let mut val = None;
    let mut pos = 0usize;
    while let Some((tag, next)) = read_varint(sub, pos) {
        pos = next;
        let field_num = tag >> 3;
        let wire_type = (tag & 0x07) as u8;
        if wire_type != 2 {
            return None;
        }
        let (len, n) = read_varint(sub, pos)?;
        let end = n.checked_add(len as usize)?;
        if end > sub.len() {
            return None;
        }
        let s = std::str::from_utf8(&sub[n..end]).ok()?.to_string();
        match field_num {
            1 => key = Some(s),
            2 => val = Some(s),
            _ => {}
        }
        pos = end;
    }
    Some((key?, val.unwrap_or_default()))
}

fn read_varint(bytes: &[u8], mut pos: usize) -> Option<(u64, usize)> {
    let mut result = 0u64;
    let mut shift = 0u32;
    while pos < bytes.len() && shift < 64 {
        let b = bytes[pos];
        pos += 1;
        result |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Some((result, pos));
        }
        shift += 7;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npu_inference_round_trips_words() {
        for (word, expected) in [
            ("auto", NpuInference::Auto),
            ("npu", NpuInference::Npu),
            ("onnx", NpuInference::Onnx),
        ] {
            let parsed = NpuInference::parse(word).expect("valid word");
            assert_eq!(parsed, expected);
            assert_eq!(parsed.word(), word);
        }
        assert_eq!(NpuInference::parse("openvino"), Some(NpuInference::Npu));
        assert_eq!(NpuInference::parse("cpu"), Some(NpuInference::Onnx));
        assert_eq!(NpuInference::parse("unknown"), None);
    }

    #[test]
    fn protobuf_metadata_scanner_extracts_model_props() {
        // Construct a minimal ModelProto with field 14 (tag 0x72) containing
        // key="state_size", value="56436" and erb_norm_init / spec_norm_init.
        fn encode_entry(k: &str, v: &str) -> Vec<u8> {
            let mut sub = Vec::new();
            sub.push(0x0a);
            sub.push(k.len() as u8);
            sub.extend_from_slice(k.as_bytes());
            sub.push(0x12);
            sub.push(v.len() as u8);
            sub.extend_from_slice(v.as_bytes());
            let mut outer = vec![0x72, sub.len() as u8];
            outer.extend_from_slice(&sub);
            outer
        }

        let mut proto = Vec::new();
        proto.extend_from_slice(&encode_entry("state_size", "8"));
        proto.extend_from_slice(&encode_entry("erb_norm_state_size", "2"));
        proto.extend_from_slice(&encode_entry("spec_norm_state_size", "2"));
        proto.extend_from_slice(&encode_entry("erb_norm_init", "0.25, 0.5"));
        proto.extend_from_slice(&encode_entry("spec_norm_init", "1.0, 2.0"));

        let meta = parse_onnx_custom_metadata(&proto);
        assert_eq!(meta.get("state_size").map(String::as_str), Some("8"));
        let state = build_init_state_from_map(&meta, 8);
        assert_eq!(state, vec![0.25, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn npu_engine_display_and_predicate() {
        assert!(NpuEngine::Npu.is_npu());
        assert!(!NpuEngine::Onnx.is_npu());
        assert_eq!(NpuEngine::Npu.to_string(), "openvino npu");
        assert_eq!(NpuEngine::Onnx.to_string(), "onnx");
    }

    #[test]
    fn openvino_load_reports_reason_when_runtime_absent() {
        let missing = Path::new("nonexistent_model.onnx");
        let err = OpenVinoModel::load(missing, NpuInference::Onnx, None)
            .err()
            .expect("expected error");
        assert_eq!(err, "ONNX requested");
    }
}
