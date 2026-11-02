//! Hybrid engine (feature `npu`): the input feature block runs on ONNX Runtime
//! CPU, the convolution/GRU body runs on the OpenVINO `NPU`.
//!
//! The Intel NPU compiles the whole DPDFNet graph in FP16 and evaluates the
//! feature block (Pow, ReduceSum, Sqrt, Add, Log, Div) wrongly, which corrupts
//! the recurrent state within a few hops. The body is accurate on the NPU when
//! it is given features computed in FP32. `scripts/split-npu-model.py` cuts a
//! model at its first Conv into three sidecar files next to it:
//!
//! - `<stem>.front.onnx`: `spec`, `state_in` -> cut tensors (CPU)
//! - `<stem>.body.onnx`: cut tensors -> `spec_e`, `state_out` (NPU)
//! - `<stem>.split.txt`: the body's input names, one per line
//!
//! The front costs about 0.13 ms per hop, so the NPU carries nearly all of the
//! per-hop work.

use crate::model::session_builder;
use crate::openvino::OpenVinoModel;
use crate::stft::{FREQ_BINS, SPEC_LEN};
use ort::session::Session;
use ort::value::TensorRef;
use std::path::{Path, PathBuf};

/// Where one body input comes from each hop.
enum Feed {
    Spec,
    StateIn,
    /// Index into the front session's outputs by name.
    Front(String),
}

pub(crate) struct HybridModel {
    front: Session,
    body: OpenVinoModel,
    feeds: Vec<Feed>,
}

/// Sidecar paths for `model_path`: `(front, body, split list)`.
pub(crate) fn split_paths(model_path: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let stem = model_path.with_extension("");
    let with = |ext: &str| {
        let mut s = stem.clone().into_os_string();
        s.push(ext);
        PathBuf::from(s)
    };
    (
        with(".front.onnx"),
        with(".body.onnx"),
        with(".split.txt"),
    )
}

/// True when all three split sidecars exist for `model_path`.
pub(crate) fn split_available(model_path: &Path) -> bool {
    let (f, b, s) = split_paths(model_path);
    f.is_file() && b.is_file() && s.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_paths_sit_next_to_the_model() {
        let (front, body, list) = split_paths(Path::new("models/dpdfnet2_48khz_hr.onnx"));
        assert_eq!(front, Path::new("models/dpdfnet2_48khz_hr.front.onnx"));
        assert_eq!(body, Path::new("models/dpdfnet2_48khz_hr.body.onnx"));
        assert_eq!(list, Path::new("models/dpdfnet2_48khz_hr.split.txt"));
    }

    #[test]
    fn split_is_unavailable_without_sidecars() {
        assert!(!split_available(Path::new("no/such/model.onnx")));
    }
}

impl HybridModel {
    pub(crate) fn state_size(&self) -> usize {
        self.body.state_size()
    }

    pub(crate) fn load(
        model_path: &Path,
        cache_dir: Option<&Path>,
    ) -> Result<(HybridModel, Vec<f32>), String> {
        let (front_path, body_path, list_path) = split_paths(model_path);
        let list = std::fs::read_to_string(&list_path)
            .map_err(|e| format!("read {}: {e}", list_path.display()))?;
        let names: Vec<String> = list
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
        if names.is_empty() {
            return Err(format!("{} lists no body inputs", list_path.display()));
        }
        let feeds: Vec<Feed> = names
            .iter()
            .map(|n| match n.as_str() {
                "spec" => Feed::Spec,
                "state_in" => Feed::StateIn,
                other => Feed::Front(other.to_string()),
            })
            .collect();

        let front = session_builder()?
            .commit_from_file(&front_path)
            .map_err(|e| format!("commit_from_file({}): {e}", front_path.display()))?;
        for feed in &feeds {
            if let Feed::Front(name) = feed {
                if !front.outputs().iter().any(|o| o.name() == name) {
                    return Err(format!(
                        "{} has no output '{name}' (re-run scripts/split-npu-model.py)",
                        front_path.display()
                    ));
                }
            }
        }

        let (body, init_state) =
            OpenVinoModel::load_graph(&body_path, model_path, cache_dir, &names)?;
        Ok((HybridModel { front, body, feeds }, init_state))
    }

    pub(crate) fn run(
        &mut self,
        spec: &[f32; SPEC_LEN],
        state_in: &[f32],
        spec_e: &mut [f32; SPEC_LEN],
        state_out: &mut Vec<f32>,
    ) -> Result<(), String> {
        let spec_t = TensorRef::from_array_view(([1usize, 1, FREQ_BINS, 2], spec.as_slice()))
            .map_err(|e| e.to_string())?;
        let state_t =
            TensorRef::from_array_view(([state_in.len()], state_in)).map_err(|e| e.to_string())?;
        let outputs = self
            .front
            .run(ort::inputs! { "spec" => spec_t, "state_in" => state_t })
            .map_err(|e| e.to_string())?;

        let mut slices: Vec<&[f32]> = Vec::with_capacity(self.feeds.len());
        for feed in &self.feeds {
            slices.push(match feed {
                Feed::Spec => spec.as_slice(),
                Feed::StateIn => state_in,
                Feed::Front(name) => {
                    let (_, data) = outputs
                        .get(name.as_str())
                        .ok_or_else(|| format!("front graph has no output '{name}'"))?
                        .try_extract_tensor::<f32>()
                        .map_err(|e| e.to_string())?;
                    data
                }
            });
        }
        self.body.run_feeds(&slices, spec_e, state_out)
    }
}
