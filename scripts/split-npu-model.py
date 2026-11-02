#!/usr/bin/env python3
"""Split a DPDFNet ONNX model into a CPU front end and an NPU body.

The Intel NPU compiles the whole graph in FP16 and gets the input feature block
(Pow -> ReduceSum -> Sqrt -> Add -> Log -> Div) numerically wrong, which breaks
the recurrent state within a few hops. The convolution and GRU body is accurate
on the NPU. This script cuts the graph at the first Conv node:

  <stem>.front.onnx   inputs spec, state_in   -> cut tensors   (runs on CPU)
  <stem>.body.onnx    inputs cut tensors      -> spec_e, state_out (runs on NPU)
  <stem>.split.txt    one body input name per line

Usage: python scripts/split-npu-model.py MODEL.onnx [MODEL2.onnx ...]
Requires the `onnx` package.
"""
import sys
from pathlib import Path

import onnx
from onnx import utils


def split(model_path: Path) -> None:
    model = onnx.load(str(model_path))
    nodes = list(model.graph.node)
    first_conv = next((i for i, n in enumerate(nodes) if n.op_type == "Conv"), None)
    if first_conv is None or first_conv == 0:
        raise SystemExit(f"{model_path}: no Conv node to split at")

    inits = {i.name for i in model.graph.initializer}
    front_out = {o for n in nodes[:first_conv] for o in n.output}
    graph_in = {i.name for i in model.graph.input}

    cut = []
    for node in nodes[first_conv:]:
        for t in node.input:
            if t and t not in inits and (t in front_out or t in graph_in) and t not in cut:
                cut.append(t)
    front_cut = [t for t in cut if t in front_out]
    unexpected = [t for t in cut if t in graph_in and t not in ("spec", "state_in")]
    if unexpected or not front_cut:
        raise SystemExit(f"{model_path}: unexpected cut {cut}")

    stem = model_path.with_suffix("")
    front = Path(f"{stem}.front.onnx")
    body = Path(f"{stem}.body.onnx")
    utils.extract_model(str(model_path), str(front), ["spec", "state_in"], front_cut, check_model=False)
    utils.extract_model(str(model_path), str(body), cut, ["spec_e", "state_out"], check_model=False)
    Path(f"{stem}.split.txt").write_text("\n".join(cut) + "\n", encoding="utf-8")
    print(f"{model_path.name}: split at node {first_conv}; {len(front_cut)} front tensors + "
          f"{len(cut) - len(front_cut)} direct inputs")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        split(Path(arg))
