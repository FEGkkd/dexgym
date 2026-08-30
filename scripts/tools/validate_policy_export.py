"""Validate exported policy files against a DexterousHands contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _shape_from_value(value) -> list[int | str | None]:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--observation-dim", type=int, default=207)
    parser.add_argument("--action-dim", type=int, default=20)
    args = parser.parse_args()

    jit_path = args.export_dir / "policy.pt"
    onnx_path = args.export_dir / "policy.onnx"
    if not jit_path.is_file():
        raise FileNotFoundError(jit_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    policy = torch.jit.load(str(jit_path), map_location="cpu").eval()
    jit_output = policy(torch.zeros((1, args.observation_dim), dtype=torch.float32))
    if tuple(jit_output.shape) != (1, args.action_dim):
        raise AssertionError(
            f"JIT output shape {tuple(jit_output.shape)} != {(1, args.action_dim)}"
        )

    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    if len(onnx_model.graph.input) != 1 or len(onnx_model.graph.output) != 1:
        raise AssertionError("Expected one ONNX input and one output.")
    input_shape = _shape_from_value(onnx_model.graph.input[0])
    output_shape = _shape_from_value(onnx_model.graph.output[0])
    expected_input = [1, args.observation_dim]
    expected_output = [1, args.action_dim]
    if input_shape != expected_input or output_shape != expected_output:
        raise AssertionError(
            f"ONNX shapes input={input_shape}, output={output_shape}; "
            f"expected input={expected_input}, output={expected_output}"
        )

    print(
        "POLICY_EXPORT_OK "
        f"jit={jit_path} onnx={onnx_path} "
        f"input={input_shape} output={output_shape}"
    )


if __name__ == "__main__":
    main()
