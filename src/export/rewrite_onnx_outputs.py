#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite ONNX graph outputs to selected intermediate tensors.")
    parser.add_argument("--input", required=True, help="Input ONNX path.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--outputs", nargs="+", required=True, help="Tensor names to expose as graph outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    import onnx
    from onnx import helper, shape_inference

    if not input_path.exists():
        print(f"[ERROR] ONNX not found: {input_path}")
        return 1

    model = onnx.load(str(input_path))
    inferred = shape_inference.infer_shapes(model)

    value_info = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
        value_info[vi.name] = vi

    new_outputs = []
    missing = []
    for name in args.outputs:
        vi = value_info.get(name)
        if vi is None:
            missing.append(name)
            continue
        new_outputs.append(helper.make_tensor_value_info(name, vi.type.tensor_type.elem_type, [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in vi.type.tensor_type.shape.dim
        ]))

    if missing:
        print(f"[ERROR] Missing tensor(s): {missing}")
        return 1

    del model.graph.output[:]
    model.graph.output.extend(new_outputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"[OK] rewritten outputs saved: {output_path.resolve()}")
    print(f"[OK] outputs: {args.outputs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
