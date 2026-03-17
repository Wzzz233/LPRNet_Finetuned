#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from pathlib import Path
import sys


DEFAULT_FOCUS_OPS = [
    "NMSRotated",
    "GatherElements",
    "NonMaxSuppression",
    "TopK",
    "Range",
    "NonZero",
    "Where",
    "GridSample",
]


def load_onnx_model(path: Path):
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required. Install it with `pip install onnx`.") from exc
    return onnx.load(str(path))


def summarize_onnx_ops(model, focus_ops=None):
    focus_ops = list(focus_ops or [])
    counter = Counter(node.op_type for node in model.graph.node)
    focus_presence = {name: counter.get(name, 0) for name in focus_ops}
    return {
        "node_count": int(sum(counter.values())),
        "unique_op_count": len(counter),
        "op_counts": dict(sorted(counter.items())),
        "focus_ops": focus_presence,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect ONNX op types and highlight RKNN risk ops.")
    parser.add_argument("onnx", help="Path to ONNX model.")
    parser.add_argument(
        "--focus-ops",
        nargs="*",
        default=DEFAULT_FOCUS_OPS,
        help="Op types to highlight. Defaults to common RKNN risk ops.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Optional path to write a JSON summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"[ERROR] ONNX not found: {onnx_path}")
        return 1

    model = load_onnx_model(onnx_path)
    summary = summarize_onnx_ops(model, focus_ops=args.focus_ops)

    print(f"[ONNX] {onnx_path.resolve()}")
    print(f"[nodes] total={summary['node_count']} unique_ops={summary['unique_op_count']}")
    print("[focus]")
    for name in args.focus_ops:
        print(f"  {name}: {summary['focus_ops'].get(name, 0)}")

    print("[op_counts]")
    for op_type, count in summary["op_counts"].items():
        print(f"  {op_type}: {count}")

    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] JSON saved: {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
