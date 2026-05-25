#!/usr/bin/env python3
import argparse
from pathlib import Path

from rknn.api import RKNN


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert a graph-normalized B2D ONNX to RKNN without RKNN runtime mean/std preprocessing."
    )
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-platform", default="rk3568")
    ap.add_argument("--float-dtype", default="float16")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    rknn = RKNN(verbose=args.verbose)
    try:
        ret = rknn.config(
            target_platform=args.target_platform,
            float_dtype=args.float_dtype,
            optimization_level=0,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")
    finally:
        rknn.release()

    print(f"[OK] saved {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
