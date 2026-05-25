#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys

from rknn.api import RKNN


# From git latest PyTorch preprocessing:
# img = img.astype('float32'); img -= 127.5; img *= 0.0078125
# Equivalent RKNN config: (x - 127.5) / 128.0
MEAN_VALUES = [[127.5, 127.5, 127.5]]
STD_VALUES = [[128.0, 128.0, 128.0]]
FLOAT_DTYPE = "float16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LPRNet ONNX to RKNN for rk3568 (FP16 by default)."
    )
    parser.add_argument(
        "onnx_model",
        nargs="?",
        default="./LPRNet_stage3_git_latest.onnx",
        help="Input ONNX path.",
    )
    parser.add_argument(
        "--target-platform",
        default="rk3568",
        choices=["rk3562", "rk3566", "rk3568", "rk3576", "rk3588", "rv1109", "rv1126", "rk1808"],
        help="RKNN target platform.",
    )
    parser.add_argument(
        "--dtype",
        default="fp",
        choices=["fp", "i8", "u8"],
        help="Output model dtype. Use fp for FP16 build.",
    )
    parser.add_argument(
        "--output",
        default="./LPRNet_stage3_rk3568_fp16.rknn",
        help="Output RKNN path.",
    )
    parser.add_argument(
        "--dataset",
        default="./dataset.txt",
        help="Calibration dataset path (only used for i8/u8).",
    )
    parser.add_argument(
        "--input-color-order",
        default="bgr",
        choices=["bgr", "rgb"],
        help="Color order fed by runtime before RKNN.",
    )
    parser.add_argument(
        "--model-color-order",
        default="bgr",
        choices=["bgr", "rgb"],
        help="Color order used by training/model preprocessing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable RKNN verbose log.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx_model)
    output_path = Path(args.output)

    if not onnx_path.exists():
        print(f"[ERROR] ONNX not found: {onnx_path}")
        return 1

    do_quant = args.dtype in ("i8", "u8")
    if do_quant:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"[ERROR] Quant dataset not found: {dataset_path}")
            return 1
    else:
        dataset_path = None

    # If runtime input order differs from model order, request channel swap.
    # For this project (BGR->BGR), this should stay False.
    need_swap = args.input_color_order != args.model_color_order

    rknn = RKNN(verbose=args.verbose)

    print("--> Config model")
    ret = rknn.config(
        mean_values=MEAN_VALUES,
        std_values=STD_VALUES,
        target_platform=args.target_platform,
        quant_img_RGB2BGR=need_swap,
        float_dtype=FLOAT_DTYPE,
        optimization_level=0,
    )
    if ret != 0:
        print("Config model failed!")
        rknn.release()
        return ret
    print("done")

    print("--> Loading ONNX model")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        print("Load ONNX model failed!")
        rknn.release()
        return ret
    print("done")

    print("--> Building RKNN model")
    if do_quant:
        ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("Build RKNN model failed!")
        rknn.release()
        return ret
    print("done")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("--> Export RKNN model")
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        print("Export RKNN model failed!")
        rknn.release()
        return ret
    print("done")
    print(f"[OK] RKNN saved: {output_path.resolve()}")

    rknn.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
