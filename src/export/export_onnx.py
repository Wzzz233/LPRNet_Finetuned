#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from data.load_data import CHARS
from model.LPRNet import build_lprnet


def ensure_onnx():
    try:
        import onnx  # type: ignore
        return onnx
    except Exception:
        print("[Info] onnx not found, installing via pip ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx"])
        import onnx  # type: ignore
        return onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LPRNet stage3 checkpoint to ONNX and validate it.")
    parser.add_argument("--weights", default="./weights_red_stage3/Final_LPRNet_model.pth", help="Path to .pth weights.")
    parser.add_argument("--output", default="./LPRNet_stage3.onnx", help="Output ONNX file path.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = torch.device("cpu")
    lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    state = torch.load(str(weights_path), map_location=device)
    lprnet.load_state_dict(state)
    lprnet.to(device)
    lprnet.eval()

    # Required input shape: [N, C, H, W] = [1, 3, 24, 94]
    dummy_input = torch.randn(1, 3, 24, 94, device=device)

    torch.onnx.export(
        lprnet,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    onnx = ensure_onnx()
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)

    print("ONNX 模型导出并校验成功！")
    print(f"ONNX 路径: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
