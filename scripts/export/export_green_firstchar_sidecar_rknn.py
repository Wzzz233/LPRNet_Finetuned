#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import models

from rknn.api import RKNN


PROVINCE_CHARS = [
    "京", "津", "冀", "晋", "蒙", "辽", "吉", "黑",
    "沪", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
    "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云",
    "藏", "陕", "甘", "青", "宁", "新", "渝",
]


def build_model():
    model = models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(PROVINCE_CHARS))
    return model


def load_model(checkpoint):
    model = build_model()
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def export_onnx(model, onnx_path):
    dummy = torch.zeros(1, 1, 72, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
    )


def export_rknn(onnx_path, rknn_path):
    rknn = RKNN(verbose=False)
    try:
        # Board code feeds UINT8 grayscale. This turns 0..255 into the 0..1 range used in training.
        ret = rknn.config(
            target_platform="rk3568",
            mean_values=[[0]],
            std_values=[[255]],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")
        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")
    finally:
        rknn.release()


def smoke_rknn(rknn_path):
    rknn = RKNN(verbose=False)
    try:
        ret = rknn.load_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.load_rknn failed: {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            return {"runtime": "unavailable", "init_runtime_ret": int(ret)}
        x = np.zeros((1, 72, 224, 1), dtype=np.uint8)
        outs = rknn.inference(inputs=[x])
        if not outs:
            raise RuntimeError("rknn.inference returned no outputs")
        arr = np.asarray(outs[0])
        return {
            "runtime": "ok",
            "output_shape": list(arr.shape),
            "output_min": float(arr.min()),
            "output_max": float(arr.max()),
        }
    finally:
        rknn.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="artifacts/green_firstchar_sidecar")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / f"{args.name}.onnx"
    rknn_path = out_dir / f"{args.name}_rk3568_fp16.rknn"
    report_path = out_dir / f"{args.name}_export_report.json"

    model = load_model(checkpoint)
    export_onnx(model, onnx_path)
    export_rknn(onnx_path, rknn_path)
    smoke = smoke_rknn(rknn_path)

    report = {
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "rknn": str(rknn_path),
        "input": "uint8 grayscale NHWC on board, logical size 224x72",
        "training_scale": "gray / 255.0",
        "rknn_preprocess": {"mean_values": [[0]], "std_values": [[255]]},
        "classes": PROVINCE_CHARS,
        "smoke": smoke,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
