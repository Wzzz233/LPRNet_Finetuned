#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from inspect_onnx_ops import DEFAULT_FOCUS_OPS, load_onnx_model, summarize_onnx_ops


def detect_package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def export_yolov8_obb_to_onnx(weights, output_path, imgsz, opset, simplify):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Install it with `pip install ultralytics`.") from exc

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
    )
    exported_path = Path(str(exported))
    if not exported_path.exists():
        raise FileNotFoundError(f"Export reported success but file not found: {exported_path}")

    if exported_path.resolve() != output_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(exported_path.read_bytes())
    return output_path


def rknn_probe(onnx_path, target_platform, run_build, verbose, export_rknn_path=None):
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError("rknn-toolkit2 is required. Install it with `pip install rknn-toolkit2`.") from exc

    result = {
        "config_ret": None,
        "load_onnx_ret": None,
        "build_ret": None,
        "export_rknn_ret": None,
    }
    rknn = RKNN(verbose=verbose)
    try:
        result["config_ret"] = int(
            rknn.config(
                target_platform=target_platform,
                optimization_level=0,
            )
        )
        if result["config_ret"] != 0:
            return result

        result["load_onnx_ret"] = int(rknn.load_onnx(model=str(onnx_path)))
        if result["load_onnx_ret"] != 0 or not run_build:
            return result

        result["build_ret"] = int(rknn.build(do_quantization=False))
        if result["build_ret"] != 0 or export_rknn_path is None:
            return result

        export_rknn_path.parent.mkdir(parents=True, exist_ok=True)
        result["export_rknn_ret"] = int(rknn.export_rknn(str(export_rknn_path)))
        return result
    finally:
        rknn.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLOv8n-OBB ONNX/RKNN compatibility for RK3568.")
    parser.add_argument("--weights", default="yolov8n-obb.pt", help="YOLOv8 OBB weight path or model name.")
    parser.add_argument("--onnx", default="./artifacts/yolov8n_obb/yolov8n-obb.onnx", help="Output ONNX path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset.")
    parser.add_argument("--target-platform", default="rk3568", help="RKNN target platform.")
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX export and reuse --onnx.")
    parser.add_argument("--skip-build", action="store_true", help="Stop after RKNN load_onnx.")
    parser.add_argument("--simplify", action="store_true", help="Enable Ultralytics ONNX simplification.")
    parser.add_argument("--verbose", action="store_true", help="Enable RKNN verbose output.")
    parser.add_argument(
        "--export-rknn",
        default="./artifacts/yolov8n_obb/yolov8n-obb.rknn",
        help="Optional RKNN output path. Pass empty string to skip export.",
    )
    parser.add_argument(
        "--report-json",
        default="./artifacts/yolov8n_obb/yolov8n_obb_rk3568_report.json",
        help="Path to save the combined report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    report_path = Path(args.report_json)
    report = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "ultralytics": detect_package_version("ultralytics"),
            "onnx": detect_package_version("onnx"),
            "rknn-toolkit2": detect_package_version("rknn-toolkit2"),
        },
        "inputs": {
            "weights": args.weights,
            "onnx": str(onnx_path),
            "imgsz": args.imgsz,
            "opset": args.opset,
            "target_platform": args.target_platform,
            "skip_export": args.skip_export,
            "skip_build": args.skip_build,
            "simplify": args.simplify,
            "export_rknn": args.export_rknn,
        },
        "steps": {},
    }

    try:
        if args.skip_export:
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX not found: {onnx_path}")
            print(f"[export] skipped, reusing {onnx_path.resolve()}")
        else:
            print("[export] exporting YOLOv8 OBB to ONNX")
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            exported_path = export_yolov8_obb_to_onnx(
                args.weights,
                onnx_path,
                imgsz=args.imgsz,
                opset=args.opset,
                simplify=args.simplify,
            )
            print(f"[export] saved: {exported_path.resolve()}")
        report["steps"]["export"] = {"status": "ok"}
    except Exception as exc:
        report["steps"]["export"] = {"status": "error", "error": str(exc)}
        return finalize(report, report_path, exit_code=1)

    try:
        print("[onnx] auditing ops")
        onnx_model = load_onnx_model(onnx_path)
        summary = summarize_onnx_ops(onnx_model, focus_ops=DEFAULT_FOCUS_OPS)
        report["steps"]["onnx_audit"] = {"status": "ok", **summary}
        for name, count in summary["focus_ops"].items():
            print(f"[onnx] {name}: {count}")
    except Exception as exc:
        report["steps"]["onnx_audit"] = {"status": "error", "error": str(exc)}
        return finalize(report, report_path, exit_code=1)

    try:
        print("[rknn] probing RK3568 compatibility")
        probe = rknn_probe(
            onnx_path,
            target_platform=args.target_platform,
            run_build=not args.skip_build,
            verbose=args.verbose,
            export_rknn_path=Path(args.export_rknn) if args.export_rknn else None,
        )
        report["steps"]["rknn_probe"] = {"status": "ok", **probe}
        print(
            "[rknn] ret codes: "
            f"config={probe['config_ret']} "
            f"load_onnx={probe['load_onnx_ret']} "
            f"build={probe['build_ret']} "
            f"export_rknn={probe['export_rknn_ret']}"
        )
    except Exception as exc:
        report["steps"]["rknn_probe"] = {"status": "error", "error": str(exc)}
        return finalize(report, report_path, exit_code=1)

    return finalize(report, report_path, exit_code=0)


def finalize(report, report_path, exit_code):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] saved: {report_path.resolve()}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
