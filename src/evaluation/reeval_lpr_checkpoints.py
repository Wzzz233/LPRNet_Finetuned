#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

from eval_lpr_detailed import evaluate as run_detailed_eval


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {v}")


def parse_args():
    ap = argparse.ArgumentParser(description="Re-evaluate all LPRNet checkpoints under a run directory.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--test-img-dirs", required=True)
    ap.add_argument("--data-mode", default="ccpd_board", choices=["standard", "ccpd_board"])
    ap.add_argument("--ocr_channel_order", default="bgr", choices=["rgb", "bgr"])
    ap.add_argument("--ocr_crop_mode", default="obb_warp", choices=["fixed", "box", "tight", "box-pad", "match", "obb_warp"])
    ap.add_argument("--ocr_resize_mode", default="letterbox", choices=["stretch", "letterbox"])
    ap.add_argument("--ocr_resize_kernel", default="nn", choices=["nn", "bilinear"])
    ap.add_argument("--ocr_preproc", default="none", choices=["none", "raw", "gray", "gray3", "bin"])
    ap.add_argument("--ocr_min_occ_ratio", default=0.90, type=float)
    ap.add_argument("--ocr_quad_pad_ratio", default=0.0, type=float)
    ap.add_argument("--test_batch_size", default=120, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--cuda", default=True, type=str2bool)
    ap.add_argument("--hard-val", default="")
    ap.add_argument("--hard-test", default="")
    ap.add_argument("--normal-val", default="")
    ap.add_argument("--normal-test", default="")
    ap.add_argument("--ranking-split", default="hard_val")
    ap.add_argument("--normal-guard-split", default="normal_val")
    ap.add_argument("--out-json", default="")
    return ap.parse_args()


def build_eval_args(args, txt_file: str, model_path: str):
    return SimpleNamespace(
        img_size=[94, 24],
        test_img_dirs=args.test_img_dirs,
        txt_file=txt_file,
        dropout_rate=0.0,
        lpr_max_len=8,
        data_mode=args.data_mode,
        ocr_channel_order=args.ocr_channel_order,
        ocr_crop_mode=args.ocr_crop_mode,
        ocr_resize_mode=args.ocr_resize_mode,
        ocr_resize_kernel=args.ocr_resize_kernel,
        ocr_preproc=args.ocr_preproc,
        ocr_min_occ_ratio=args.ocr_min_occ_ratio,
        ocr_quad_pad_ratio=args.ocr_quad_pad_ratio,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        cuda=args.cuda,
        pretrained_model=model_path,
        out_json="",
        bad_case_topk=20,
    )


def parse_stage_and_iter(path: Path):
    stage_match = re.search(r"weights_stage([A-Z])", str(path.parent))
    stage = stage_match.group(1) if stage_match else "Z"
    if path.name == "Final_LPRNet_model.pth":
        return stage, 10**12
    iter_match = re.search(r"_iteration_(\d+)\.pth$", path.name)
    if iter_match:
        return stage, int(iter_match.group(1))
    return stage, 10**11


def checkpoint_tag(path: Path):
    return f"{path.parent.name}/{path.name}"


def metric_or_zero(metrics: dict, split_name: str, key: str):
    split = metrics.get(split_name, {})
    if key == "pos3plus_alnum":
        return float(split.get("position_accuracy", {}).get("pos3plus_alnum", 0.0))
    return float(split.get(key, 0.0))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    split_map = {
        "hard_val": args.hard_val,
        "hard_test": args.hard_test,
        "normal_val": args.normal_val,
        "normal_test": args.normal_test,
    }
    split_map = {k: v for k, v in split_map.items() if v}
    if args.ranking_split not in split_map:
        raise RuntimeError(f"ranking split not available: {args.ranking_split}")
    if args.normal_guard_split not in split_map:
        raise RuntimeError(f"normal guard split not available: {args.normal_guard_split}")

    checkpoints = sorted(run_dir.glob("weights_stage*/*.pth"), key=parse_stage_and_iter)
    if not checkpoints:
        raise RuntimeError(f"no checkpoints found under {run_dir}")

    metrics_dir = run_dir / "checkpoint_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ckpt in checkpoints:
        ckpt_metrics = {}
        for split_name, txt_file in split_map.items():
            report = run_detailed_eval(build_eval_args(args, txt_file, str(ckpt)))
            ckpt_metrics[split_name] = report
            out_path = metrics_dir / f"{ckpt.parent.name}__{ckpt.stem}__{split_name}.json"
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        rows.append({
            "checkpoint": str(ckpt),
            "tag": checkpoint_tag(ckpt),
            "stage": ckpt.parent.name,
            "metrics": ckpt_metrics,
        })

    ranking_split = args.ranking_split
    normal_guard_split = args.normal_guard_split
    rows.sort(
        key=lambda row: (
            metric_or_zero(row["metrics"], normal_guard_split, "exact_plate_acc"),
            metric_or_zero(row["metrics"], ranking_split, "pos3plus_alnum"),
            metric_or_zero(row["metrics"], ranking_split, "length_correct_acc"),
            metric_or_zero(row["metrics"], ranking_split, "exact_plate_acc"),
        ),
        reverse=True,
    )
    rows.sort(
        key=lambda row: (
            metric_or_zero(row["metrics"], ranking_split, "exact_plate_acc"),
            metric_or_zero(row["metrics"], ranking_split, "length_correct_acc"),
            metric_or_zero(row["metrics"], ranking_split, "pos3plus_alnum"),
            metric_or_zero(row["metrics"], normal_guard_split, "exact_plate_acc"),
        ),
        reverse=True,
    )

    best = rows[0]
    report = {
        "run_dir": str(run_dir),
        "ranking_split": ranking_split,
        "normal_guard_split": normal_guard_split,
        "checkpoint_count": len(rows),
        "best_checkpoint": best["checkpoint"],
        "best_tag": best["tag"],
        "leaderboard": [
            {
                "checkpoint": row["checkpoint"],
                "tag": row["tag"],
                "ranking_exact": metric_or_zero(row["metrics"], ranking_split, "exact_plate_acc"),
                "ranking_length": metric_or_zero(row["metrics"], ranking_split, "length_correct_acc"),
                "ranking_pos3plus": metric_or_zero(row["metrics"], ranking_split, "pos3plus_alnum"),
                "normal_guard_exact": metric_or_zero(row["metrics"], normal_guard_split, "exact_plate_acc"),
            }
            for row in rows
        ],
    }

    out_json = Path(args.out_json) if args.out_json else run_dir / "checkpoint_scoreboard.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
