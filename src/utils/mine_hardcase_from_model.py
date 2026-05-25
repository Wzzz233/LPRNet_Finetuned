#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

from eval_lpr_detailed import evaluate as run_detailed_eval

CONFUSABLE_TAIL = {"I", "L", "V", "1"}


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {v}")


def read_label_rows(path: Path):
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, text = line.split(maxsplit=1)
            if rel_path in seen:
                continue
            seen.add(rel_path)
            rows.append((rel_path, text))
    return rows


def write_label_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rel_path, text in rows:
            f.write(f"{rel_path} {text}\n")


def path_set(rows):
    return {rel_path for rel_path, _ in rows}


def relative_ccpd_path(image_path: str, ccpd_root: Path):
    raw = str(image_path).replace("\\", "/")
    p = Path(raw)
    if p.is_absolute():
        try:
            return p.relative_to(ccpd_root).as_posix()
        except ValueError:
            pass
    marker = "/CCPD2019/"
    if marker in raw:
        return raw.split(marker, 1)[1]
    return raw.lstrip("./")


def edit_distance(a: str, b: str):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = cur
    return prev[-1]


def pos_mismatch_count(gt: str, pred: str, start_pos: int):
    gt = gt or ""
    pred = pred or ""
    max_len = max(len(gt), len(pred))
    mismatches = 0
    for pos in range(start_pos, max_len):
        gt_ch = gt[pos] if pos < len(gt) else None
        pred_ch = pred[pos] if pos < len(pred) else None
        if gt_ch != pred_ch:
            mismatches += 1
    return mismatches


def is_tail_confusable(gt: str, pred: str):
    pred = pred or ""
    gt = gt or ""
    if not pred or pred == gt:
        return False
    gt_tail = gt[-1] if gt else ""
    return pred[-1] in CONFUSABLE_TAIL and pred[-1] != gt_tail


def severity_features(gt: str, pred: str):
    pred = pred or ""
    gt = gt or ""
    dist = edit_distance(gt, pred)
    length_gap = abs(len(gt) - len(pred))
    length_mismatch = len(gt) != len(pred)
    province_wrong = (not pred) or (not gt) or pred[0] != gt[0]
    short_pred = len(pred) <= 4
    empty_pred = pred == ""
    over_predict = len(pred) > len(gt)
    pos2_wrong = len(gt) > 1 and (len(pred) <= 1 or pred[1] != gt[1])
    pos3plus_mismatch_count = pos_mismatch_count(gt, pred, start_pos=2)
    pos3plus_error = pos3plus_mismatch_count > 0
    tail_confusable = is_tail_confusable(gt, pred)
    score = (
        dist * 10
        + pos3plus_mismatch_count * 12
        + (24 if length_mismatch else 0)
        + length_gap * 8
        + (15 if tail_confusable else 0)
        + (10 if province_wrong else 0)
        + (6 if pos2_wrong else 0)
        + (12 if short_pred and pred != gt else 0)
        + (18 if empty_pred else 0)
        + (6 if over_predict else 0)
    )
    return {
        "edit_distance": dist,
        "length_gap": length_gap,
        "length_mismatch": length_mismatch,
        "province_wrong": province_wrong,
        "pos2_wrong": pos2_wrong,
        "pos3plus_mismatch_count": pos3plus_mismatch_count,
        "pos3plus_error": pos3plus_error,
        "tail_confusable": tail_confusable,
        "short_pred": short_pred,
        "empty_pred": empty_pred,
        "over_predict": over_predict,
        "severity_score": score,
    }


def ensure_no_overlap(train_rows, val_rows, extra_sets):
    train_set = path_set(train_rows)
    val_set = path_set(val_rows)
    overlap = {
        "train_vs_val": len(train_set & val_set),
    }
    for tag, other_set in extra_sets.items():
        overlap[tag] = len(train_set & other_set)
        overlap[f"val_vs_{tag.split('train_vs_', 1)[-1]}"] = len(val_set & other_set)
    return overlap


def build_eval_args(args, bad_case_topk: int):
    return SimpleNamespace(
        img_size=[94, 24],
        test_img_dirs=args.test_img_dirs,
        txt_file=args.source_train_txt,
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
        pretrained_model=args.model,
        out_json="",
        bad_case_topk=bad_case_topk,
    )


def bucket_counts(items):
    return {
        "count": len(items),
        "length_mismatch_count": sum(1 for item in items if item["length_mismatch"]),
        "pos2_wrong_count": sum(1 for item in items if item["pos2_wrong"]),
        "pos3plus_error_count": sum(1 for item in items if item["pos3plus_error"]),
        "pos3plus_mismatch_sum": sum(item["pos3plus_mismatch_count"] for item in items),
        "tail_confusable_count": sum(1 for item in items if item["tail_confusable"]),
        "province_wrong_count": sum(1 for item in items if item["province_wrong"]),
        "short_pred_count": sum(1 for item in items if item["short_pred"]),
        "empty_pred_count": sum(1 for item in items if item["empty_pred"]),
        "over_predict_count": sum(1 for item in items if item["over_predict"]),
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Mine next-round hardcases from a trained LPRNet model or an eval json.")
    ap.add_argument("--model", required=True, help="model weights used for hardcase mining")
    ap.add_argument("--source-train-txt", required=True, help="source train labels to mine from")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-json", default="", help="optional existing eval json; if absent run model eval on source train txt")
    ap.add_argument("--hard-val", default="", help="optional hard val txt for overlap reporting")
    ap.add_argument("--hard-test", default="", help="optional hard test txt for overlap reporting")
    ap.add_argument("--selected-train", type=int, default=10000)
    ap.add_argument("--selected-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260319)
    ap.add_argument("--test-img-dirs", default="./CCPD2019")
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
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_label_rows(Path(args.source_train_txt))
    source_map = {rel_path: text for rel_path, text in source_rows}

    eval_json_path = Path(args.eval_json) if args.eval_json else out_dir / "source_train_eval.json"
    if args.eval_json:
        report = json.loads(eval_json_path.read_text(encoding="utf-8"))
    else:
        eval_args = build_eval_args(args, bad_case_topk=len(source_rows))
        report = run_detailed_eval(eval_args)
        eval_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ccpd_root = Path(args.test_img_dirs.split(",")[0]).resolve()
    candidates = []
    rng = random.Random(args.seed)
    for item in report.get("bad_cases", []):
        rel_path = relative_ccpd_path(item["image_path"], ccpd_root)
        if rel_path not in source_map:
            continue
        gt_text = item.get("gt", source_map[rel_path])
        pred_text = item.get("pred", "")
        features = severity_features(gt_text, pred_text)
        candidates.append({
            "rel_path": rel_path,
            "gt": gt_text,
            "pred": pred_text,
            **features,
            "_tie": rng.random(),
        })

    total_needed = args.selected_train + args.selected_val
    if len(candidates) < total_needed:
        raise RuntimeError(
            f"not enough mined wrong cases: need={total_needed}, have={len(candidates)}; "
            f"eval_json={eval_json_path}"
        )

    candidates.sort(
        key=lambda x: (
            -int(x["length_mismatch"]),
            -int(x["pos3plus_error"]),
            -int(x["tail_confusable"]),
            -x["pos3plus_mismatch_count"],
            -x["severity_score"],
            -x["edit_distance"],
            -x["length_gap"],
            x["_tie"],
            x["rel_path"],
        )
    )

    val_candidates = candidates[:args.selected_val]
    train_candidates = candidates[args.selected_val:args.selected_val + args.selected_train]
    train_rows = [(item["rel_path"], source_map[item["rel_path"]]) for item in train_candidates]
    val_rows = [(item["rel_path"], source_map[item["rel_path"]]) for item in val_candidates]

    extra_sets = {}
    if args.hard_val:
        extra_sets["train_vs_hard_val"] = path_set(read_label_rows(Path(args.hard_val)))
    if args.hard_test:
        extra_sets["train_vs_hard_test"] = path_set(read_label_rows(Path(args.hard_test)))
    overlap = ensure_no_overlap(train_rows, val_rows, extra_sets)

    train_txt = out_dir / "hardcase_train.txt"
    val_txt = out_dir / "hardcase_val.txt"
    write_label_rows(train_txt, train_rows)
    write_label_rows(val_txt, val_rows)

    summary = {
        "seed": args.seed,
        "model": args.model,
        "source_train_txt": args.source_train_txt,
        "eval_json": str(eval_json_path),
        "sample_count": int(report.get("sample_count", 0)),
        "wrong_candidates_total": len(candidates),
        "selected_train": len(train_rows),
        "selected_val": len(val_rows),
        "overlap": overlap,
        "outputs": {
            "hardcase_train_txt": str(train_txt),
            "hardcase_val_txt": str(val_txt),
        },
        "severity": {
            "max_score": max(item["severity_score"] for item in candidates),
            "min_score": min(item["severity_score"] for item in candidates),
            "train_mean_score": sum(item["severity_score"] for item in train_candidates) / max(1, len(train_candidates)),
            "val_mean_score": sum(item["severity_score"] for item in val_candidates) / max(1, len(val_candidates)),
        },
        "bucket_counts": {
            "candidate_pool": bucket_counts(candidates),
            "selected_train": bucket_counts(train_candidates),
            "selected_val": bucket_counts(val_candidates),
        },
        "top_examples": [
            {
                "rel_path": item["rel_path"],
                "gt": item["gt"],
                "pred": item["pred"],
                "severity_score": item["severity_score"],
                "edit_distance": item["edit_distance"],
                "length_gap": item["length_gap"],
                "length_mismatch": item["length_mismatch"],
                "province_wrong": item["province_wrong"],
                "pos2_wrong": item["pos2_wrong"],
                "pos3plus_mismatch_count": item["pos3plus_mismatch_count"],
                "pos3plus_error": item["pos3plus_error"],
                "tail_confusable": item["tail_confusable"],
                "short_pred": item["short_pred"],
                "empty_pred": item["empty_pred"],
                "over_predict": item["over_predict"],
            }
            for item in candidates[:10]
        ],
    }
    summary_path = out_dir / "hardcase_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
