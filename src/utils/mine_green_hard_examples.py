#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.load_data import CHARS, CCPDBoardDataLoader, LPRDataLoader
from model.LPRNet import build_lprnet
from test_LPRNet import collate_fn
from eval_lpr_detailed import decode_logits


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Mine hard CCPD green training examples from model errors.")
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--img_dirs", required=True, help="comma-separated roots")
    parser.add_argument("--txt_file", required=True)
    parser.add_argument("--output_txt", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--img_size", default=[94, 24], nargs=2, type=int)
    parser.add_argument("--lpr_max_len", default=8, type=int)
    parser.add_argument("--data_mode", default="ccpd_board", choices=["standard", "ccpd_board"])
    parser.add_argument("--ocr_channel_order", default="bgr", choices=["rgb", "bgr"])
    parser.add_argument("--ocr_crop_mode", default="obb_warp", choices=["fixed", "box", "tight", "box-pad", "match", "obb_warp"])
    parser.add_argument("--ocr_resize_mode", default="letterbox", choices=["stretch", "letterbox"])
    parser.add_argument("--ocr_resize_kernel", default="nn", choices=["nn", "bilinear"])
    parser.add_argument("--ocr_preproc", default="none", choices=["none", "raw", "gray", "gray3", "bin"])
    parser.add_argument("--ocr_min_occ_ratio", default=0.90, type=float)
    parser.add_argument("--ocr_quad_pad_ratio", default=0.0, type=float)
    parser.add_argument("--decode_mode", default="greedy", choices=["greedy", "green_ctc_beam"])
    parser.add_argument("--beam_size", default=30, type=int)
    parser.add_argument("--beam_topk", default=15, type=int)
    parser.add_argument("--max_per_stratum", default=500, type=int)
    parser.add_argument("--min_edit_distance", default=2, type=int)
    parser.add_argument("--batch_size", default=120, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--cuda", default=False, type=str2bool)
    parser.add_argument("--seed", default=20260320, type=int)
    return parser.parse_args()


def edit_distance(a, b):
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if ca == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lb]


def adjacent_repeat_pairs(text):
    if not text:
        return 0
    cnt = 0
    for i in range(len(text) - 1):
        if text[i] == text[i + 1]:
            cnt += 1
    return cnt


def ne_type(text):
    if len(text) > 2 and text[2] in {"D", "F"}:
        return "small"
    if len(text) > 7 and text[7] in {"D", "F"}:
        return "large"
    return "unknown"


def repeat_bucket(text):
    pairs = adjacent_repeat_pairs(text)
    if pairs <= 0:
        return "none"
    if pairs == 1:
        return "one"
    return "multi"


def build_stratum(text):
    if not text:
        return "?|unknown|none"
    return f"{text[0]}|{ne_type(text)}|{repeat_bucket(text)}"


def build_dataset(args):
    if args.data_mode == "ccpd_board":
        return CCPDBoardDataLoader(
            args.img_dirs.split(","),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.txt_file,
            ocr_channel_order=args.ocr_channel_order,
            ocr_crop_mode=args.ocr_crop_mode,
            ocr_resize_mode=args.ocr_resize_mode,
            ocr_resize_kernel=args.ocr_resize_kernel,
            ocr_preproc=args.ocr_preproc,
            ocr_min_occ_ratio=args.ocr_min_occ_ratio,
            ocr_quad_pad_ratio=args.ocr_quad_pad_ratio,
        )
    return LPRDataLoader(args.img_dirs.split(","), args.img_size, args.lpr_max_len, txt_file=args.txt_file)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    net = build_lprnet(lpr_max_len=args.lpr_max_len, phase=False, class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    net.to(device)
    net.eval()

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    all_hard = []
    all_count = 0
    mismatch_count = 0
    skipped_near_miss_count = 0
    kept_len_mismatch_count = 0
    kept_edit_threshold_count = 0
    seen = 0
    stratum_count = Counter()

    with torch.no_grad():
        for images, labels, lengths in loader:
            start = 0
            targets = []
            for length in lengths:
                gt_ids = labels[start:start + length].tolist()
                targets.append("".join(CHARS[int(c)] for c in gt_ids))
                start += length

            if args.cuda:
                images = images.to(device)
            logits = net(images).detach().cpu().numpy()
            decoded = decode_logits(logits, args.decode_mode, args.beam_size, args.beam_topk)

            for i, pred_ids in enumerate(decoded):
                all_count += 1
                gt_text = targets[i]
                pred_text = "".join(CHARS[int(c)] for c in pred_ids)
                if pred_text == gt_text:
                    continue
                mismatch_count += 1
                ed = edit_distance(pred_text, gt_text)
                if len(pred_text) == len(gt_text) and ed < args.min_edit_distance:
                    skipped_near_miss_count += 1
                    continue
                if len(pred_text) != len(gt_text):
                    kept_len_mismatch_count += 1
                else:
                    kept_edit_threshold_count += 1
                rel_path = getattr(dataset, "img_rel_paths", [None] * len(dataset))[seen + i]
                if rel_path is None:
                    rel_path = Path(getattr(dataset, "img_paths")[seen + i]).name
                key = build_stratum(gt_text)
                stratum_count[key] += 1
                all_hard.append((key, rel_path, gt_text, pred_text, ed))
            seen += len(decoded)

    rng = random.Random(args.seed)
    buckets = defaultdict(list)
    for row in all_hard:
        buckets[row[0]].append(row)

    selected = []
    for key in sorted(buckets.keys()):
        rows = buckets[key]
        rng.shuffle(rows)
        cap = args.max_per_stratum if args.max_per_stratum > 0 else len(rows)
        selected.extend(rows[:cap])

    seen_rel = set()
    output_rows = []
    for _, rel_path, gt_text, _, _ in selected:
        if rel_path in seen_rel:
            continue
        seen_rel.add(rel_path)
        output_rows.append((rel_path, gt_text))

    output_txt = Path(args.output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as f:
        for rel_path, gt_text in output_rows:
            f.write(f"{rel_path} {gt_text}\n")

    report = {
        "model": args.pretrained_model,
        "txt_file": args.txt_file,
        "decode_mode": args.decode_mode,
        "min_edit_distance": args.min_edit_distance,
        "sample_count": all_count,
        "mismatch_count": mismatch_count,
        "skipped_near_miss_count": skipped_near_miss_count,
        "hard_candidate_count": len(all_hard),
        "kept_len_mismatch_count": kept_len_mismatch_count,
        "kept_edit_threshold_count": kept_edit_threshold_count,
        "selected_count": len(output_rows),
        "max_per_stratum": args.max_per_stratum,
        "stratum_candidate_distribution": dict(sorted(stratum_count.items())),
        "output_txt": str(output_txt),
    }

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
