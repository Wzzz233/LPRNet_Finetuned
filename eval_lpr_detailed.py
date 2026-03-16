#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
import argparse
import json
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data.load_data import CHARS, LPRDataLoader, CCPDBoardDataLoader
from model.LPRNet import build_lprnet
from test_LPRNet import collate_fn, greedy_decode_logits


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser(description="Detailed LPRNet evaluation with plate/length/position metrics.")
    parser.add_argument("--img_size", default=[94, 24], nargs=2, type=int)
    parser.add_argument("--test_img_dirs", required=True)
    parser.add_argument("--txt_file", required=True)
    parser.add_argument("--dropout_rate", default=0, type=float)
    parser.add_argument("--lpr_max_len", default=8, type=int)
    parser.add_argument("--data_mode", default="standard", choices=["standard", "ccpd_board"])
    parser.add_argument("--ocr_channel_order", default="bgr", choices=["rgb", "bgr"])
    parser.add_argument("--ocr_crop_mode", default="match", choices=["fixed", "box", "tight", "box-pad", "match"])
    parser.add_argument("--ocr_resize_mode", default="letterbox", choices=["stretch", "letterbox"])
    parser.add_argument("--ocr_resize_kernel", default="nn", choices=["nn", "bilinear"])
    parser.add_argument("--ocr_preproc", default="none", choices=["none", "raw", "gray", "gray3", "bin"])
    parser.add_argument("--ocr_min_occ_ratio", default=0.90, type=float)
    parser.add_argument("--test_batch_size", default=100, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--cuda", default=False, type=str2bool)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--out_json", default="")
    parser.add_argument("--bad_case_topk", default=20, type=int)
    return parser


def build_dataset(args):
    test_img_dirs = args.test_img_dirs
    if args.data_mode == "ccpd_board":
        return CCPDBoardDataLoader(
            test_img_dirs.split(","),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.txt_file,
            ocr_channel_order=args.ocr_channel_order,
            ocr_crop_mode=args.ocr_crop_mode,
            ocr_resize_mode=args.ocr_resize_mode,
            ocr_resize_kernel=args.ocr_resize_kernel,
            ocr_preproc=args.ocr_preproc,
            ocr_min_occ_ratio=args.ocr_min_occ_ratio,
        )
    return LPRDataLoader(test_img_dirs.split(","), args.img_size, args.lpr_max_len, txt_file=args.txt_file)


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def evaluate(args):
    device = torch.device("cuda:0" if args.cuda else "cpu")
    net = build_lprnet(lpr_max_len=args.lpr_max_len, phase=False, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    net.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    net.to(device)
    net.eval()

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    exact = 0
    total = 0
    length_ok = 0
    correct_chars = 0
    total_chars = 0
    pos_total = Counter()
    pos_correct = Counter()
    province_total = 0
    province_correct = 0
    province_rows = {}
    bad_cases = []
    seen_count = 0
    start_time = time.time()

    for images, labels, lengths in loader:
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start + length]
            targets.append(label)
            start += length
        targets = [el.numpy() for el in targets]

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        with torch.no_grad():
            prebs = net(images)
        decoded = greedy_decode_logits(prebs.cpu().detach().numpy())

        for i, pred_ids in enumerate(decoded):
            gt_ids = targets[i].tolist()
            pred_text = "".join(CHARS[int(c)] for c in pred_ids)
            gt_text = "".join(CHARS[int(c)] for c in gt_ids)
            total += 1

            if len(pred_ids) == len(gt_ids):
                length_ok += 1
            if pred_text == gt_text:
                exact += 1

            if gt_text:
                province_total += 1
                bucket = province_rows.setdefault(gt_text[0], {
                    "sample_count": 0,
                    "exact_plate_correct": 0,
                    "first_char_correct": 0,
                })
                bucket["sample_count"] += 1
                if pred_text and pred_text[0] == gt_text[0]:
                    province_correct += 1
                    bucket["first_char_correct"] += 1
                if pred_text == gt_text:
                    bucket["exact_plate_correct"] += 1

            max_len = max(len(gt_ids), len(pred_ids))
            for pos in range(max_len):
                bucket = "pos1_province" if pos == 0 else ("pos2_alpha" if pos == 1 else "pos3plus_alnum")
                if pos < len(gt_ids):
                    pos_total[bucket] += 1
                    total_chars += 1
                    if pos < len(pred_ids) and pred_ids[pos] == gt_ids[pos]:
                        pos_correct[bucket] += 1
                        correct_chars += 1

            if pred_text != gt_text and len(bad_cases) < args.bad_case_topk:
                img_name = dataset.img_paths[seen_count + i]
                bad_cases.append({
                    "image_path": img_name,
                    "gt": gt_text,
                    "pred": pred_text,
                })
        seen_count += len(decoded)

    elapsed = time.time() - start_time
    report = {
        "model": args.pretrained_model,
        "data_mode": args.data_mode,
        "sample_count": total,
        "exact_plate_acc": safe_div(exact, total),
        "length_correct_acc": safe_div(length_ok, total),
        "char_acc": safe_div(correct_chars, total_chars),
        "province_first_char_acc": safe_div(province_correct, province_total),
        "position_accuracy": {
            "pos1_province": safe_div(pos_correct["pos1_province"], pos_total["pos1_province"]),
            "pos2_alpha": safe_div(pos_correct["pos2_alpha"], pos_total["pos2_alpha"]),
            "pos3plus_alnum": safe_div(pos_correct["pos3plus_alnum"], pos_total["pos3plus_alnum"]),
        },
        "province_breakdown": {
            province: {
                "sample_count": row["sample_count"],
                "exact_plate_acc": safe_div(row["exact_plate_correct"], row["sample_count"]),
                "first_char_acc": safe_div(row["first_char_correct"], row["sample_count"]),
            }
            for province, row in sorted(province_rows.items())
        },
        "throughput_sec_per_sample": elapsed / len(dataset) if len(dataset) else 0.0,
        "bad_cases": bad_cases,
    }
    return report


def main():
    args = get_parser().parse_args()
    report = evaluate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
