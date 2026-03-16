#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.load_data import CHARS, PROVINCE_COUNT, load_label_items, read_ppm_p6_payload
from model.LPRNet import build_lprnet
from test_LPRNet import greedy_decode_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate raw board OCR dump anchors.')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--img_dirs', default='.')
    parser.add_argument('--txt_file', required=True)
    parser.add_argument('--first_char_time_steps', type=int, default=6)
    parser.add_argument('--out_json', default='')
    return parser.parse_args()


def resolve_anchor_paths(img_dirs, txt_file):
    rows = []
    for rel_path, filename, text in load_label_items(txt_file):
        found = None
        for img_dir in img_dirs:
            root = Path(img_dir)
            for cand in (root / rel_path, root / filename):
                if cand.exists():
                    found = cand
                    break
            if found is not None:
                break
        if found is not None:
            rows.append((found, text))
    return rows


def decode_logits(logits: np.ndarray) -> str:
    decoded = greedy_decode_logits(np.expand_dims(logits, axis=0))[0]
    return ''.join(CHARS[int(c)] for c in decoded)


def main() -> int:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    rows = resolve_anchor_paths(args.img_dirs.split(','), args.txt_file)
    if not rows:
        raise RuntimeError(f'no valid board anchors found from {args.txt_file}')

    net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(str(weights), map_location='cpu'))
    net.eval()

    exact = 0
    first_ok = 0
    blank_mean = 0.0
    details = []

    with torch.no_grad():
        for image_path, gt_text in rows:
            image = read_ppm_p6_payload(image_path)
            x = image.astype('float32')
            x -= 127.5
            x *= 0.0078125
            x = np.transpose(x, (2, 0, 1))[None, ...]
            logits = net(torch.from_numpy(x)).cpu().numpy()[0]
            pred = decode_logits(logits)
            blank_idx = len(CHARS) - 1
            blank_ratio = float(np.mean(np.argmax(logits, axis=0) == blank_idx))
            aux_steps = max(1, min(args.first_char_time_steps, logits.shape[1]))
            first_proxy = torch.softmax(torch.from_numpy(logits[:PROVINCE_COUNT, :aux_steps].mean(axis=1)), dim=0)
            topk = torch.topk(first_proxy, k=min(5, PROVINCE_COUNT))
            top5 = [(CHARS[int(idx)], float(val)) for val, idx in zip(topk.values.tolist(), topk.indices.tolist())]

            exact += int(pred == gt_text)
            first_ok += int(bool(gt_text) and bool(pred) and gt_text[0] == pred[0])
            blank_mean += blank_ratio
            details.append({
                'image_path': str(image_path),
                'gt': gt_text,
                'pred': pred,
                'blank_top1_ratio': blank_ratio,
                'first_char_top5': top5,
            })

    report = {
        'model': str(weights),
        'sample_count': len(rows),
        'exact_plate_acc': exact / len(rows),
        'first_char_acc': first_ok / len(rows),
        'blank_top1_mean': blank_mean / len(rows),
        'details': details,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
