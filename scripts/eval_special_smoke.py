#!/usr/bin/env python3
"""Per-family evaluation for special plate smoke training."""
import argparse, sys, os
from collections import defaultdict
import torch
import numpy as np

sys.path.insert(0, '/home/wzzz/LPRNet')
sys.path.insert(0, '/home/wzzz/LPRNet/src')

from LPRNet import LPRNet
import load_data as _ld_mod
from load_data import UnifiedManifestDataset


def collate_fn(batch):
    """Collate for UnifiedManifestDataset: (image, label, length, family)."""
    images = torch.stack([torch.from_numpy(item[0]) for item in batch])
    labels = [item[1] for item in batch]
    lengths = [item[2] for item in batch]
    families = [item[3] for item in batch]
    return images, labels, lengths, families


def greedy_decode(prebs, chars):
    """CTC greedy decode. Input: [B, C, T] (model raw output)."""
    prebs = prebs.cpu().detach().numpy()
    results = []
    for i in range(prebs.shape[0]):
        preb = prebs[i]  # [C, T]
        preb_labels = np.argmax(preb, axis=0)  # argmax over C → [T]
        merged = []
        prev = -1
        for idx in preb_labels:
            if idx != prev and idx != (len(chars) - 1):
                merged.append(chars[idx])
            prev = idx
        results.append(''.join(merged))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--keys-file', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--dataset-root', default='/home/wzzz/LPRNet')
    ap.add_argument('--split-filter', default='test', help='split_filter for dataset (default: test; set to val for cvreplace manifests)')
    ap.add_argument('--img-size', type=int, nargs=2, default=[94, 24])
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    # Load keys
    chars = []
    with open(args.keys_file, 'r', encoding='utf-8') as f:
        for line in f:
            c = line.strip()
            if c:
                chars.append(c)
    chars.append('-')
    class_num = len(chars)
    # Override global CHARS_DICT so loader can encode special characters
    _ld_mod.CHARS.clear()
    _ld_mod.CHARS.extend(chars)
    _ld_mod.CHARS_DICT.clear()
    _ld_mod.CHARS_DICT.update({c: i for i, c in enumerate(chars)})
    print(f'[Keys] {class_num-1} real keys, class_num={class_num}')

    # Load model
    device = torch.device(args.device)
    model = LPRNet(lpr_max_len=8, phase='test', class_num=class_num, dropout_rate=0.0)
    model.to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    print(f'[Model] Loaded {args.model}')

    # Load dataset
    ds = UnifiedManifestDataset(
        manifest_path=args.manifest,
        img_size=tuple(args.img_size),
        lpr_max_len=8,
        split_filter=args.split_filter,
        dataset_root=args.dataset_root,
        ocr_channel_order='bgr',
        ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_min_occ_ratio=0.0,
        ocr_quad_pad_ratio=0.0,
        gray3_prob=0.0,
    )
    print(f'[Data] {len(ds)} samples')

    # Evaluate
    family_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'samples': []})
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    with torch.no_grad():
        for images, labels, lengths, families in loader:
            images = images.to(device)
            prebs = model(images)
            preds = greedy_decode(prebs, chars)

            for i, family in enumerate(families):
                gt_ids = labels[i]
                gt = ''.join(chars[idx] for idx in gt_ids if idx < len(chars))
                pred = preds[i]
                correct = (pred == gt)
                family_stats[family]['total'] += 1
                if correct:
                    family_stats[family]['correct'] += 1
                if len(family_stats[family]['samples']) < 5:
                    family_stats[family]['samples'].append(
                        f'GT={gt} PRED={pred} {"OK" if correct else "ERR"}')

    # Report
    print(f'\n{"="*60}')
    print(f'Per-family Evaluation Results')
    print(f'{"="*60}')
    total_correct, total_all = 0, 0
    for family in sorted(family_stats.keys()):
        s = family_stats[family]
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        total_correct += s['correct']
        total_all += s['total']
        print(f'\n--- {family} ---')
        print(f'  Accuracy: {s["correct"]}/{s["total"]} = {acc:.2f}%')
        for t in s['samples']:
            print(f'  {t}')
    overall = total_correct / total_all * 100 if total_all > 0 else 0
    print(f'\n{"="*60}')
    print(f'Overall: {total_correct}/{total_all} = {overall:.2f}%')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
