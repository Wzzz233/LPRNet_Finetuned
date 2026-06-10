#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/home/wzzz/LPRNet')
SRC = ROOT / 'src'
for p in (SRC, SRC / 'training'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import load_data as _ld
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from load_data import UnifiedManifestDataset
from training.train_LPRNet import collate_fn as train_collate_fn

CHARS = list(_ld.CHARS)
BLANK_IDX = len(CHARS) - 1
CLASS_NUM = len(CHARS)


def decode(logits):
    if logits.ndim == 3:
        logits = logits[0]
    pred = logits.argmax(axis=0).tolist()
    out = []
    prev = None
    for idx in pred:
        if idx == BLANK_IDX or idx == prev:
            prev = idx
            continue
        out.append(CHARS[idx])
        prev = idx
    return ''.join(out)


def load_model(path, device):
    state = torch.load(path, map_location='cpu')
    net, cfg = build_lprnet_multihead_from_state_dict(state, lpr_max_len=8, phase=False, class_num=CLASS_NUM)
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device)
    net.eval()
    return net, cfg


def evaluate_manifest(manifest_path, model, device, dataset_root, batch_size=120, limit=0):
    dataset = UnifiedManifestDataset(
        manifest_path=str(manifest_path),
        img_size=(94, 24),
        lpr_max_len=8,
        dataset_root=str(dataset_root),
    )
    if limit and limit > 0:
        dataset.samples = dataset.samples[:limit]
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=train_collate_fn)
    exact = 0
    total = 0
    province_ok = 0
    rear_ok = 0
    rows = []
    with torch.no_grad():
        for images, labels_flat, lengths_list, families in loader:
            images = images.to(device)
            out = model(images)
            logits = out['green8'] if isinstance(out, dict) else out
            logits = logits.cpu().numpy()
            ptr = 0
            for i in range(logits.shape[0]):
                pred = decode(logits[i:i + 1])
                L = int(lengths_list[i])
                gt = ''.join(CHARS[int(labels_flat[ptr + j])] for j in range(L) if 0 <= int(labels_flat[ptr + j]) < CLASS_NUM)
                ptr += L
                total += 1
                exact += int(pred == gt)
                province_ok += int(bool(pred) and bool(gt) and pred[0] == gt[0])
                rear_ok += int(len(pred) >= 2 and len(gt) >= 2 and pred[1:] == gt[1:])
                if len(rows) < 80:
                    rows.append({'idx': total - 1, 'gt': gt, 'pred': pred, 'ok': pred == gt})
    return {
        'manifest': str(manifest_path),
        'total': total,
        'exact': exact,
        'exact_pct': round(100.0 * exact / total, 2) if total else 0.0,
        'province_ok': province_ok,
        'province_pct': round(100.0 * province_ok / total, 2) if total else 0.0,
        'rear_ok': rear_ok,
        'rear_pct': round(100.0 * rear_ok / total, 2) if total else 0.0,
        'sample_predictions': rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=[
        'experiments/a_ratio_r50_20260510/best_LPRNet_model.pth',
        'experiments/green_lowlight_aug_v1_lowlight10_20260610/best_LPRNet_model.pth',
        'experiments/green_lowlight_aug_v1_lowlight15_20260610/best_LPRNet_model.pth',
        'experiments/green_lowlight_aug_v1_lowlight25_20260610/best_LPRNet_model.pth',
    ])
    ap.add_argument('--manifests', nargs='+', default=[
        'manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv',
        'manifests_rebased/green_lowlight_aug_v1_20260610/lowlight_heldout_6000.csv',
        'manifests_rebased/green_lowlight_aug_v1_20260610/real_green_dark_39.csv',
    ])
    ap.add_argument('--dataset-root', default=str(ROOT))
    ap.add_argument('--out', default='experiments/green_lowlight_aug_v1_20260610_eval/eval_comparison.json')
    ap.add_argument('--batch-size', type=int, default=120)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--cuda', default='auto', choices=['auto', 'true', 'false'])
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() if args.cuda == 'auto' else args.cuda == 'true'
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    results = {}
    for model_path_raw in args.models:
        model_path = Path(model_path_raw)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.exists():
            print(f'[skip] missing model: {model_path}')
            continue
        print(f'[model] {model_path}')
        model, cfg = load_model(model_path, device)
        model_results = {'config': cfg, 'manifests': {}}
        for manifest_raw in args.manifests:
            manifest = Path(manifest_raw)
            if not manifest.is_absolute():
                manifest = ROOT / manifest
            if not manifest.exists():
                print(f'  [skip] missing manifest: {manifest}')
                continue
            res = evaluate_manifest(manifest, model, device, Path(args.dataset_root), args.batch_size, args.limit)
            model_results['manifests'][manifest.name] = res
            print(f"  {manifest.name}: {res['exact']}/{res['total']} exact={res['exact_pct']:.2f}% prov={res['province_pct']:.2f}% rear={res['rear_pct']:.2f}%")
        results[str(model_path)] = model_results

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[saved] {out_path}')


if __name__ == '__main__':
    main()
