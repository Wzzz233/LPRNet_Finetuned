#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
for p in [ROOT / 'src', ROOT / 'src' / 'evaluation', ROOT / 'src' / 'training', ROOT / 'src' / 'utils']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import CHARS, prepare_board_ocr_input_bgr888  # noqa: E402
from eval_lpr_detailed import decode_logits  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa: E402
from test_LPRNet import collate_fn  # noqa: E402
from train_LPRNet import forward_family_logits  # noqa: E402
from micro_rectifier.geometry import apply_parametric_warp_bgr  # noqa: E402
from micro_rectifier.model import MicroRectifier  # noqa: E402


class Cluster2RectifiedDataset(Dataset):
    def __init__(self, csv_path: Path, rectifier_ckpt: Path, grayscale: bool = False, input_size=(160, 48)):
        self.rows = []
        with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = Path(row.get('local_crop_path') or row.get('crop_path') or '')
                if img_path.exists() and row.get('gt_text'):
                    self.rows.append(row)
        self.input_w, self.input_h = input_size
        self.grayscale = grayscale
        in_channels = 1 if grayscale else 3
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(str(rectifier_ckpt), map_location=self.device)
        self.rectifier = MicroRectifier(in_channels=in_channels).to(self.device)
        self.rectifier.load_state_dict(ckpt['state_dict'])
        self.rectifier.eval()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        crop = cv2.imread(str(row['local_crop_path']), cv2.IMREAD_COLOR)
        crop = cv2.resize(crop, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        model_in = crop
        if self.grayscale:
            model_in = cv2.cvtColor(model_in, cv2.COLOR_BGR2GRAY)[..., None]
        x = model_in.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        with torch.no_grad():
            out = self.rectifier(torch.from_numpy(x).float().to(self.device))
            params = out['params'][0].detach().cpu().numpy().tolist()
        rectified_crop = apply_parametric_warp_bgr(crop, *params)
        ocrin, _occ = prepare_board_ocr_input_bgr888(rectified_crop, 94, 24, 'letterbox', 'nn', 'none', 'bgr')
        image = ocrin.astype(np.float32)
        image -= 127.5
        image *= 0.0078125
        image = np.transpose(image, (2, 0, 1))
        label = [CHARS.index(ch) for ch in row['gt_text']]
        sample = {
            'image': torch.from_numpy(image).float(),
            'label': torch.tensor(label, dtype=torch.int32),
            'length': len(label),
            'gt_text': row['gt_text'],
            'sample_id': row['sample_id'],
            'params': params,
        }
        return sample


def collate_cluster2(samples):
    images = torch.stack([s['image'] for s in samples], dim=0)
    labels = [s['label'].tolist() for s in samples]
    lengths = [s['length'] for s in samples]
    return images, labels, lengths, samples


def load_model(model_path: Path, device):
    state = torch.load(str(model_path), map_location=device)
    net, _ = build_lprnet_multihead_from_state_dict(state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device)
    net.eval()
    return net


def evaluate(model_path: Path, dataset: Cluster2RectifiedDataset, batch_size: int, num_workers: int):
    device = dataset.device
    net = load_model(model_path, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_cluster2)
    total = 0
    exact = 0
    first = 0
    details = []
    for images, _labels, _lengths, samples in loader:
        images = images.to(device)
        families = ['green8'] * images.shape[0]
        with torch.no_grad():
            logits = forward_family_logits(net, images, sample_families=families).detach().cpu().numpy()
            preds = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=families)
        for pred, meta in zip(preds, samples):
            gt = meta['gt_text']
            total += 1
            if pred == gt:
                exact += 1
            if pred[:1] == gt[:1]:
                first += 1
            details.append({'sample_id': meta['sample_id'], 'gt_text': gt, 'pred_text': pred, 'params': meta['params']})
    return {
        'model': str(model_path),
        'sample_count': total,
        'exact_acc': exact / total if total else 0.0,
        'first_char_acc': first / total if total else 0.0,
        'details': details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster2-csv', required=True)
    ap.add_argument('--rectifier-ckpt', required=True)
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--grayscale', action='store_true')
    args = ap.parse_args()

    dataset = Cluster2RectifiedDataset(Path(args.cluster2_csv), Path(args.rectifier_ckpt), grayscale=args.grayscale)
    results = []
    for m in args.models:
        results.append(evaluate(Path(m), dataset, args.batch_size, args.num_workers))
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'results': [{k: v for k, v in r.items() if k != 'details'} for r in results]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
