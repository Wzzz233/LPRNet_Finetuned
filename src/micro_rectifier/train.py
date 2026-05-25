#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from micro_rectifier.dataset import MicroRectifierDataset, build_record_from_crop_pair
from micro_rectifier.model import MicroRectifier


def load_csv_records(path: Path):
    rows = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(build_record_from_crop_pair(
                sample_id=row['sample_id'],
                input_path=row['input_path'],
                target_path=row['target_path'],
                text=row.get('text', ''),
                split=row['split'],
                source_name=row['source_name'],
                dx=float(row['dx']),
                dy=float(row['dy']),
                sx=float(row['sx']),
                sy=float(row['sy']),
                shx=float(row['shx']),
            ))
    return rows


def collate(samples):
    batch = {}
    for key in ['image', 'target_image', 'params']:
        batch[key] = torch.stack([s[key] for s in samples], dim=0)
    for key in ['sample_id', 'text', 'source_name']:
        batch[key] = [s[key] for s in samples]
    return batch


def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    total = {'loss': 0.0, 'param_loss': 0.0, 'img_loss': 0.0}
    count = 0
    for batch in loader:
        image = batch['image'].to(device)
        target = batch['target_image'].to(device)
        params = batch['params'].to(device)
        with torch.set_grad_enabled(train):
            out = model(image)
            param_loss = F.smooth_l1_loss(out['params'], params)
            img_loss = F.l1_loss(out['rectified'], target)
            loss = param_loss + 2.0 * img_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bs = image.shape[0]
        total['loss'] += float(loss.detach().cpu()) * bs
        total['param_loss'] += float(param_loss.detach().cpu()) * bs
        total['img_loss'] += float(img_loss.detach().cpu()) * bs
        count += bs
    return {k: (v / count if count else 0.0) for k, v in total.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--val-csv', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--grayscale', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = MicroRectifierDataset(load_csv_records(Path(args.train_csv)), input_size=(160, 48), grayscale=args.grayscale)
    val_ds = MicroRectifierDataset(load_csv_records(Path(args.val_csv)), input_size=(160, 48), grayscale=args.grayscale)
    in_channels = 1 if args.grayscale else 3
    model = MicroRectifier(in_channels=in_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    history = []
    best = float('inf')
    for epoch in range(args.epochs):
        train_stats = run_epoch(model, train_loader, optimizer, device, True)
        val_stats = run_epoch(model, val_loader, optimizer, device, False)
        row = {'epoch': epoch, 'train': train_stats, 'val': val_stats}
        history.append(row)
        torch.save({'state_dict': model.state_dict(), 'history': history, 'args': vars(args)}, out_dir / 'last.pt')
        if val_stats['loss'] < best:
            best = val_stats['loss']
            torch.save({'state_dict': model.state_dict(), 'history': history, 'args': vars(args)}, out_dir / 'best.pt')
        print(json.dumps(row, ensure_ascii=False))
    (out_dir / 'train_summary.json').write_text(json.dumps({'best_val_loss': best, 'history': history}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
