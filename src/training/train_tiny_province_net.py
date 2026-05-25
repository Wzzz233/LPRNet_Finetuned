#!/usr/bin/env python3
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src' / 'training']:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from load_data import UnifiedManifestDataset, PROVINCE_COUNT
from training.train_LPRNet import collate_fn, configure_runtime


class TinyProvinceNet(nn.Module):
    def __init__(self, num_classes=PROVINCE_COUNT, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class FirstCharCropDataset(Dataset):
    def __init__(self, manifest_path, split_filter, ocr_preproc='none', resize_to=None):
        self.ocr_preproc = ocr_preproc
        self.resize_to = resize_to
        self.records = []
        self.targets = []
        manifest_path = Path(manifest_path)
        import csv
        with manifest_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split_filter and (row.get('split') or '').strip() != split_filter:
                    continue
                text = (row.get('text') or '').strip()
                img_path = (row.get('img_path') or '').strip()
                if not text or not img_path:
                    continue
                if not Path(img_path).exists():
                    continue
                self.records.append(row)
                self.targets.append(int(self.province_index(text[0])))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        img = cv2.imread(row['img_path'], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row['img_path'])
        
        # Quad-based plate extraction for full-frame images
        has_quad = row.get('has_quad', '0').strip() in ('1', 'True')
        if has_quad and self.resize_to and self.resize_to[0] > 94:
            try:
                pts = np.array([
                    [float(row.get('quad_1x', 0)), float(row.get('quad_1y', 0))],
                    [float(row.get('quad_2x', 0)), float(row.get('quad_2y', 0))],
                    [float(row.get('quad_3x', 0)), float(row.get('quad_3y', 0))],
                    [float(row.get('quad_4x', 0)), float(row.get('quad_4y', 0))],
                ], dtype='float32')
                if pts.min() >= 0 and pts.max() > 0:
                    x_min, y_min = int(pts[:,0].min()), int(pts[:,1].min())
                    x_max, y_max = int(pts[:,0].max()), int(pts[:,1].max())
                    if x_max > x_min and y_max > y_min:
                        # Only do perspective warp if the quad has valid area
                        src_pts_norm = pts - [float(x_min), float(y_min)]
                        dst_pts = np.array([[0,0],[self.resize_to[0]-1,0],[self.resize_to[0]-1,self.resize_to[1]-1],[0,self.resize_to[1]-1]], dtype='float32')
                        # Check if src_pts forms a valid convex quad (all 4 points distinct)
                        if src_pts_norm.shape == (4,2):
                            crop = img[y_min:y_max, x_min:x_max]
                            matrix = cv2.getPerspectiveTransform(src_pts_norm, dst_pts)
                            img = cv2.warpPerspective(crop, matrix, self.resize_to)
            except (KeyError, ValueError, cv2.error):
                pass
        
        if self.ocr_preproc in ('gray', 'gray3'):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.ocr_preproc == 'gray3':
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                img = gray[:, :, None]
        elif self.ocr_preproc == 'bin':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _thr, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        if self.resize_to:
            img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img.astype('float32') / 255.0)
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        img = img.permute(2, 0, 1).contiguous()
        label = self.targets[idx]
        labels = torch.tensor([label], dtype=torch.long)
        return img, labels, 1, row.get('family', '')

    @staticmethod
    def province_index(ch):
        chars = UnifiedManifestDataset.CHARS if hasattr(UnifiedManifestDataset, 'CHARS') else None
        if chars is not None:
            return chars.index(ch)
        from load_data import CHARS
        return CHARS.index(ch)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_manifest', required=True)
    ap.add_argument('--test_manifest', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20260420)
    ap.add_argument('--ocr_preproc', default='none', choices=['none', 'raw', 'gray', 'gray3', 'bin'])
    ap.add_argument('--input_mode', default='ocr_94x24', choices=['ocr_94x24', 'full_crop'])
    ap.add_argument('--full_crop_height', type=int, default=64)
    ap.add_argument('--log_interval', type=int, default=200)
    return ap


def make_datasets(args):
    if args.input_mode == 'full_crop':
        target_h = int(args.full_crop_height)
        target_w = int(round(target_h * 128 / 48))
        resize_to = (target_w, target_h)
        ds_train = FirstCharCropDataset(args.train_manifest, split_filter='train', ocr_preproc=args.ocr_preproc, resize_to=resize_to)
        ds_test = FirstCharCropDataset(args.test_manifest, split_filter='test', ocr_preproc=args.ocr_preproc, resize_to=resize_to)
    else:
        ds_train = UnifiedManifestDataset(
            args.train_manifest,
            [94, 24],
            8,
            split_filter='train',
            ocr_preproc=args.ocr_preproc,
        )
        ds_test = UnifiedManifestDataset(
            args.test_manifest,
            [94, 24],
            8,
            split_filter='test',
            ocr_preproc=args.ocr_preproc,
        )
    return ds_train, ds_test


def build_model(gray_input=False):
    in_channels = 1 if gray_input else 3
    return TinyProvinceNet(in_channels=in_channels)


def convert_batch_inputs(images, ocr_preproc):
    if ocr_preproc in ('gray', 'gray3'):
        gray = images[:, 0:1, :, :] * 0.1140 + images[:, 1:2, :, :] * 0.5870 + images[:, 2:3, :, :] * 0.2990
        return gray
    return images


def extract_first_targets(labels, lengths):
    out = []
    start = 0
    for length in lengths:
        if length <= 0:
            out.append(PROVINCE_COUNT - 1)
        else:
            out.append(int(labels[start]))
        start += length
    return torch.tensor(out, dtype=torch.long)


def pad_images_to_batch(images):
    max_h = max(int(img.shape[1]) for img in images)
    max_w = max(int(img.shape[2]) for img in images)
    batch = []
    for img in images:
        c, h, w = img.shape
        canvas = torch.zeros((c, max_h, max_w), dtype=img.dtype)
        y0 = (max_h - h) // 2
        x0 = (max_w - w) // 2
        canvas[:, y0:y0+h, x0:x0+w] = img
        batch.append(canvas)
    return torch.stack(batch, dim=0)


def fullcrop_collate_fn(batch):
    images, labels, lengths, families = zip(*batch)
    return pad_images_to_batch(images), torch.cat(labels, dim=0), list(lengths), list(families)


def _format_seconds(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f'{h:02d}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'


def evaluate(model, loader, device, ocr_preproc):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    per_class_hit = Counter()
    per_class_tot = Counter()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels, lengths, _families in loader:
            images = convert_batch_inputs(images.to(device), ocr_preproc)
            targets = extract_first_targets(labels, lengths).to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)
            total += targets.numel()
            correct += int((preds == targets).sum().item())
            loss_sum += float(loss.item()) * targets.numel()
            for t, p in zip(targets.tolist(), preds.tolist()):
                per_class_tot[t] += 1
                if t == p:
                    per_class_hit[t] += 1
    macro = 0.0
    if per_class_tot:
        macro = sum(per_class_hit[k] / per_class_tot[k] for k in per_class_tot) / len(per_class_tot)
    return {
        'loss': loss_sum / max(1, total),
        'acc': correct / max(1, total),
        'macro_acc': macro,
        'count': total,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    configure_runtime(args.seed, True)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds_train, ds_test = make_datasets(args)
    collate = fullcrop_collate_fn if args.input_mode == 'full_crop' else collate_fn
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    model = build_model(gray_input=args.ocr_preproc in ('gray', 'gray3')).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    best = None
    best_state = None
    global_start = time.time()
    total_train_steps = max(1, len(train_loader) * args.epochs)
    print(json.dumps({
        'event': 'train_start',
        'device': str(device),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'ocr_preproc': args.ocr_preproc,
        'input_mode': args.input_mode,
        'train_size': len(ds_train),
        'test_size': len(ds_test),
        'train_steps_per_epoch': len(train_loader),
        'total_train_steps': total_train_steps,
    }, ensure_ascii=False), flush=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0
        epoch_start = time.time()
        for step_idx, (images, labels, lengths, _families) in enumerate(train_loader, start=1):
            step_start = time.time()
            images = convert_batch_inputs(images.to(device), args.ocr_preproc)
            targets = extract_first_targets(labels, lengths).to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            preds = logits.argmax(dim=1)
            batch_count = targets.numel()
            train_total += batch_count
            train_correct += int((preds == targets).sum().item())
            train_loss_sum += float(loss.item()) * batch_count
            global_step += 1

            if step_idx == 1 or step_idx % args.log_interval == 0 or step_idx == len(train_loader):
                elapsed = time.time() - global_start
                step_time = time.time() - step_start
                avg_step = elapsed / max(1, global_step)
                remaining_steps = max(0, total_train_steps - global_step)
                eta_seconds = avg_step * remaining_steps
                heartbeat = {
                    'event': 'heartbeat',
                    'epoch': epoch,
                    'step': step_idx,
                    'steps_in_epoch': len(train_loader),
                    'global_step': global_step,
                    'total_steps': total_train_steps,
                    'batch_loss': float(loss.item()),
                    'train_running_acc': train_correct / max(1, train_total),
                    'elapsed_sec': round(elapsed, 2),
                    'step_sec': round(step_time, 3),
                    'avg_step_sec': round(avg_step, 3),
                    'eta_sec': round(eta_seconds, 2),
                    'eta_hms': _format_seconds(eta_seconds),
                }
                print(json.dumps(heartbeat, ensure_ascii=False), flush=True)

        train_metrics = {
            'loss': train_loss_sum / max(1, train_total),
            'acc': train_correct / max(1, train_total),
            'count': train_total,
            'epoch_sec': round(time.time() - epoch_start, 2),
        }
        test_metrics = evaluate(model, test_loader, device, args.ocr_preproc)
        row = {'epoch': epoch, 'train': train_metrics, 'test': test_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if best is None or (test_metrics['acc'], test_metrics['macro_acc']) > (best['test']['acc'], best['test']['macro_acc']):
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_dir / 'best.pt')
            print(json.dumps({
                'event': 'new_best',
                'epoch': epoch,
                'best_test_acc': test_metrics['acc'],
                'best_test_macro_acc': test_metrics['macro_acc'],
                'best_path': str(save_dir / 'best.pt'),
            }, ensure_ascii=False), flush=True)

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(last_state, save_dir / 'last.pt')
    summary = {
        'device': str(device),
        'epochs': args.epochs,
        'ocr_preproc': args.ocr_preproc,
        'input_mode': args.input_mode,
        'train_size': len(ds_train),
        'test_size': len(ds_test),
        'best': best,
        'history': history,
        'best_path': str(save_dir / 'best.pt'),
        'last_path': str(save_dir / 'last.pt'),
        'log_interval': args.log_interval,
        'total_elapsed_sec': round(time.time() - global_start, 2),
    }
    (save_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'train_done',
        'summary_path': str(save_dir / 'summary.json'),
        'best_path': str(save_dir / 'best.pt'),
        'last_path': str(save_dir / 'last.pt'),
        'total_elapsed_sec': summary['total_elapsed_sec'],
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
