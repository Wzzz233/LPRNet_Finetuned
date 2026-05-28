#!/usr/bin/env python3
"""Police Province Sidecar training script.
ResNet18 classifier: 224x72 fullplate input -> 31 province classes.
Based on green Route A' successful recipe.
"""
import argparse, csv, json, os, sys, time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

ROOT = Path('/home/wzzz/LPRNet')


class ProvinceSidecarDataset(Dataset):
    def __init__(self, manifest_path, preproc='gray3', resize_to=(224, 72),
                 province_map=None):
        self.preproc = preproc
        self.resize_to = resize_to
        self.records = []
        self.labels = []

        with open(manifest_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row.get('img_path', '').strip()
                if not img_path:
                    continue
                full = img_path if os.path.isabs(img_path) else str(ROOT / img_path)
                if not Path(full).exists():
                    continue
                lbl = int(row.get('label', -1))
                if lbl < 0:
                    continue
                self.records.append({'img_path': full, 'label': lbl,
                                     'text': row.get('text', ''),
                                     'province': row.get('province', ''),
                                     'source': row.get('source', '')})
                self.labels.append(lbl)

        # Province name mapping (optional)
        self.province_map = province_map or {}
        self.idx2prov = {v: k for k, v in self.province_map.items()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec['img_path'])
        if img is None:
            raise FileNotFoundError(rec['img_path'])

        # Resize
        img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_LINEAR)

        # Preproc
        if self.preproc in ('gray', 'gray3'):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.preproc == 'gray3':
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                img = gray[:, :, None]

        # To tensor [0,1] range
        img = torch.from_numpy(img.astype('float32') / 255.0)
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        img = img.permute(2, 0, 1).contiguous()  # CHW

        return img, rec['label'], rec


def collate_sidecar(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    records = [b[2] for b in batch]
    return images, labels, records


def build_model(num_classes=31, pretrained=True):
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    # First conv: adapt to 1-channel gray input (we handle this via input channels)
    # We keep 3-channel input via gray3 (3 identical channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device, criterion):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    all_preds = []
    all_labels = []
    all_records = []

    with torch.no_grad():
        for images, labels, records in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total += labels.numel()
            correct += int((preds == labels).sum().item())
            loss_sum += float(loss.item()) * labels.numel()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_records.extend(records)

    return {
        'loss': loss_sum / max(1, total),
        'acc': correct / max(1, total),
        'correct': correct,
        'total': total,
        'preds': all_preds,
        'labels': all_labels,
        'records': all_records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_manifest', required=True)
    ap.add_argument('--val_manifest', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=20260528)
    ap.add_argument('--pretrained', type=lambda x: x.lower() in ('true','1','yes'), default=True)
    ap.add_argument('--preproc', default='gray3', choices=['gray3','color','gray'])
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--num_classes', type=int, default=31)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load province mapping
    province_map = {}
    with open('keys/police_keys.txt') as f:
        chars = [l.strip() for l in f if l.strip()]
    for i, c in enumerate(chars[:31]):
        province_map[c] = i

    # Datasets
    ds_train = ProvinceSidecarDataset(
        args.train_manifest, preproc=args.preproc,
        resize_to=(224, 72), province_map=province_map)
    ds_val = ProvinceSidecarDataset(
        args.val_manifest, preproc=args.preproc,
        resize_to=(224, 72), province_map=province_map)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_sidecar)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_sidecar)

    # Model
    in_channels = 1 if args.preproc == 'gray' else 3
    model = build_model(num_classes=args.num_classes, pretrained=args.pretrained)
    # If 1-channel, replace first conv
    if in_channels == 1:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy averaged weights from old conv
        with torch.no_grad():
            model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(json.dumps({
        'event': 'train_start',
        'device': str(device),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'pretrained': args.pretrained,
        'preproc': args.preproc,
        'train_size': len(ds_train),
        'val_size': len(ds_val),
        'num_classes': args.num_classes,
    }, ensure_ascii=False), flush=True)

    history = []
    best_val_acc = -1.0
    best_state = None
    global_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0
        epoch_start = time.time()

        for images, labels, _records in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            preds = logits.argmax(dim=1)
            n = labels.numel()
            train_total += n
            train_correct += int((preds == labels).sum().item())
            train_loss_sum += float(loss.item()) * n

        train_acc = train_correct / max(1, train_total)
        train_loss = train_loss_sum / max(1, train_total)

        val_metrics = evaluate(model, val_loader, device, criterion)
        val_acc = val_metrics['acc']
        val_loss = val_metrics['loss']

        epoch_sec = time.time() - epoch_start
        row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'epoch_sec': round(epoch_sec, 2),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_dir / 'best.pt')
            print(json.dumps({'event': 'new_best', 'epoch': epoch, 'val_acc': round(val_acc, 4)},
                             ensure_ascii=False), flush=True)

    # Save last
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(last_state, save_dir / 'last.pt')

    # Final val predictions CSV
    model.load_state_dict(best_state)
    model.eval()
    final_metrics = evaluate(model, val_loader, device, criterion)

    # Write predictions CSV
    with open(save_dir / 'predictions_val.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'img_path', 'gt_province', 'gt_label', 'pred_label',
                     'pred_province', 'correct'])
        idx2prov = {v: k for k, v in province_map.items()}
        for i in range(len(final_metrics['preds'])):
            gt = final_metrics['labels'][i]
            pred = final_metrics['preds'][i]
            rec = final_metrics['records'][i]
            w.writerow([i, rec['img_path'], rec['province'], gt, pred,
                        idx2prov.get(pred, '?'), int(gt == pred)])

    # Confusion matrix CSV
    confusion = Counter()
    for gt, pred in zip(final_metrics['labels'], final_metrics['preds']):
        if gt != pred:
            confusion[(idx2prov.get(gt, '?'), idx2prov.get(pred, '?'))] += 1
    with open(save_dir / 'province_confusion.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gt', 'pred', 'count'])
        for (gt, pred), cnt in confusion.most_common():
            w.writerow([gt, pred, cnt])

    # Summary
    summary = {
        'args': vars(args),
        'best_val_acc': round(best_val_acc, 4),
        'best_epoch': max(range(len(history)), key=lambda i: history[i]['val_acc']) + 1,
        'history': history,
        'final_val': {
            'acc': round(final_metrics['acc'], 4),
            'correct': final_metrics['correct'],
            'total': final_metrics['total'],
        },
        'total_elapsed_sec': round(time.time() - global_start, 2),
        'best_path': str(save_dir / 'best.pt'),
        'last_path': str(save_dir / 'last.pt'),
    }
    with open(save_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        'event': 'train_done',
        'best_val_acc': round(best_val_acc, 4),
        'best_epoch': summary['best_epoch'],
        'total_elapsed_sec': summary['total_elapsed_sec'],
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
