#!/usr/bin/env python3
"""Route A' Phase 3: Train large-crop province classifier.

Uses ResNet18 backbone (torchvision) for 31-way province classification.

Features:
- ResNet18 backbone with modified first conv for 1-channel gray3 input
- BatchNorm throughout
- 11.19M params (vs 0.06M in old tiny net)
- Best model selection based on province_stress macro_acc (not test_acc)

Usage:
  python src/training/train_province_largecrop_net.py \
    --train_dir datasets/routeA_prime_quadwarp_20260512/fullplate_224x72 \
    --val_dir datasets/routeA_prime_quadwarp_20260512/fullplate_224x72_test \
    --save_dir experiments/routeA_prime_quadwarp_20260512/B1_fullplate_color_224x72_raw \
    --input_size 224 72 \
    --epochs 30 --batch_size 64 --lr 0.001
"""
import argparse, json, time, sys, csv
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

ROOT = Path('/home/wzzz/LPRNet')
PROVINCE_CHARS = ['京','津','冀','晋','蒙','辽','吉','黑',
                  '沪','苏','浙','皖','闽','赣','鲁','豫',
                  '鄂','湘','粤','桂','琼','川','贵','云',
                  '藏','陕','甘','青','宁','新','渝']
PROVINCE_DICT = {c: i for i, c in enumerate(PROVINCE_CHARS)}
NUM_CLASSES = len(PROVINCE_CHARS)


class PlateImageDataset(Dataset):
    """Reads exported quad-warp images from a directory.

    Structure: dir/*.png where filenames contain the text.
    Also reads manifest.csv if present.
    """
    def __init__(self, img_dir, ocr_preproc='none', input_size=(224, 72)):
        self.img_dir = Path(img_dir)
        self.ocr_preproc = ocr_preproc
        self.input_size = input_size  # (width, height)
        self.samples = []

        # Try reading manifest.csv first
        manifest_path = self.img_dir / 'manifest.csv'
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    img_path = r.get('img_path', '').strip()
                    text = r.get('text', '').strip()
                    if img_path and text:
                        self.samples.append((img_path, text[0]))
        else:
            # Fall back to scanning PNG files
            for f in sorted(self.img_dir.glob('*.png')):
                # Parse text from filename: {idx}_{text}.png
                parts = f.stem.split('_', 1)
                if len(parts) >= 2:
                    text = parts[1]
                    if text and text[0] in PROVINCE_DICT:
                        self.samples.append((str(f), text[0]))

        print(f'  Loaded {len(self.samples)} samples from {img_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, province_char = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # Return blank on error
            img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype='uint8')

        # Apply preprocessing
        if self.ocr_preproc in ('gray', 'gray3'):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.ocr_preproc == 'gray3':
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                img = gray[:, :, None]
        elif self.ocr_preproc == 'bin':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        # Resize if needed
        h, w = img.shape[:2]
        if w != self.input_size[0] or h != self.input_size[1]:
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize
        img_t = torch.from_numpy(img.astype('float32') / 255.0)
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(-1)
        img_t = img_t.permute(2, 0, 1).contiguous()

        label = PROVINCE_DICT.get(province_char, 0)
        return img_t, torch.tensor(label, dtype=torch.long)


def build_model(in_channels=3, num_classes=NUM_CLASSES):
    """Build ResNet18 modified for province classification."""
    model = models.resnet18(weights=None)
    
    # Modify first conv for input channels
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final FC layer for num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def evaluate(model, loader, device, ocr_preproc):
    """Evaluate model on a dataset. Returns accuracy and per-province stats."""
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    per_class_hit = Counter()
    per_class_tot = Counter()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # For gray3 input: convert batch if needed (model gets 3-channel)
            if ocr_preproc in ('gray', 'gray3'):
                # Convert to grayscale single channel
                gray = images[:, 0:1, :, :] * 0.1140 + images[:, 1:2, :, :] * 0.5870 + images[:, 2:3, :, :] * 0.2990
                images = gray  # Model expects 1-channel for gray/gray3

            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total += labels.numel()
            correct += (preds == labels).sum().item()
            loss_sum += loss.item() * labels.numel()

            for t, p in zip(labels.tolist(), preds.tolist()):
                per_class_tot[t] += 1
                if t == p:
                    per_class_hit[t] += 1
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Macro
    macro = 0.0
    if per_class_tot:
        macro = sum(per_class_hit[k] / per_class_tot[k] for k in per_class_tot) / len(per_class_tot)

    # Per-province accuracy
    prov_accs = {}
    for k in sorted(per_class_tot.keys()):
        prov = PROVINCE_CHARS[k] if k < len(PROVINCE_CHARS) else '?'
        prov_accs[prov] = round(per_class_hit[k] / per_class_tot[k], 4)

    return {
        'loss': loss_sum / max(1, total),
        'acc': correct / max(1, total),
        'macro_acc': macro,
        'count': total,
        'per_province_acc': prov_accs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', required=True)
    ap.add_argument('--val_dirs', nargs='+', default=[])
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--input_size', type=int, nargs=2, default=[224, 72])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=20260512)
    ap.add_argument('--ocr_preproc', default='none', choices=['none', 'gray', 'gray3', 'bin'])
    ap.add_argument('--log_interval', type=int, default=100)
    ap.add_argument('--eval_interval', type=int, default=1)  # epochs between full evals
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    input_size = tuple(args.input_size)  # (W, H)
    in_channels = 1 if args.ocr_preproc in ('gray', 'gray3') else 3

    # Dataset
    print(f'Loading train data from: {args.train_dir}')
    ds_train = PlateImageDataset(args.train_dir, args.ocr_preproc, input_size)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    # Validation datasets
    val_loaders = {}
    val_names = []
    for vd in args.val_dirs:
        name = Path(vd).name
        ds_val = PlateImageDataset(vd, args.ocr_preproc, input_size)
        if len(ds_val) > 0:
            val_loaders[name] = DataLoader(ds_val, batch_size=args.batch_size,
                                           shuffle=False, num_workers=args.num_workers,
                                           pin_memory=True)
            val_names.append(name)
            print(f'  Val [{name}]: {len(ds_val)} samples')

    # Model
    model = build_model(in_channels=in_channels, num_classes=NUM_CLASSES)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    global_start = time.time()

    print(f'\nTraining: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}')
    print(f'  Device: {device}, Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    print(f'  Input size: {input_size}, preproc: {args.ocr_preproc}')
    print(f'  Train samples: {len(ds_train)}, Val sets: {val_names}')
    print(f'  Best selection: dump2 first_char_acc > province_stress macro > holdout non-major\n')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0

        for step_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            # Handle gray/gray3 input
            if args.ocr_preproc in ('gray', 'gray3'):
                gray = images[:, 0:1, :, :] * 0.1140 + images[:, 1:2, :, :] * 0.5870 + images[:, 2:3, :, :] * 0.2990
                images = gray

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            batch_count = labels.numel()
            train_total += batch_count
            train_correct += (preds == labels).sum().item()
            train_loss_sum += loss.item() * batch_count

            if step_idx == 1 or step_idx % args.log_interval == 0 or step_idx == len(train_loader):
                elapsed = time.time() - global_start
                eta = (elapsed / (step_idx + (epoch - 1) * len(train_loader))) * (args.epochs * len(train_loader) - (step_idx + (epoch - 1) * len(train_loader)))
                acc = train_correct / max(1, train_total)
                print(f'  E{epoch:02d}/{step_idx:04d} loss={loss.item():.3f} acc={acc:.3f} eta={eta/60:.0f}min')

        # Epoch metrics
        train_metrics = {
            'loss': train_loss_sum / max(1, train_total),
            'acc': train_correct / max(1, train_total),
            'epoch_sec': round(time.time() - epoch_start, 2),
        }

        # Evaluate on validation sets
        row = {'epoch': epoch, 'train': train_metrics}
        print(f'\n  Epoch {epoch} eval:')
        for name in val_names:
            loader = val_loaders[name]
            metrics = evaluate(model, loader, device, args.ocr_preproc)
            row[f'val_{name}'] = metrics
            print(f'    {name}: acc={metrics["acc"]:.4f} macro={metrics["macro_acc"]:.4f}')

        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        scheduler.step()

        # Best model selection: use province_stress macro as primary
        # (dump2 is not available at training time, so use best available)
        primary_score = 0.0
        # Prefer province_stress macro as primary
        for name in val_names:
            if 'stress' in name.lower() or 'province' in name.lower():
                m = row.get(f'val_{name}', {})
                primary_score = max(primary_score, m.get('macro_acc', 0))
        # Fallback to holdout macro
        for name in val_names:
            if 'holdout' in name.lower() or 'major' in name.lower():
                m = row.get(f'val_{name}', {})
                primary_score = max(primary_score, m.get('macro_acc', 0))

        if primary_score > best_score:
            best_score = primary_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_path = save_dir / 'best.pt'
            torch.save(best_state, best_path)
            print(f'  * New best: epoch {epoch} (score={primary_score:.4f}) -> {best_path}')

        # Save last
        torch.save(model.state_dict(), save_dir / 'last.pt')

    # Final summary
    total_elapsed = round(time.time() - global_start, 2)
    summary = {
        'device': str(device),
        'epochs': args.epochs,
        'input_size': list(input_size),
        'ocr_preproc': args.ocr_preproc,
        'train_samples': len(ds_train),
        'model': 'resnet18',
        'model_params_m': round(sum(p.numel() for p in model.parameters())/1e6, 2),
        'best_epoch': best_epoch,
        'best_score': best_score,
        'best_path': str(save_dir / 'best.pt'),
        'last_path': str(save_dir / 'last.pt'),
        'history': history,
        'total_elapsed_sec': total_elapsed,
    }
    (save_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\nDone. Best: epoch {best_epoch} (score={best_score:.4f})')
    print(f'Summary: {save_dir / "summary.json"}')


if __name__ == '__main__':
    main()
