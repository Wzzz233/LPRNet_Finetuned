#!/usr/bin/env python3
"""Route A Epoch Curve Training: unified best.pt + intermediate CKPTs.

Key improvements over previous scripts:
  1. best.pt uses province_stress macro (not train acc, not epoch 1 artifact)
  2. Saves checkpoints at epochs 1,2,3,5,10,15,20,25,30
  3. Refuses to run without --val_dirs (no silent best.pt bugs)
  4. Full evaluate() on val set each epoch (negligible cost: 1240 images)
"""
import argparse, json, time, sys, csv, random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models

ROOT = Path('/home/wzzz/LPRNet')
PROVINCE_CHARS = ['京','津','冀','晋','蒙','辽','吉','黑',
                  '沪','苏','浙','皖','闽','赣','鲁','豫',
                  '鄂','湘','粤','桂','琼','川','贵','云',
                  '藏','陕','甘','青','宁','新','渝']
PROVINCE_DICT = {c: i for i, c in enumerate(PROVINCE_CHARS)}
NUM_CLASSES = len(PROVINCE_CHARS)
SAVE_EPOCHS = {1, 2, 3, 5, 10, 15, 20, 25, 30}


class EpochCurveDataset(Dataset):
    """Reads exported quad-warp images."""
    def __init__(self, img_dir, ocr_preproc='none', input_size=(224, 72),
                 boardlike_aug=False, real_upweight=0):
        self.img_dir = Path(img_dir)
        self.ocr_preproc = ocr_preproc
        self.input_size = input_size
        self.boardlike_aug = boardlike_aug
        self.samples = []
        self.is_real = []

        manifest_path = self.img_dir / 'manifest.csv'
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    img_path = r.get('img_path', '').strip()
                    text = r.get('text', '').strip()
                    src = r.get('source', '').strip()
                    if img_path and text:
                        self.samples.append((img_path, text[0]))
                        self.is_real.append('ccpd2020' in src.lower())
        else:
            for f in sorted(self.img_dir.glob('*.png')):
                parts = f.stem.split('_', 1)
                if len(parts) >= 2 and parts[1] and parts[1][0] in PROVINCE_DICT:
                    self.samples.append((str(f), parts[1][0]))
                    self.is_real.append(False)
        print(f'  Loaded {len(self.samples)} samples from {img_dir}')

    def __len__(self): return len(self.samples)

    def _boardlike_augment(self, img):
        h, w = img.shape[:2]
        if random.random() < 0.3:
            dx = int(random.uniform(-4, 4) * w / 224)
            dy = int(random.uniform(-4, 4) * h / 72)
            x1, y1 = max(0, dx), max(0, dy)
            x2, y2 = min(w, w + dx), min(h, h + dy)
            if x2 > x1 and y2 > y1:
                img = cv2.resize(img[y1:y2, x1:x2], (w, h))
        if random.random() < 0.25:
            img = cv2.GaussianBlur(img, (random.choice([3,5]),)*2, 0)
        if random.random() < 0.3:
            img = cv2.convertScaleAbs(img, alpha=random.uniform(0.7,1.3), beta=random.uniform(-20,20))
        if random.random() < 0.2:
            _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(60,95)])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if random.random() < 0.15:
            img = cv2.resize(img, (int(w*random.uniform(0.92,1.08)), h))
            img = cv2.resize(img, (w, h))
        return img

    def __getitem__(self, idx):
        img_path, province_char = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype='uint8')
        if self.boardlike_aug:
            img = self._boardlike_augment(img)
        if self.ocr_preproc in ('gray', 'gray3'):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.ocr_preproc == 'gray3':
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                img = gray[:, :, None]
        h, w = img.shape[:2]
        if w != self.input_size[0] or h != self.input_size[1]:
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img.astype('float32')/255.0)
        if img_t.ndim == 2: img_t = img_t.unsqueeze(-1)
        img_t = img_t.permute(2, 0, 1).contiguous()
        return img_t, torch.tensor(PROVINCE_DICT.get(province_char, 0), dtype=torch.long)


def build_model(in_channels=1, num_classes=NUM_CLASSES):
    model = models.resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device, ocr_preproc):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    per_hit, per_tot = Counter(), Counter()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if ocr_preproc in ('gray','gray3'):
                images = images[:,0:1,:,:]*0.1140 + images[:,1:2,:,:]*0.5870 + images[:,2:3,:,:]*0.2990
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(1)
            total += labels.numel()
            correct += (preds==labels).sum().item()
            loss_sum += loss.item()*labels.numel()
            for t,p in zip(labels.tolist(),preds.tolist()):
                per_tot[t] += 1
                if t==p: per_hit[t] += 1
    macro = sum(per_hit[k]/per_tot[k] for k in per_tot)/len(per_tot) if per_tot else 0.0
    prov_accs = {PROVINCE_CHARS[k]: round(per_hit[k]/per_tot[k],4) for k in sorted(per_tot.keys())}
    return {'loss': loss_sum/max(1,total), 'acc': correct/max(1,total),
            'macro_acc': macro, 'count': total, 'per_province_acc': prov_accs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', required=True)
    ap.add_argument('--val_dir', help='Validation dir with manifest.csv (province_stress). REQUIRED for proper best.pt.')
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--input_size', type=int, nargs=2, default=[224,72])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=20260512)
    ap.add_argument('--ocr_preproc', default='gray3', choices=['none','gray','gray3','bin'])
    ap.add_argument('--log_interval', type=int, default=200)
    ap.add_argument('--boardlike_aug', action='store_true')
    ap.add_argument('--real_upweight', type=float, default=0)
    args = ap.parse_args()

    # ── Validation required ──
    if not args.val_dir:
        print('FATAL: --val_dir is REQUIRED. Without val set, best.pt selection is broken.')
        print('  Provide a province_stress image directory with manifest.csv.')
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    input_size = tuple(args.input_size)
    in_channels = 1 if args.ocr_preproc in ('gray','gray3') else 3

    ds_train = EpochCurveDataset(args.train_dir, args.ocr_preproc, input_size,
                                  boardlike_aug=args.boardlike_aug, real_upweight=args.real_upweight)
    ds_val = EpochCurveDataset(args.val_dir, args.ocr_preproc, input_size)

    if args.real_upweight > 0:
        weights = [args.real_upweight if r else 1.0 for r in ds_train.is_real]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
        print(f'  Weighted sampler: real_upweight={args.real_upweight}x')
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(ds_train, args.batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(in_channels=in_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_val_macro = -1.0
    best_epoch = 0
    best_path = save_dir / 'best.pt'
    ckpt_dir = save_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    global_start = time.time()

    config = {
        'device': str(device), 'epochs': args.epochs, 'input_size': list(input_size),
        'ocr_preproc': args.ocr_preproc, 'train_samples': len(ds_train),
        'val_samples': len(ds_val), 'model': 'resnet18',
        'boardlike_aug': args.boardlike_aug, 'real_upweight': args.real_upweight,
        'batch_size': args.batch_size, 'lr': args.lr, 'seed': args.seed,
        'best_metric': 'province_stress_macro',
    }
    print(json.dumps({'event':'train_start', **config}))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss_sum = train_total = train_correct = 0

        for step_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            if args.ocr_preproc in ('gray','gray3'):
                images = images[:,0:1,:,:]*0.1140 + images[:,1:2,:,:]*0.5870 + images[:,2:3,:,:]*0.2990
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(1)
            train_total += labels.numel()
            train_correct += (preds==labels).sum().item()
            train_loss_sum += loss.item()*labels.numel()
            if step_idx==1 or step_idx%args.log_interval==0 or step_idx==len(train_loader):
                elapsed = time.time()-global_start
                eta = elapsed/(step_idx+(epoch-1)*len(train_loader))*(args.epochs*len(train_loader)-step_idx-(epoch-1)*len(train_loader))
                print(f'  E{epoch:02d}/{step_idx:04d} loss={loss.item():.3f} acc={train_correct/max(1,train_total):.3f} eta={eta/60:.0f}min')

        train_metrics = {'loss': train_loss_sum/max(1,train_total), 'acc': train_correct/max(1,train_total), 'epoch_sec': round(time.time()-epoch_start,2)}

        # ── Validate on province_stress ──
        val_metrics = evaluate(model, val_loader, device, args.ocr_preproc)
        val_macro = val_metrics['macro_acc']

        print(json.dumps({'epoch':epoch, 'train':train_metrics, 'val':val_metrics}))

        scheduler.step()

        # ── Best.pt: use province_stress macro ──
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f'  * NEW BEST: epoch {epoch} (val_macro={val_macro:.4f})')

        # ── Save intermediate checkpoints ──
        torch.save(model.state_dict(), save_dir / 'last.pt')
        if epoch in SAVE_EPOCHS:
            ckpt_path = ckpt_dir / f'epoch_{epoch:03d}.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f'  * Saved checkpoint: {ckpt_path}')

    total_elapsed = round(time.time()-global_start, 2)
    summary = {**config, 'best_val_macro': best_val_macro, 'best_epoch': best_epoch,
               'best_path': str(best_path), 'last_path': str(save_dir/'last.pt'),
               'ckpt_dir': str(ckpt_dir), 'save_epochs': sorted(SAVE_EPOCHS),
               'total_elapsed_sec': total_elapsed}
    (save_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'\nDone. Best: epoch {best_epoch} (val_macro={best_val_macro:.4f})')


if __name__ == '__main__':
    main()
