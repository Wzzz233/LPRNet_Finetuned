#!/usr/bin/env python3
"""Route A' Next Stage: Modified training script with boardlike aug and real upweight.
Extends train_province_largecrop_net.py with:
  --boardlike_aug : add board-representative perturbations
  --real_upweight N : oversample real data by Nx within each batch
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
import torchvision.transforms as T

ROOT = Path('/home/wzzz/LPRNet')
PROVINCE_CHARS = ['京','津','冀','晋','蒙','辽','吉','黑',
                  '沪','苏','浙','皖','闽','赣','鲁','豫',
                  '鄂','湘','粤','桂','琼','川','贵','云',
                  '藏','陕','甘','青','宁','新','渝']
PROVINCE_DICT = {c: i for i, c in enumerate(PROVINCE_CHARS)}
NUM_CLASSES = len(PROVINCE_CHARS)


class AugmentedPlateDataset(Dataset):
    """Reads exported quad-warp images with optional boardlike augmentations."""
    def __init__(self, img_dir, ocr_preproc='none', input_size=(224, 72),
                 boardlike_aug=False, real_upweight=0):
        self.img_dir = Path(img_dir)
        self.ocr_preproc = ocr_preproc
        self.input_size = input_size  # (width, height)
        self.boardlike_aug = boardlike_aug
        self.real_upweight = real_upweight
        self.samples = []
        self.is_real = []

        manifest_path = self.img_dir / 'manifest.csv'
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    img_path = r.get('img_path', '').strip()
                    text = r.get('text', '').strip()
                    src = r.get('source', '').strip()
                    if img_path and text:
                        self.samples.append((img_path, text[0]))
                        self.is_real.append('ccpd2020' in src.lower())
        else:
            for f in sorted(self.img_dir.glob('*.png')):
                parts = f.stem.split('_', 1)
                if len(parts) >= 2:
                    text = parts[1]
                    if text and text[0] in PROVINCE_DICT:
                        self.samples.append((str(f), text[0]))
                        self.is_real.append(False)

        print(f'  Loaded {len(self.samples)} samples from {img_dir}')
        print(f'    Real: {sum(self.is_real)} ({sum(self.is_real)/len(self.samples)*100:.1f}%)')

    def __len__(self):
        return len(self.samples)

    def _boardlike_augment(self, img):
        """Apply board-representative perturbations.
        - Quad shift simulation (crop translate/scale)
        - Mild blur
        - Brightness/contrast
        - JPEG compression simulation
        - Does NOT target any specific dump.
        """
        h, w = img.shape[:2]

        # 1. Random slight crop/scale shift (simulates quad perturbation)
        if random.random() < 0.3:
            dx = int(random.uniform(-4, 4) * w / 224)
            dy = int(random.uniform(-4, 4) * h / 72)
            x1 = max(0, dx)
            y1 = max(0, dy)
            x2 = min(w, w + dx)
            y2 = min(h, h + dy)
            if x2 > x1 and y2 > y1:
                img = img[y1:y2, x1:x2]
                img = cv2.resize(img, (w, h))

        # 2. Mild Gaussian blur (simulates focus/defocus)
        if random.random() < 0.25:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 3. Brightness/contrast jitter
        if random.random() < 0.3:
            alpha = random.uniform(0.7, 1.3)  # contrast
            beta = random.uniform(-20, 20)    # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 4. Mild JPEG compression simulation (re-encode quality)
        if random.random() < 0.2:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 95)]
            _, enc = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # 5. Slight scaling distortion (aspect ratio tweak)
        if random.random() < 0.15:
            sx = random.uniform(0.92, 1.08)
            new_w = int(w * sx)
            img = cv2.resize(img, (new_w, h))
            img = cv2.resize(img, (w, h))

        return img

    def __getitem__(self, idx):
        img_path, province_char = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype='uint8')

        # Boardlike aug
        if self.boardlike_aug:
            img = self._boardlike_augment(img)

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


def build_model(in_channels=1, num_classes=NUM_CLASSES):
    model = models.resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device, ocr_preproc):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    per_class_hit = Counter()
    per_class_tot = Counter()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if ocr_preproc in ('gray', 'gray3'):
                gray = images[:, 0:1, :, :] * 0.1140 + images[:, 1:2, :, :] * 0.5870 + images[:, 2:3, :, :] * 0.2990
                images = gray
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

    macro = 0.0
    if per_class_tot:
        macro = sum(per_class_hit[k] / per_class_tot[k] for k in per_class_tot) / len(per_class_tot)
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
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--input_size', type=int, nargs=2, default=[224, 72])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=20260512)
    ap.add_argument('--ocr_preproc', default='gray3', choices=['none', 'gray', 'gray3', 'bin'])
    ap.add_argument('--log_interval', type=int, default=200)
    ap.add_argument('--boardlike_aug', action='store_true', help='Add board-representative augmentations')
    ap.add_argument('--real_upweight', type=float, default=0, help='Real data oversampling multiplier')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    input_size = tuple(args.input_size)
    in_channels = 1 if args.ocr_preproc in ('gray', 'gray3') else 3

    # Dataset
    ds_train = AugmentedPlateDataset(
        args.train_dir, args.ocr_preproc, input_size,
        boardlike_aug=args.boardlike_aug, real_upweight=args.real_upweight
    )

    # Weighted sampler for real upweight
    if args.real_upweight > 0:
        # Higher weight for real samples
        weights = [args.real_upweight if is_r else 1.0 for is_r in ds_train.is_real]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
        print(f'  Using weighted sampler: real_upweight={args.real_upweight}x')
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True
    )

    # Model
    model = build_model(in_channels=in_channels, num_classes=NUM_CLASSES)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_score = -1.0
    best_state = None
    global_start = time.time()

    config = {
        'device': str(device), 'epochs': args.epochs, 'input_size': list(input_size),
        'ocr_preproc': args.ocr_preproc, 'train_samples': len(ds_train),
        'model': 'resnet18', 'boardlike_aug': args.boardlike_aug,
        'real_upweight': args.real_upweight, 'batch_size': args.batch_size,
        'lr': args.lr, 'seed': args.seed,
    }
    print(json.dumps({'event': 'train_start', **config}, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0

        for step_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            if args.ocr_preproc in ('gray', 'gray3'):
                gray = images[:, 0:1, :, :] * 0.1140 + images[:, 1:2, :, :] * 0.5870 + images[:, 2:3, :, :] * 0.2990
                images = gray

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_total += labels.numel()
            train_correct += (preds == labels).sum().item()
            train_loss_sum += loss.item() * labels.numel()

            if step_idx == 1 or step_idx % args.log_interval == 0 or step_idx == len(train_loader):
                elapsed = time.time() - global_start
                eta = (elapsed / (step_idx + (epoch - 1) * len(train_loader))) * (args.epochs * len(train_loader) - (step_idx + (epoch - 1) * len(train_loader)))
                print(f'  E{epoch:02d}/{step_idx:04d} loss={loss.item():.3f} acc={train_correct/max(1,train_total):.3f} eta={eta/60:.0f}min')

        train_metrics = {
            'loss': train_loss_sum / max(1, train_total),
            'acc': train_correct / max(1, train_total),
            'epoch_sec': round(time.time() - epoch_start, 2),
        }
        print(json.dumps({'epoch': epoch, 'train': train_metrics}, ensure_ascii=False))
        scheduler.step()

        # Save best (score = train acc as proxy, real best selection via board eval)
        score = train_metrics['acc']
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_dir / 'best.pt')
            print(f'  * New best: epoch {epoch}')

        # Save last
        torch.save(model.state_dict(), save_dir / 'last.pt')

    summary = {
        **config,
        'best_score': best_score,
        'best_path': str(save_dir / 'best.pt'),
        'last_path': str(save_dir / 'last.pt'),
        'total_elapsed_sec': round(time.time() - global_start, 2),
    }
    (save_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\nDone. Best: epoch with score={best_score:.4f}')


if __name__ == '__main__':
    main()
