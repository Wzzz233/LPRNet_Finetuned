#!/usr/bin/env python3
"""B-line: High-resolution province classifier diagnostic.
Train a tiny 31-class province classifier on native-res warp province crops.
Test on Cluster2/3 to see if province info exists at higher resolution."""
import csv, json, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'src' / 'utils'))

# ── Province mapping ────────────────────────────────────────────────
ALL_PROVS = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
             '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
             '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
PROV2IDX = {p: i for i, p in enumerate(ALL_PROVS)}

class ProvinceCropDataset(Dataset):
    def __init__(self, samples):
        """samples: list of (province_crop_BGR, province_char, gt_full_text)"""
        self.samples = [(cv2.resize(img, (64, 80)), PROV2IDX[label]) for img, label, _ in samples]
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        x = img.astype(np.float32).transpose(2, 0, 1)
        x = (x - 127.5) * 0.0078125
        return torch.from_numpy(x), label

class TinyProvinceNet(nn.Module):
    def __init__(self, num_classes=31):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),  # 64x80 → 32x40
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 32x40 → 16x20
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16x20 → 8x10
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), # 8x10 → 1x1
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)

# ── Extract training data from prov_degrade manifest ─────────────────
def extract_training_provinces():
    """Extract province crops from prov_degrade_train_v1 manifest."""
    print("Extracting training province crops...")
    manifest = ROOT / 'manifests' / 'province_degrade_train_v1' / 'train_province_degrade_v1.csv'
    samples = []
    with open(manifest, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row['img_path']
            text = row['text']
            if not text: continue
            pquad = np.float32([[float(row['quad_1x']),float(row['quad_1y'])],
                                 [float(row['quad_2x']),float(row['quad_2y'])],
                                 [float(row['quad_3x']),float(row['quad_3y'])],
                                 [float(row['quad_4x']),float(row['quad_4y'])]])
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Warp at native resolution (larger than 94x24)
            w_top = max(np.linalg.norm(pquad[1]-pquad[0]), np.linalg.norm(pquad[2]-pquad[3]))
            h_left = max(np.linalg.norm(pquad[3]-pquad[0]), np.linalg.norm(pquad[2]-pquad[1]))
            dst_w, dst_h = max(1, int(round(w_top))), max(1, int(round(h_left)))
            src_pts = pquad.astype(np.float32)
            dst_pts = np.float32([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
            
            # Province region: left ~32%
            pw = max(1, int(dst_w * 0.32))
            prov_crop = warped[:, :pw, :]
            samples.append((prov_crop, text[0], text))
            
            if len(samples) % 2000 == 0:
                print(f'  {len(samples)}...')
            if len(samples) >= 8000:
                break
    
    print(f'  Total: {len(samples)} samples')
    print(f'  Province distribution: {dict(sorted(Counter(s[1] for s in samples).items()))}')
    return samples

def extract_cluster_provinces(csv_path, crop_dir=None):
    """Extract province crops from Cluster CSV (using crop PPMs)."""
    samples = []
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop_path = row.get('local_crop_path', '')
            gt = row.get('gt_text', '').strip()
            if crop_path and Path(crop_path).exists() and gt:
                crop = cv2.imread(crop_path)
                if crop is None: continue
                h, w = crop.shape[:2]
                pw = max(1, int(w * 0.32))
                prov_crop = crop[:, :pw, :]
                samples.append((prov_crop, gt[0], gt))
    return samples

def extract_ocrin_provinces(csv_path, dump_dir=None):
    """Extract province region from 94x24 ocrin PPMs (lower-res baseline)."""
    samples = []
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ocrin_path = row.get('local_ocrin_path', '')
            gt = row.get('gt_text', '').strip()
            if ocrin_path and Path(ocrin_path).exists() and gt:
                ocrin = cv2.imread(ocrin_path)
                if ocrin is None: continue
                # Province region: left ~20% of 94 = 19px
                prov_crop = ocrin[:, :20, :]
                samples.append((prov_crop, gt[0], gt))
    return samples

# ── Main ────────────────────────────────────────────────────────────
print("=" * 60)
print("B-line: High-res province classifier diagnostic")
print("=" * 60)

# 1. Extract training data (from prov_degrade manifest, warp at native res)
train_samples = extract_training_provinces()

# 2. Extract Cluster2/3 evaluation data
c2_crop = extract_cluster_provinces(
    '/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv')
c3_crop = extract_cluster_provinces(
    '/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv')

# Also get ocrin (94x24) versions for comparison
c2_ocrin = extract_ocrin_provinces(
    '/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv')
c3_ocrin = extract_ocrin_provinces(
    '/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv')

print(f'\nTraining: {len(train_samples)} samples (native-res warp provinces)')
print(f'Cluster2 crop: {len(c2_crop)} samples')
print(f'Cluster3 crop: {len(c3_crop)} samples')
print(f'Cluster2 ocrin: {len(c2_ocrin)} samples')
print(f'Cluster3 ocrin: {len(c3_ocrin)} samples')

# 3. Train classifier
print(f'\n--- Training TinyProvinceNet ---')
dataset = ProvinceCropDataset(train_samples)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = TinyProvinceNet(31).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f'  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} val_acc={correct/total*100:.2f}%')

# 4. Evaluate on Cluster2/3
print(f'\n--- Evaluation ---')

def evaluate_model_on_samples(model, samples, name):
    ds = ProvinceCropDataset(samples)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model.eval()
    
    results = []
    with torch.no_grad():
        for i, (img, label) in enumerate(ds):
            img = img.unsqueeze(0).to(device)
            logits = model(img)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = logits.argmax(1).item()
            pred_prov = ALL_PROVS[pred_idx]
            gt_prov = ALL_PROVS[label]
            
            top3_vals, top3_idx = torch.topk(probs, 3)
            top3 = [(ALL_PROVS[idx.item()], val.item()) for idx, val in zip(top3_idx, top3_vals)]
            
            results.append({
                'gt': samples[i][2], 'gt_prov': gt_prov,
                'pred_prov': pred_prov, 'correct': pred_prov == gt_prov,
                'top3': top3, 'gt_prob': probs[label].item(),
            })
    
    correct = sum(1 for r in results if r['correct'])
    print(f'\n{name} ({len(results)} samples):')
    print(f'  Province accuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%')
    for r in results:
        top3_str = ' '.join(f'{p}({v:.2f})' for p,v in r['top3'])
        ck = '✓' if r['correct'] else '✗'
        print(f'  {ck} {r["gt"]:>12} → {r["pred_prov"]:>2}  (gt_prob={r["gt_prob"]:.3f})  top3: {top3_str}')
    
    return results

# High-res (crop) evaluation
c2_results = evaluate_model_on_samples(model, c2_crop, 'Cluster2 crop (native-res)')
c3_results = evaluate_model_on_samples(model, c3_crop, 'Cluster3 crop (native-res)')

# Low-res (ocrin) evaluation for comparison
c2o_results = evaluate_model_on_samples(model, c2_ocrin, 'Cluster2 ocrin (94x24)')
c3o_results = evaluate_model_on_samples(model, c3_ocrin, 'Cluster3 ocrin (94x24)')

# 5. Summary
print(f'\n{"="*60}')
print('B-LINE DIAGNOSIS SUMMARY')
print(f'{"="*60}')
for name, results in [('Cluster2 crop (native)', c2_results), ('Cluster3 crop (native)', c3_results),
                       ('Cluster2 ocrin (94x24)', c2o_results), ('Cluster3 ocrin (94x24)', c3o_results)]:
    correct = sum(1 for r in results if r['correct'])
    print(f'{name:<30}: {correct}/{len(results)} = {correct/len(results)*100:.1f}%')
PYEOF
