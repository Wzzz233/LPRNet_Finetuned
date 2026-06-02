#!/usr/bin/env python3
"""Quick debug: run 1 epoch on 2000 samples to test pipeline."""
import sys, csv, time, random
from collections import defaultdict, Counter
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

random.seed(42)
torch.manual_seed(42)
LPRNET_ROOT = Path("/home/wzzz/LPRNet")
MANIFEST = LPRNET_ROOT / "manifests_rebased/plate_type_classifier_6cls_20260602/train.csv"
IMG_W, IMG_H = 224, 72
BATCH_SIZE = 64
NUM_CLASSES = 6

class DebugDS(Dataset):
    def __init__(self, csv_path, max_n=2000):
        rows = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
        # Stratify
        by_lbl = defaultdict(list)
        for r in rows:
            by_lbl[int(r["label"])].append(r)
        sampled = []
        per = max_n // NUM_CLASSES
        for lbl in range(NUM_CLASSES):
            pool = by_lbl.get(lbl, [])
            random.shuffle(pool)
            sampled.extend(pool[:per])
        random.shuffle(sampled)
        self.rows = sampled
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        print(f"DS: {len(self.rows)} samples, classes={Counter(r['label'] for r in self.rows)}", flush=True)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["img_path"]).convert("RGB").resize((IMG_W, IMG_H), Image.LANCZOS)
        return self.tf(img), int(r["label"])

import time as time_mod
t0 = time_mod.time()
ds = DebugDS(MANIFEST, 2000)
print(f"Dataset init: {time_mod.time()-t0:.1f}s", flush=True)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

loader = DataLoader(ds, BATCH_SIZE, shuffle=True, num_workers=0)
print(f"DataLoader created", flush=True)

model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

# Train 1 epoch
model.train()
total, correct = 0, 0
t0 = time_mod.time()
for bi, (imgs, labels) in enumerate(loader):
    imgs, labels = imgs.to(device), labels.to(device)
    optimizer.zero_grad()
    out = model(imgs)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    total += labels.size(0)
    correct += out.max(1)[1].eq(labels).sum().item()
    if bi % 5 == 0:
        dt = time_mod.time() - t0
        imgs_per_sec = (bi + 1) * BATCH_SIZE / max(dt, 0.01)
        print(f"  batch {bi}: loss={loss.item():.4f} acc={correct/max(total,1)*100:.1f}% "
              f"imgs/sec={imgs_per_sec:.0f} dt={dt:.1f}s", flush=True)
print(f"Epoch done: {correct}/{total} = {100.*correct/total:.1f}%", flush=True)
