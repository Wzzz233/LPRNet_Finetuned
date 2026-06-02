#!/usr/bin/env python3
"""
Train 6-class plate type classifier.
Model: ResNet18, 3x72x224 input, /255 normalization.
Loss: CrossEntropy with sample_weight support.
"""
import os, sys, json, csv, time, math, random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

random.seed(20260602)
torch.manual_seed(20260602)
np.random.seed(20260602)

LPRNET_ROOT = Path("/home/wzzz/LPRNet")
MANIFEST_DIR = LPRNET_ROOT / "manifests_rebased" / "plate_type_classifier_6cls_20260602"
EXPERIMENT_DIR = LPRNET_ROOT / "experiments" / "plate_type_classifier_6cls_20260602"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["blue", "green", "yellow", "police", "embassy", "other"]
NUM_CLASSES = 6
IMG_W, IMG_H = 224, 72
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
LR_STEP = 8
LR_GAMMA = 0.5
MAX_TRAIN = 60000  # cap training to ~60K per class for speed


class PlateTypeDataset(Dataset):
    def __init__(self, csv_path, transform=None, max_samples=None):
        rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows.append(row)

        if max_samples and len(rows) > max_samples:
            # Stratified sample per class
            by_label = defaultdict(list)
            for r in rows:
                by_label[int(r["label"])].append(r)
            sampled = []
            per_class = max_samples // NUM_CLASSES
            for label in range(NUM_CLASSES):
                pool = by_label.get(label, [])
                if len(pool) > per_class:
                    random.shuffle(pool)
                    sampled.extend(pool[:per_class])
                else:
                    sampled.extend(pool)
            rows = sampled
            random.shuffle(rows)

        self.rows = rows
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.labels = [int(r["label"]) for r in self.rows]
        self.label_counts = Counter(self.labels)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            img = Image.open(row["img_path"]).convert("RGB")
            img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", (IMG_W, IMG_H), (128, 128, 128))
        img = self.transform(img)
        label = int(row["label"])
        return img, label, float(row.get("sample_weight", 1.0))


def create_model():
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels, weights in loader:
        imgs, labels, weights = imgs.to(device), labels.to(device), weights.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        weighted_loss = (loss * weights).mean()
        weighted_loss.backward()
        optimizer.step()
        total_loss += weighted_loss.item() * imgs.size(0)
        total += labels.size(0)
        correct += outputs.max(1)[1].eq(labels).sum().item()
    return total_loss / max(total, 1), 100. * correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = Counter()
    class_total = Counter()
    with torch.no_grad():
        for imgs, labels, weights in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.sum().item()
            preds = outputs.max(1)[1]
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            for l, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                class_total[l] += 1
                if l == p: class_correct[l] += 1
    acc = 100. * correct / max(total, 1)
    per_class = {}
    per_class_list = []
    for i in range(NUM_CLASSES):
        c = class_total.get(i, 0)
        if c > 0:
            pa = 100. * class_correct.get(i, 0) / c
            per_class[CLASS_NAMES[i]] = pa
            per_class_list.append(pa)
    macro = sum(per_class_list) / max(len(per_class_list), 1)
    return total_loss / max(total, 1), acc, macro, per_class


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_ds = PlateTypeDataset(MANIFEST_DIR / "train.csv", max_samples=MAX_TRAIN)
    val_c_ds = PlateTypeDataset(MANIFEST_DIR / "val_clean.csv")
    val_h_ds = PlateTypeDataset(MANIFEST_DIR / "val_hard.csv")
    val_x_ds = PlateTypeDataset(MANIFEST_DIR / "val_cross_source.csv")

    print(f"Train: {len(train_ds)} {dict(train_ds.label_counts)}", flush=True)
    print(f"Val clean: {len(val_c_ds)}", flush=True)
    print(f"Val hard: {len(val_h_ds)}", flush=True)
    print(f"Val cross: {len(val_x_ds)}", flush=True)

    # Weighted sampler to rebalance
    class_counts = train_ds.label_counts
    cw = [1.0 / max(class_counts.get(i, 0), 1) for i in range(NUM_CLASSES)]
    sw = [cw[l] for l in train_ds.labels]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    t_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=0)
    vc_loader = DataLoader(val_c_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    vh_loader = DataLoader(val_h_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    vx_loader = DataLoader(val_x_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, LR_STEP, LR_GAMMA)

    best_acc, best_ep = 0, -1
    history = []
    print(f"\n{'Ep':>3} {'Loss':>8} {'TAcc':>7} {'VAcc':>7} {'VMc':>7} {'HAcc':>7} {'XAcc':>7} {'LR':>9}", flush=True)
    print("-" * 65, flush=True)

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        loss, tacc = train_epoch(model, t_loader, criterion, optimizer, device)
        _, vacc, vmacro, vpc = evaluate(model, vc_loader, criterion, device)
        _, hacc, _, _ = evaluate(model, vh_loader, criterion, device)
        _, xacc, _, _ = evaluate(model, vx_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"{ep:3d} {loss:8.4f} {tacc:6.2f}% {vacc:6.2f}% {vmacro:6.2f}% "
              f"{hacc:6.2f}% {xacc:6.2f}% {lr:.1e} [{dt:.0f}s]", flush=True)
        history.append({"epoch": ep, "train_loss": loss, "train_acc": tacc,
                        "val_acc": vacc, "val_macro": vmacro,
                        "val_per_class": vpc,
                        "hard_acc": hacc, "cross_acc": xacc})

        if vacc > best_acc:
            best_acc, best_ep = vacc, ep
            torch.save(model.state_dict(), EXPERIMENT_DIR / "best_model.pth")
            print(f"  → best (val={vacc:.2f}%)", flush=True)
        if ep % 5 == 0:
            torch.save(model.state_dict(), EXPERIMENT_DIR / f"ep{ep}.pth")

    torch.save(model.state_dict(), EXPERIMENT_DIR / "final_model.pth")
    with open(EXPERIMENT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final eval on best model
    model.load_state_dict(torch.load(EXPERIMENT_DIR / "best_model.pth"))
    _, vacc, vmacro, vpc = evaluate(model, vc_loader, criterion, device)
    _, hacc, _, hpc = evaluate(model, vh_loader, criterion, device)
    _, xacc, _, xpc = evaluate(model, vx_loader, criterion, device)

    results = {
        "best_epoch": best_ep,
        "val_clean": {"accuracy": vacc, "macro": vmacro, "per_class": vpc},
        "val_hard": {"accuracy": hacc, "per_class": hpc},
        "val_cross": {"accuracy": xacc, "per_class": xpc},
    }
    with open(EXPERIMENT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Best epoch={best_ep} val={best_acc:.2f}%", flush=True)
    print(f"Final val_clean={vacc:.2f}% macro={vmacro:.2f}%", flush=True)
    print(f"  per-class: {vpc}", flush=True)
    print(f"Final val_hard={hacc:.2f}% per-class: {hpc}", flush=True)
    print(f"Final val_cross={xacc:.2f}% per-class: {xpc}", flush=True)
    print(f"Experiment: {EXPERIMENT_DIR}", flush=True)


if __name__ == "__main__":
    main()
