#!/usr/bin/env python3
"""Train corrected 6-class plate type classifier on warped plate inputs."""
from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

from plate_type_classifier_common import (
    CLASS_NAMES,
    IMG_H,
    IMG_W,
    ROOT,
    apply_real_board_aug,
    bgr_to_tensor_array,
    load_plate_bgr,
)

DATE = "20260602"
MANIFEST_DIR = ROOT / "manifests_rebased" / f"plate_type_classifier_6cls_warped_nocrop_{DATE}"
EXP_DIR = ROOT / "experiments" / f"plate_type_classifier_6cls_warped_nocrop_{DATE}"
EXP_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 6
BATCH_SIZE = 128
EPOCHS = 12
LR = 8e-4
WEIGHT_DECAY = 1e-4
MAX_TRAIN = 52000
SEED = 20260602

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class PlateTypeDataset(Dataset):
    def __init__(self, csv_path: Path, train: bool = False, max_samples: int | None = None):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if max_samples and len(rows) > max_samples:
            by_label = defaultdict(list)
            for row in rows:
                by_label[int(row["label"])].append(row)
            sampled = []
            per_class = max_samples // NUM_CLASSES
            for label in range(NUM_CLASSES):
                pool = by_label[label]
                random.shuffle(pool)
                sampled.extend(pool[:per_class] if len(pool) > per_class else pool)
            random.shuffle(sampled)
            rows = sampled
        self.rows = rows
        self.train = train
        self.labels = [int(row["label"]) for row in rows]
        self.label_counts = Counter(self.labels)
        self.rng = random.Random(SEED + (1 if train else 0))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        plate = load_plate_bgr(row, IMG_W, IMG_H)
        if self.train and row.get("source") == "special_v2" and row.get("label_name") in ("police", "embassy"):
            plate = apply_real_board_aug(plate, self.rng)
        arr = bgr_to_tensor_array(plate)
        img = torch.from_numpy(arr)
        label = int(row["label"])
        weight = float(row.get("sample_weight", "1.0") or 1.0)
        return img, label, weight


def build_model() -> nn.Module:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    cls_total = Counter()
    cls_correct = Counter()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels, _weights in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            total += labels.numel()
            correct += int((preds == labels).sum().item())
            for gt, pr in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                cls_total[gt] += 1
                conf[gt, pr] += 1
                if gt == pr:
                    cls_correct[gt] += 1
    per_class = {}
    vals = []
    for idx, name in enumerate(CLASS_NAMES):
        if cls_total[idx]:
            acc = 100.0 * cls_correct[idx] / cls_total[idx]
            per_class[name] = acc
            vals.append(acc)
    return {
        "accuracy": 100.0 * correct / max(1, total),
        "macro": sum(vals) / max(1, len(vals)),
        "per_class": per_class,
        "confusion_matrix": conf.tolist(),
        "total": total,
    }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for imgs, labels, weights in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True).float()
        logits = model(imgs)
        loss_vec = criterion(logits, labels)
        loss = (loss_vec * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += labels.numel()
        loss_sum += float(loss.item()) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum().item())
    return loss_sum / max(1, total), 100.0 * correct / max(1, total)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = PlateTypeDataset(MANIFEST_DIR / "train.csv", train=True, max_samples=MAX_TRAIN)
    val_clean = PlateTypeDataset(MANIFEST_DIR / "val_clean.csv")
    val_hard = PlateTypeDataset(MANIFEST_DIR / "val_hard.csv")
    val_cross = PlateTypeDataset(MANIFEST_DIR / "val_cross_source.csv")
    holdout = PlateTypeDataset(MANIFEST_DIR / "final_holdout.csv")

    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Train: {len(train_ds)} {dict(train_ds.label_counts)}", flush=True)
    print(f"Val clean/hard/cross/holdout: {len(val_clean)}/{len(val_hard)}/{len(val_cross)}/{len(holdout)}", flush=True)

    class_counts = train_ds.label_counts
    class_weights = [1.0 / max(1, class_counts.get(idx, 0)) for idx in range(NUM_CLASSES)]
    sample_weights = [class_weights[label] for label in train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    loaders = {
        "train": DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True),
        "val_clean": DataLoader(val_clean, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
        "val_hard": DataLoader(val_hard, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
        "val_cross": DataLoader(val_cross, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
        "final_holdout": DataLoader(holdout, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True),
    }

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = []
    best_score = -1.0
    best_epoch = -1
    print(f"{'Ep':>3} {'Loss':>8} {'TAcc':>7} {'Clean':>7} {'Hard':>7} {'Cross':>7} {'HoldEmb':>8} {'sec':>6}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, loaders["train"], optimizer, criterion, device)
        clean = run_eval(model, loaders["val_clean"], device)
        hard = run_eval(model, loaders["val_hard"], device)
        cross = run_eval(model, loaders["val_cross"], device)
        hold = run_eval(model, loaders["final_holdout"], device) if len(holdout) else {"accuracy": 0.0, "per_class": {}}
        scheduler.step()
        elapsed = time.time() - start
        hold_emb = hold.get("per_class", {}).get("embassy", 0.0)
        print(f"{epoch:3d} {train_loss:8.4f} {train_acc:6.2f}% {clean['accuracy']:6.2f}% {hard['accuracy']:6.2f}% {cross['accuracy']:6.2f}% {hold_emb:7.2f}% {elapsed:6.0f}", flush=True)
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_clean": clean, "val_hard": hard, "val_cross": cross, "final_holdout": hold}
        history.append(row)
        score = clean["macro"] + 0.5 * hard["macro"] + hold_emb
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), EXP_DIR / "best_model.pth")
            print(f"  best epoch={epoch} score={score:.2f}", flush=True)
        if epoch % 4 == 0:
            torch.save(model.state_dict(), EXP_DIR / f"ep{epoch}.pth")

    torch.save(model.state_dict(), EXP_DIR / "final_model.pth")
    with open(EXP_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    model.load_state_dict(torch.load(EXP_DIR / "best_model.pth", map_location=device))
    results = {
        "best_epoch": best_epoch,
        "val_clean": run_eval(model, loaders["val_clean"], device),
        "val_hard": run_eval(model, loaders["val_hard"], device),
        "val_cross": run_eval(model, loaders["val_cross"], device),
        "final_holdout": run_eval(model, loaders["final_holdout"], device) if len(holdout) else {},
    }
    with open(EXP_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done. best_epoch={best_epoch} exp={EXP_DIR}", flush=True)


if __name__ == "__main__":
    main()

