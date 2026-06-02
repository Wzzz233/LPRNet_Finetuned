#!/usr/bin/env python3
"""Detailed evaluation for the corrected warped plate type classifier."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from plate_type_classifier_common import CLASS_NAMES, IMG_H, IMG_W, ROOT, bgr_to_tensor_array, load_plate_bgr

DATE = "20260602"
MANIFEST_DIR = ROOT / "manifests_rebased" / f"plate_type_classifier_6cls_warped_nocrop_{DATE}"
EXP_DIR = ROOT / "experiments" / f"plate_type_classifier_6cls_warped_nocrop_{DATE}"
NUM_CLASSES = 6


class EvalDataset(Dataset):
    def __init__(self, csv_path: Path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        plate = load_plate_bgr(row, IMG_W, IMG_H)
        return torch.from_numpy(bgr_to_tensor_array(plate)), int(row["label"]), row


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    rows = [b[2] for b in batch]
    return imgs, labels, rows


def build_model():
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def eval_split(model, csv_path: Path, device: torch.device) -> dict:
    ds = EvalDataset(csv_path)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate)
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    by_source = defaultdict(lambda: [0, 0])
    cls_total = Counter()
    cls_correct = Counter()
    high_risk = []
    all_conf = []
    all_margin = []
    model.eval()
    with torch.no_grad():
        for imgs, labels, rows in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            top2 = np.sort(probs, axis=1)[:, -2:]
            margins = top2[:, 1] - top2[:, 0]
            for idx, row in enumerate(rows):
                gt = int(labels[idx].item())
                pr = int(preds[idx])
                source = row.get("source", "")
                conf[gt, pr] += 1
                cls_total[gt] += 1
                by_source[source][1] += 1
                if gt == pr:
                    cls_correct[gt] += 1
                    by_source[source][0] += 1
                c = float(probs[idx, pr])
                m = float(margins[idx])
                all_conf.append(c)
                all_margin.append(m)
                gt_name = CLASS_NAMES[gt]
                pr_name = CLASS_NAMES[pr]
                risky = False
                if gt_name in ("blue", "green", "yellow", "other") and pr_name in ("police", "embassy"):
                    risky = True
                if gt_name == "police" and pr_name == "embassy":
                    risky = True
                if gt_name == "embassy" and pr_name == "police":
                    risky = True
                if risky:
                    high_risk.append({
                        "img_path": row.get("img_path", ""),
                        "source": source,
                        "label": gt_name,
                        "prediction": pr_name,
                        "conf": c,
                        "margin": m,
                        "plate_text": row.get("plate_text", ""),
                    })
    total = int(conf.sum())
    correct = int(np.trace(conf))
    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        if cls_total[idx]:
            per_class[name] = 100.0 * cls_correct[idx] / cls_total[idx]
    return {
        "total": total,
        "overall_accuracy": 100.0 * correct / max(1, total),
        "macro_accuracy": sum(per_class.values()) / max(1, len(per_class)),
        "per_class_accuracy": per_class,
        "per_source_accuracy": {k: 100.0 * v[0] / max(1, v[1]) for k, v in sorted(by_source.items())},
        "confusion_matrix": conf.tolist(),
        "confidence_distribution": summarize(all_conf),
        "margin_distribution": summarize(all_margin),
        "high_risk_false_routes": high_risk[:200],
        "high_risk_count": len(high_risk),
    }


def summarize(vals):
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    ckpt = EXP_DIR / "best_model.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    results = {}
    for split in ["val_clean", "val_hard", "val_cross_source", "final_holdout"]:
        path = MANIFEST_DIR / f"{split}.csv"
        if path.exists():
            results[split] = eval_split(model, path, device)
            print(split, results[split]["overall_accuracy"], results[split].get("per_class_accuracy", {}), flush=True)
    with open(EXP_DIR / "eval_full_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

