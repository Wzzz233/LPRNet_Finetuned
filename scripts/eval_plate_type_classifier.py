#!/usr/bin/env python3
"""
Evaluate 6-class plate type classifier.
Outputs: overall/macro/per-class accuracy, per-source accuracy,
confusion matrix, confidence distribution, high-risk false route list.
"""
import os, sys, json, csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

LPRNET_ROOT = Path("/home/wzzz/LPRNet")
MANIFEST_DIR = LPRNET_ROOT / "manifests_rebased" / "plate_type_classifier_6cls_20260602"
EXPERIMENT_DIR = LPRNET_ROOT / "experiments" / "plate_type_classifier_6cls_20260602"
CHECKPOINT = EXPERIMENT_DIR / "best_model.pth"
IMG_W, IMG_H = 224, 72
BATCH_SIZE = 128
NUM_CLASSES = 6
CLASS_NAMES = {0: "blue", 1: "green", 2: "yellow", 3: "police", 4: "embassy", 5: "other"}
CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}

# High-risk confusion pairs
RISK_PAIRS = [
    (5, 3), (0, 3), (1, 3), (2, 3),  # other/blue/green/yellow → police
    (5, 4), (0, 4), (1, 4), (2, 4),  # other/blue/green/yellow → embassy
    (4, 3),  # embassy → police
    (3, 4),  # police → embassy
]


def load_model():
    from torchvision import models
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


class ManifestDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            img = Image.open(row["img_path"]).convert("RGB")
            img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
        except:
            img = Image.new("RGB", (IMG_W, IMG_H), (128, 128, 128))
        img = self.transform(img)
        label = int(row["label"])
        return img, label, row["source"], row.get("plate_text", ""), row.get("img_path", "")


def predict_loader(model, loader, device):
    """Run inference on a DataLoader, return detailed results."""
    model.to(device)
    model.eval()
    results = []

    with torch.no_grad():
        for imgs, labels, sources, texts, paths in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_preds = probs.topk(2, dim=1)
            top1_conf = top_probs[:, 0].cpu().tolist()
            top2_conf = top_probs[:, 1].cpu().tolist()
            top1_margin = [(c1 - c2) for c1, c2 in zip(top1_conf, top2_conf)]
            preds = top_preds[:, 0].cpu().tolist()

            for i in range(imgs.size(0)):
                results.append({
                    "label": labels[i].item(),
                    "prediction": preds[i],
                    "top1_conf": top1_conf[i],
                    "top1_margin": top1_margin[i],
                    "source": sources[i],
                    "text": texts[i],
                    "path": paths[i],
                })
    return results


def compute_metrics(results, split_name):
    """Compute all metrics from prediction results."""
    n = len(results)
    if n == 0:
        return {"error": "empty"}

    # Overall accuracy
    correct = sum(1 for r in results if r["label"] == r["prediction"])
    overall_acc = 100. * correct / n

    # Per-class
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        per_class[r["label"]]["total"] += 1
        if r["label"] == r["prediction"]:
            per_class[r["label"]]["correct"] += 1
    per_class_acc = {
        CLASS_NAMES[lc]: 100. * pc["correct"] / pc["total"]
        for lc, pc in per_class.items()
    }
    macro_acc = sum(per_class_acc.values()) / len(per_class_acc)

    # Per-source
    per_source = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        per_source[r["source"]]["total"] += 1
        if r["label"] == r["prediction"]:
            per_source[r["source"]]["correct"] += 1
    per_source_acc = {
        src: 100. * ps["correct"] / ps["total"]
        for src, ps in per_source.items()
    }

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for r in results:
        cm[r["label"], r["prediction"]] += 1

    # Confidence distribution
    confs = [r["top1_conf"] for r in results]
    conf_hist = {
        "count": n,
        "mean": float(np.mean(confs)),
        "median": float(np.median(confs)),
        "p5": float(np.percentile(confs, 5)),
        "p25": float(np.percentile(confs, 25)),
        "p75": float(np.percentile(confs, 75)),
        "p95": float(np.percentile(confs, 95)),
        "conf_low_count": sum(1 for c in confs if c < 0.5),
        "conf_low_pct": 100. * sum(1 for c in confs if c < 0.5) / n,
    }

    # Margin distribution
    margins = [r["top1_margin"] for r in results]
    margin_hist = {
        "mean": float(np.mean(margins)),
        "median": float(np.median(margins)),
        "p5": float(np.percentile(margins, 5)),
        "p25": float(np.percentile(margins, 25)),
        "p75": float(np.percentile(margins, 75)),
        "p95": float(np.percentile(margins, 95)),
        "margin_low_count": sum(1 for m in margins if m < 0.1),
        "margin_low_pct": 100. * sum(1 for m in margins if m < 0.1) / n,
    }

    # High-risk false routes
    risky = []
    for idx, r in enumerate(results):
        label = r["label"]
        pred = r["prediction"]
        # Check risk pairs
        for true_cls, false_cls in RISK_PAIRS:
            if label == true_cls and pred == false_cls:
                risky.append({
                    "img_path": r["path"],
                    "label": CLASS_NAMES[label],
                    "prediction": CLASS_NAMES[pred],
                    "source": r["source"],
                    "text": r["text"],
                    "conf": r["top1_conf"],
                    "margin": r["top1_margin"],
                    "type": "false_route",
                })
        # Also check special: 领/港/澳/学/挂/黑 being predicted as police/embassy
        if label == CLASS_IDS["other"] and pred in (CLASS_IDS["police"], CLASS_IDS["embassy"]):
            text = r["text"]
            if any(c in text for c in "领港澳学挂"):
                risky.append({
                    "img_path": r["path"],
                    "label": f"other({text[:6]})",
                    "prediction": CLASS_NAMES[pred],
                    "source": r["source"],
                    "text": text,
                    "conf": r["top1_conf"],
                    "margin": r["top1_margin"],
                    "type": "special_other_as_police_embassy",
                })

    # Sort risky by margin ascending (most dangerous first)
    risky.sort(key=lambda x: x["margin"])

    return {
        "overall_accuracy": round(overall_acc, 2),
        "macro_accuracy": round(macro_acc, 2),
        "per_class_accuracy": {k: round(v, 2) for k, v in per_class_acc.items()},
        "per_source_accuracy": {k: round(v, 2) for k, v in per_source_acc.items()},
        "confusion_matrix": cm.tolist(),
        "confidence_distribution": conf_hist,
        "margin_distribution": margin_hist,
        "high_risk_false_routes": risky[:50],  # top 50 most risky
        "total_samples": n,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model()
    print(f"Model loaded from {CHECKPOINT}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    splits = ["val_clean", "val_hard", "val_cross_source"]
    all_metrics = {}

    for split in splits:
        csv_path = MANIFEST_DIR / f"{split}.csv"
        if not csv_path.exists():
            continue
        dataset = ManifestDataset(csv_path, transform=transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"\nEvaluating {split} ({len(dataset)} samples)...")
        results = predict_loader(model, loader, device)
        metrics = compute_metrics(results, split)
        all_metrics[split] = metrics

        # Print summary
        print(f"  Overall: {metrics['overall_accuracy']:.2f}%")
        print(f"  Macro:   {metrics['macro_accuracy']:.2f}%")
        print(f"  Per-class: {metrics['per_class_accuracy']}")
        print(f"  Per-source: {metrics['per_source_accuracy']}")
        print(f"  Risky routes: {len(metrics['high_risk_false_routes'])}")
        # Print confusion matrix
        print(f"  Confusion matrix (rows=label, cols=prediction):")
        cm = metrics['confusion_matrix']
        header = "        " + " ".join(f"{CLASS_NAMES[i]:>8}" for i in range(NUM_CLASSES))
        print(header)
        for i in range(NUM_CLASSES):
            row = " ".join(f"{cm[i][j]:8d}" for j in range(NUM_CLASSES))
            print(f"  {CLASS_NAMES[i]:>6} {row}")

    # Save full metrics
    output_path = EXPERIMENT_DIR / "eval_full_results.json"
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2, cls=NumpyEncoder)
    print(f"\nFull eval results saved to {output_path}")

    # Summary for training report
    print("\n=== HIGH-RISK FALSE ROUTE SUMMARY ===")
    for split, metrics in all_metrics.items():
        risky = metrics.get("high_risk_false_routes", [])
        if not risky:
            continue
        print(f"\n{split} ({len(risky)} risky):")
        for r in risky[:10]:
            print(f"  [{r['type']}] {r['source']} | "
                  f"true={r['label']} pred={r['prediction']} "
                  f"conf={r['conf']:.3f} margin={r['margin']:.3f} | {r['text'][:20]}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    main()
