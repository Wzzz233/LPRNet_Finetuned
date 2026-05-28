#!/usr/bin/env python3
"""Evaluate green ResNet18 sidecar on police province val set.
No training, no export, read-only eval.
"""
import csv, os, sys, json, time
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torchvision.models as models

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import CHARS

PROVINCE_COUNT = 31
PROVINCES = CHARS[:PROVINCE_COUNT]

OUT_DIR = ROOT / 'experiments/police_green_resnet18_transfer_20260528'
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_MANIFEST = ROOT / 'manifests_rebased/police_province_sidecar_20260528/val.csv'

CHECKPOINTS = {
    'G0_baseline_repro': ROOT / 'experiments/routeA_nextstage_20260512/G0_baseline_repro/best.pt',
    'B3_fullplate_gray3': ROOT / 'experiments/routeA_prime_quadwarp_20260512/B3_fullplate_gray3_224x72_bal31/best.pt',
}

def build_resnet18_1ch(num_classes=31, checkpoint_path=None):
    """Build ResNet18 with 1-channel input, load checkpoint."""
    model = models.resnet18(weights=None)
    old_conv = model.conv1
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if checkpoint_path:
        state = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model

def preprocess_gray(img_path, size=(224, 72)):
    """Load image, resize, convert to single-channel gray, normalize to [0,1]."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(str(img_path))
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # single channel
    tensor = torch.from_numpy(gray.astype('float32') / 255.0).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return tensor

def evaluate(model, manifest_path, device='cpu'):
    """Evaluate province accuracy on a manifest."""
    rows = list(csv.DictReader(open(manifest_path)))
    correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    confusion = Counter()
    all_preds = []
    all_gts = []

    for r in rows:
        img_path = r['img_path']
        if not os.path.isabs(img_path):
            img_path = str(ROOT / img_path)
        if not Path(img_path).exists():
            continue

        img_t = preprocess_gray(img_path).to(device)
        gt_label = int(r['label'])

        with torch.no_grad():
            logits = model(img_t)  # [1, 31]
            pred = int(logits.argmax(dim=1).item())
            conf = float(torch.softmax(logits, dim=1)[0, pred].item())

        total += 1
        gt_prov = PROVINCES[gt_label]
        pred_prov = PROVINCES[pred]
        per_class_total[gt_prov] += 1

        if pred == gt_label:
            correct += 1
            per_class_correct[gt_prov] += 1
        else:
            confusion[(gt_prov, pred_prov)] += 1

        all_preds.append({'img_path': img_path, 'gt_label': gt_label, 'gt_prov': gt_prov,
                          'pred_label': pred, 'pred_prov': pred_prov,
                          'correct': pred == gt_label, 'confidence': round(conf, 4)})
        all_gts.append(gt_label)

    return {
        'correct': correct, 'total': total,
        'accuracy': correct / max(1, total),
        'per_class': {p: {'correct': per_class_correct[p], 'total': per_class_total[p],
                          'acc': per_class_correct[p] / max(1, per_class_total[p])}
                      for p in sorted(per_class_total.keys())},
        'confusion': dict(confusion.most_common()),
        'predictions': all_preds,
    }

def main():
    # Read manifest for audit
    val_rows = list(csv.DictReader(open(VAL_MANIFEST)))
    print(f"Police val manifest: {len(val_rows)} samples")

    # Province distribution
    prov_dist = Counter(r['province'] for r in val_rows)
    print(f"Province distribution: {len(prov_dist)} types, per province: {dict(sorted(prov_dist.items()))}")

    # Mapping audit
    print(f"\n=== Label Mapping Audit ===")
    print(f"PROVINCE_COUNT: {PROVINCE_COUNT}")
    print(f"PROVINCES: {PROVINCES}")

    with open(ROOT / 'keys/police_keys.txt') as f:
        police_chars = [l.strip() for l in f if l.strip()]
    print(f"Police keys[:31]: {police_chars[:31]}")
    match = PROVINCES == police_chars[:31]
    print(f"Mapping match: {match}")

    if not match:
        print("FATAL: label mapping mismatch, cannot evaluate!")
        with open(OUT_DIR / 'mapping_audit.txt', 'w') as f:
            f.write(f"PROVINCES: {PROVINCES}\n")
            f.write(f"Police keys: {police_chars[:31]}\n")
            f.write(f"Match: False\n")
            f.write("Cannot evaluate green sidecar on police data - label mapping inconsistent.\n")
        return 1

    # Mapping audit OK
    with open(OUT_DIR / 'mapping_audit.txt', 'w') as f:
        f.write(f"Label mapping match: True\n")
        f.write(f"PROVINCE_COUNT: {PROVINCE_COUNT}\n")
        f.write(f"All 31 provinces identical between CHARS and police_keys.txt\n")
        f.write(f"Order: {', '.join(PROVINCES)}\n")

    # Evaluate each checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    results = {}
    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n=== Evaluating {name} ===")
        print(f"  Checkpoint: {ckpt_path}")

        t0 = time.time()
        model = build_resnet18_1ch(num_classes=PROVINCE_COUNT, checkpoint_path=ckpt_path)
        model = model.to(device)
        print(f"  Model loaded in {time.time()-t0:.1f}s")
        print(f"  conv1 in_channels: {model.conv1.in_channels}")
        print(f"  fc out_features: {model.fc.out_features}")

        t0 = time.time()
        metrics = evaluate(model, VAL_MANIFEST, device)
        dt = time.time() - t0
        print(f"  Evaluated {metrics['total']} samples in {dt:.1f}s")
        print(f"  Accuracy: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']*100:.2f}%")
        results[name] = metrics

        # Show per-class
        print(f"  Per-class accuracy:")
        for prov in sorted(metrics['per_class'].keys()):
            pc = metrics['per_class'][prov]
            print(f"    {prov}: {pc['correct']}/{pc['total']} = {pc['acc']*100:.0f}%")

        # Top confusions
        if metrics['confusion']:
            print(f"  Top confusions:")
            for (gt, pred), cnt in list(metrics['confusion'].items())[:10]:
                print(f"    {gt}->{pred}: {cnt}")

        # Save predictions CSV
        pred_csv = OUT_DIR / f'predictions_{name}.csv'
        with open(pred_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['img_path','gt_label','gt_prov','pred_label','pred_prov','correct','confidence'])
            w.writeheader()
            w.writerows(metrics['predictions'])
        print(f"  Predictions: {pred_csv}")

        # Save confusion CSV
        conf_csv = OUT_DIR / f'province_confusion_{name}.csv'
        with open(conf_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['gt', 'pred', 'count'])
            for (gt, pred), cnt in metrics['confusion'].items():
                w.writerow([gt, pred, cnt])

    # Summary JSON
    summary = {
        'val_manifest': str(VAL_MANIFEST),
        'val_samples': len(val_rows),
        'label_mapping_match': match,
        'provinces': PROVINCES,
        'checkpoints': {},
    }
    for name, m in results.items():
        summary['checkpoints'][name] = {
            'path': str(CHECKPOINTS[name]),
            'accuracy': round(m['accuracy'], 4),
            'correct': m['correct'],
            'total': m['total'],
            'per_class_acc': {p: round(m['per_class'][p]['acc'], 4) for p in sorted(m['per_class'])},
        }

    summary_path = OUT_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary: {summary_path}")

    # Comparison with police sidecar
    print(f"\n=== Comparison ===")
    police_sidecar_acc = 1.0
    for name in results:
        print(f"  {name}: {results[name]['accuracy']*100:.2f}% vs police sidecar {police_sidecar_acc*100:.1f}%")

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
