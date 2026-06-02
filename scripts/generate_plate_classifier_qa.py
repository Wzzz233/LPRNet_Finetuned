#!/usr/bin/env python3
"""
Generate QA contact sheets for 6-class plate type classifier.
After training: shows class distribution, source diversity, risky confusions.
"""
import os, sys, csv, math
from collections import defaultdict, Counter
from pathlib import Path
import cv2
import numpy as np

LPRNET_ROOT = Path("/home/wzzz/LPRNet")
MANIFEST_DIR = LPRNET_ROOT / "manifests_rebased" / "plate_type_classifier_6cls_20260602"
EXPERIMENT_DIR = LPRNET_ROOT / "experiments" / "plate_type_classifier_6cls_20260602"

# For desktop output
QA_DIR = Path("/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/plate_type_classifier_6cls_20260602")
QA_DIR.mkdir(parents=True, exist_ok=True)

SAMP = 8
CLASS_NAMES = ["blue", "green", "yellow", "police", "embassy", "other"]


def resize_pad(img, dw=224, dh=72):
    h, w = img.shape[:2]
    s = min(dw / w, dh / h)
    nw, nh = int(w * s), int(h * s)
    interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_LINEAR
    res = cv2.resize(img, (nw, nh), interpolation=interp)
    c = np.zeros((dh, dw, 3), dtype=np.uint8)
    xo, yo = (dw - nw) // 2, (dh - nh) // 2
    c[yo:yo+nh, xo:xo+nw] = res
    return c


def make_sheet(samples, title, path, cols=8):
    if not samples: return
    n = len(samples); cols = min(cols, n); rows = math.ceil(n / cols)
    cw, ch = 224, 72 + 18
    sheet = np.zeros((rows * ch + 40, cols * cw, 3), dtype=np.uint8) + 240
    cv2.putText(sheet, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for idx, (img_224, label) in enumerate(samples):
        r, c = idx // cols, idx % cols
        x, y = c * cw, r * ch + 40
        sheet[y:y+72, x:x+224] = img_224
        cv2.putText(sheet, label[:30], (x + 2, y + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1)
    cv2.imwrite(str(path), sheet)


def load_samples(csv_path):
    """Load images from manifest, sampled per class."""
    samples_by_class = defaultdict(list)
    sources_by_class = defaultdict(lambda: defaultdict(int))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            label = int(row["label"])
            cls = CLASS_NAMES[label]
            src = row["source"]
            sources_by_class[cls][src] += 1
            if len(samples_by_class[cls]) >= SAMP * 3:
                continue
            img = cv2.imread(row["img_path"])
            if img is None: continue
            rp = resize_pad(img)
            label_text = f"{src}|{row.get('plate_text','')[:12]}"
            samples_by_class[cls].append((rp, label_text))
    return samples_by_class, sources_by_class


def main():
    print("Generating QA contact sheets...", flush=True)

    for split in ["train", "val_clean", "val_hard", "val_cross_source"]:
        csv_path = MANIFEST_DIR / f"{split}.csv"
        if not csv_path.exists():
            continue
        samples_by_class, sources_by_class = load_samples(csv_path)

        # Class distribution sheet
        all_samples = []
        for cls in CLASS_NAMES:
            for im, t in samples_by_class.get(cls, [])[:SAMP]:
                all_samples.append((im, f"{cls}|{t}"))
        make_sheet(all_samples, f"{split} by class",
                   QA_DIR / f"{split}_by_class.png", cols=len(CLASS_NAMES))

        # Source distribution
        print(f"  {split}:")
        for cls in CLASS_NAMES:
            srcs = dict(sources_by_class.get(cls, {}))
            if srcs:
                print(f"    {cls}: {srcs}")

    # Other examples: show 领/港/澳/学/挂/黑
    other_examples = []
    for split in ["train", "val_clean", "val_hard", "val_cross_source"]:
        csv_path = MANIFEST_DIR / f"{split}.csv"
        if not csv_path.exists(): continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if int(row["label"]) != 5:
                    continue
                text = row.get("plate_text", "")
                if any(c in text for c in "领港澳学挂黑"):
                    if len(other_examples) >= SAMP * len(CLASS_NAMES):
                        break
                    img = cv2.imread(row["img_path"])
                    if img is None: continue
                    rp = resize_pad(img)
                    src = row["source"]
                    other_examples.append((rp, f"{src}|{text[:20]}"))
    make_sheet(other_examples, "other: 领/港/澳/学/挂/黑",
               QA_DIR / "other_examples.png", cols=8)

    # Risky confusion examples: load eval results if available
    eval_path = EXPERIMENT_DIR / "eval_full_results.json"
    if eval_path.exists():
        import json
        with open(eval_path) as f:
            eval_data = json.load(f)
        risky = []
        for split, metrics in eval_data.items():
            for r in metrics.get("high_risk_false_routes", []):
                if len(risky) >= 24:
                    break
                img = cv2.imread(r.get("img_path", ""))
                if img is None: continue
                rp = resize_pad(img)
                risky.append((rp, f"{r['label']}→{r['prediction']} c={r['conf']:.2f}"))
        make_sheet(risky, "risky confusions (false route)",
                   QA_DIR / "risky_confusion_examples.png", cols=6)

    # Source style grid: compare a class across sources  
    print("  source_style_grid.png", flush=True)
    grid_samples = []
    for src in ["ccpd2019", "ccpd2020", "crpd_single", "special_v2", "git_plate", "cblprd"]:
        for cls in ["blue", "green", "yellow", "police", "embassy", "other"]:
            if len(grid_samples) >= 36: break
            # Find first sample of this class×source
            for split in ["train", "val_clean", "val_hard", "val_cross_source"]:
                csv_path = MANIFEST_DIR / f"{split}.csv"
                if not csv_path.exists(): continue
                with open(csv_path) as f:
                    for row in csv.DictReader(f):
                        if row["source"] == src and row["label_name"] == cls:
                            img = cv2.imread(row["img_path"])
                            if img is None: continue
                            rp = resize_pad(img)
                            grid_samples.append((rp, f"{src}|{cls}"))
                            break
                if len(grid_samples) > 0 and grid_samples[-1][1].startswith(f"{src}|{cls}"):
                    break
    make_sheet(grid_samples, "source × class grid",
               QA_DIR / "source_style_grid.png", cols=6)

    print(f"\nQA images in: {QA_DIR}", flush=True)
    print(f"Files: {len(list(QA_DIR.glob('*.png')))}", flush=True)


if __name__ == "__main__":
    main()
