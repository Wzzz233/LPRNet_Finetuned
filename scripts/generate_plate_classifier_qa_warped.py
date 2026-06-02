#!/usr/bin/env python3
"""Generate QA sheets for corrected warped plate type classifier inputs."""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from plate_type_classifier_common import CLASS_NAMES, apply_real_board_aug, load_plate_bgr

ROOT = Path("/home/wzzz/LPRNet")
DATE = "20260602"
MANIFEST_DIR = ROOT / "manifests_rebased" / f"plate_type_classifier_6cls_warped_nocrop_{DATE}"
QA_DIR = Path("/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/plate_type_classifier_6cls_warped_nocrop_20260602")
QA_DIR.mkdir(parents=True, exist_ok=True)

RNG = random.Random(20260602)


def make_sheet(items, title, out_path, cols=8):
    if not items:
        return
    cols = min(cols, len(items))
    rows = math.ceil(len(items) / cols)
    cell_w, cell_h = 224, 96
    sheet = np.zeros((rows * cell_h + 34, cols * cell_w, 3), dtype=np.uint8) + 238
    cv2.putText(sheet, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    for idx, (img, label) in enumerate(items):
        r, c = divmod(idx, cols)
        x, y = c * cell_w, 34 + r * cell_h
        sheet[y:y + 72, x:x + 224] = img
        cv2.putText(sheet, label[:34], (x + 2, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), sheet)


def load_rows(name):
    path = MANIFEST_DIR / name
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def by_class_sheet(split):
    rows = load_rows(f"{split}.csv")
    buckets = defaultdict(list)
    for row in rows:
        cls = row["label_name"]
        if len(buckets[cls]) >= 8:
            continue
        img = load_plate_bgr(row)
        buckets[cls].append((img, f"{cls}|{row['source']}|{row.get('plate_text','')[:8]}"))
    items = []
    for cls in CLASS_NAMES:
        items.extend(buckets.get(cls, []))
    make_sheet(items, f"{split}: warped 224x72 by class", QA_DIR / f"{split}_by_class.png", cols=8)


def real_vs_synth():
    rows = load_rows("train.csv")
    hold = load_rows("final_holdout.csv")
    special = [r for r in rows if r.get("source") == "special_v2" and r.get("label_name") == "embassy"][:24]
    real = hold[:24]
    raw_items = []
    aug_items = []
    real_items = []
    for row in special[:16]:
        img = load_plate_bgr(row)
        raw_items.append((img, f"special|{row.get('plate_text','')[:12]}"))
        aug_items.append((apply_real_board_aug(img, RNG), f"aug|{row.get('plate_text','')[:12]}"))
    for row in real[:16]:
        img = load_plate_bgr(row)
        real_items.append((img, Path(row["img_path"]).name[:22]))
    make_sheet(raw_items, "special_v2 embassy raw warp", QA_DIR / "special_v2_embassy_raw_warp.png", cols=8)
    make_sheet(aug_items, "special_v2 embassy with real-board augmentation", QA_DIR / "special_v2_embassy_augmented.png", cols=8)
    make_sheet(real_items, "real embassy board fc224 holdout", QA_DIR / "real_embassy_fc224.png", cols=8)
    make_sheet(real_items + raw_items[:8] + aug_items[:8], "real vs synthetic vs augmented embassy", QA_DIR / "real_vs_synth_embassy.png", cols=8)


def source_sheet():
    rows = []
    for name in ["train.csv", "val_clean.csv", "val_hard.csv", "val_cross_source.csv", "final_holdout.csv"]:
        rows.extend(load_rows(name))
    buckets = defaultdict(list)
    for row in rows:
        key = f"{row['source']}|{row['label_name']}"
        if len(buckets[key]) >= 4:
            continue
        buckets[key].append((load_plate_bgr(row), key))
    items = []
    for key in sorted(buckets):
        items.extend(buckets[key])
    make_sheet(items[:64], "source style grid, all warped inputs", QA_DIR / "source_style_grid.png", cols=8)


def main():
    for split in ["train", "val_clean", "val_hard", "val_cross_source", "final_holdout"]:
        by_class_sheet(split)
    real_vs_synth()
    source_sheet()
    print(f"QA written: {QA_DIR}")
    for p in sorted(QA_DIR.glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()

