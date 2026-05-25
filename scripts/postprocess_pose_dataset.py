#!/usr/bin/env python3
"""Post-process: cap val.txt to a reasonable size + write cloud setup script."""

import os
import random
from pathlib import Path

random.seed(20260502)

POSE_DIR = Path("/home/wzzz/LPRNet/datasets/plate_true_quad_pose")
MAX_VAL = 3000  # keep val small for fast training validation
MAX_TEST = 10000

# ── 1. Cap val.txt ────────────────────────────────────────────────
val_path = POSE_DIR / "val.txt"
val_lines = val_path.read_text().strip().splitlines()
random.shuffle(val_lines)
val_capped = val_lines[:MAX_VAL]
val_path.write_text("\n".join(val_capped) + "\n")
print(f"val.txt: {len(val_lines)} -> {len(val_capped)} (capped to {MAX_VAL})")

# ── 2. Cap test.txt ───────────────────────────────────────────────
test_path = POSE_DIR / "test.txt"
test_lines = test_path.read_text().strip().splitlines()
random.shuffle(test_lines)
test_capped = test_lines[:MAX_TEST]
test_path.write_text("\n".join(test_capped) + "\n")
print(f"test.txt: {len(test_lines)} -> {len(test_capped)} (capped to {MAX_TEST})")

# ── 3. Summary stats ──────────────────────────────────────────────
print(f"\nFinal dataset stats:")
print(f"  Dataset root: {POSE_DIR}")
print(f"  dataset.yaml: {(POSE_DIR / 'dataset.yaml').exists()}")
train_lines = (POSE_DIR / "train.txt").read_text().strip().splitlines()
print(f"train.txt: {len(train_lines):>8} images")
print(f"  val.txt:  {len(val_capped):>8} images")
print(f"  test.txt: {len(test_capped):>8} images")

# Count labels per split
for split in ["train", "val", "test"]:
    label_dir = POSE_DIR / "labels" / split
    if label_dir.exists():
        n = len(os.listdir(label_dir))
        print(f"  labels/{split}/: {n} files")

print("\nDone.")
