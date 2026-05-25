#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a YOLOv8-pose dataset from CCPD2019, CCPD2020, CRPD.

Output structure:
  <output_dir>/
    images/{train,val,test}/   (symlinks)
    labels/{train,val,test}/   (.txt YOLO pose labels)
    lists/                     (per-slice image path lists)
    train.txt                  (weighted image path list for training)
    val.txt                    (validation image path list)
    test.txt                   (test image path list)
    dataset.yaml               (Ultralytics dataset config)

Usage:
  python build_true_quad_pose_dataset.py --output_dir ./datasets/plate_true_quad_pose
"""

import argparse
import csv
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────
LPRNET_ROOT = Path("/home/wzzz/LPRNet")
CCPD2019_ROOT = LPRNET_ROOT / "datasets/CCPD2019"
CCPD2020_ROOT = LPRNET_ROOT / "datasets/CCPD2020" / "ccpd_green"
CRPD_ROOT = LPRNET_ROOT / "datasets/CRPD_all"
CCPD2020_CSV_TRAIN = LPRNET_ROOT / "labels" / "curriculum_gray3" / "ccpd2020_train.csv"
CCPD2020_CSV_TEST = LPRNET_ROOT / "labels" / "curriculum_gray3" / "ccpd2020_test.csv"

# Target per-epoch sampling ratios (from the plan)
SLICE_RATIOS = {
    "ccpd2019_normal_blue": 0.20,
    "ccpd2019_hard_blue": 0.15,
    "ccpd2020_green_easy": 0.12,
    "ccpd2020_green_mid": 0.14,
    "ccpd2020_green_hard": 0.14,
    "crpd_blue": 0.10,
    "crpd_yellow": 0.08,
    "crpd_multi_complex": 0.07,
}
# Compact slice tags for filenames
SLICE_TAGS = {
    "ccpd2019_normal_blue": "c2019_nb",
    "ccpd2019_hard_blue": "c2019_hb",
    "ccpd2020_green_easy": "c2020_ge",
    "ccpd2020_green_mid": "c2020_gm",
    "ccpd2020_green_hard": "c2020_gh",
    "crpd_blue": "crpd_b",
    "crpd_yellow": "crpd_y",
    "crpd_multi_complex": "crpd_mc",
}

random.seed(20260502)

# ── CCPD quad parsing ─────────────────────────────────────────────────
def parse_ccpd_quad_from_path(img_path: str) -> np.ndarray:
    """Parse GT quad from CCPD-style filename. Returns (4,2) float32."""
    stem = Path(img_path).stem
    parts = stem.split("-")
    if len(parts) < 4:
        return None
    points_text = parts[3]
    points = []
    try:
        for item in points_text.split("_"):
            xs, ys = item.split("&", 1)
            points.append((float(xs), float(ys)))
    except ValueError:
        return None
    if len(points) != 4:
        return None
    # CCPD order: TL, TR, BR, BL
    return np.asarray(points, dtype=np.float32)


def get_image_size(img_path: str) -> tuple:
    """Return (width, height) from image header without full decode."""
    with Image.open(img_path) as im:
        return im.size


# ── Quad metrics (for difficulty scoring) ──────────────────────────────
def side_length(quad: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(quad[j] - quad[i]))


def compute_quad_difficulty_score(quad: np.ndarray) -> float:
    """
    Score based on width_ratio + height_ratio.
    Higher = more extreme perspective.
    """
    p = quad.reshape(4, 2)  # TL=0, TR=1, BR=2, BL=3
    top_w = side_length(p, 0, 1)
    bot_w = side_length(p, 3, 2)
    left_h = side_length(p, 0, 3)
    right_h = side_length(p, 1, 2)
    eps = 1.0
    w_ratio = max(top_w, bot_w) / max(eps, min(top_w, bot_w))
    h_ratio = max(left_h, right_h) / max(eps, min(left_h, right_h))
    return w_ratio + h_ratio


# ── YOLO pose label generation ────────────────────────────────────────
def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Reorder quad points to TL, TR, BR, BL order using geometric heuristic.
    TL = smallest (x+y), BR = largest (x+y),
    TR = smallest (y-x), BL = largest (y-x).
    """
    quad = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = quad.sum(axis=1)
    diffs = np.diff(quad, axis=1).reshape(-1)
    ordered[0] = quad[np.argmin(sums)]   # TL
    ordered[2] = quad[np.argmax(sums)]   # BR
    ordered[1] = quad[np.argmin(diffs)]  # TR
    ordered[3] = quad[np.argmax(diffs)]  # BL
    return ordered


def quad_to_yolo_pose_line(quad: np.ndarray, img_w: int, img_h: int, cls_id: int = 0) -> str:
    """
    Convert quad to YOLO pose format.
    Format: class cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
    Keypoint order: TL(0), TR(1), BR(2), BL(3)
    All coordinates normalized by image width/height.
    Quad is reordered to TL/TR/BR/BL before writing.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: {img_w}x{img_h}")

    # Reorder to TL/TR/BR/BL
    quad = order_quad_points(quad)

    # Axis-aligned bounding box around quad
    xs = quad[:, 0]
    ys = quad[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    parts = [f"{cls_id}"]
    parts.append(f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    for i in range(4):
        parts.append(f"{quad[i, 0] / img_w:.6f} {quad[i, 1] / img_h:.6f} 2")

    return " ".join(parts)


# ── CCPD2019 ──────────────────────────────────────────────────────────
def build_ccpd2019(output_dir: Path, link_mode: str = "symlink"):
    """
    Process CCPD2019 using ALL subdirectories.

    Scans every ccpd_* subdirectory, classifies as normal_blue (ccpd_base)
    or hard_blue (all others), and does a per-subdir random 85/10/5 split
    into train/val/test so all difficulty types appear in all splits.

    Returns dict: {split: {slice_name: [(img_path, quad), ...]}}
    """
    NORMAL_DIRS = {"ccpd_base"}
    random.seed(20260502)

    results = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    total_scanned = 0

    for subdir in sorted(os.listdir(CCPD2019_ROOT)):
        subdir_path = CCPD2019_ROOT / subdir
        if not subdir_path.is_dir() or not subdir.startswith("ccpd_"):
            continue
        if subdir.endswith(".cache"):
            continue

        if subdir in NORMAL_DIRS:
            slice_name = "ccpd2019_normal_blue"
        else:
            slice_name = "ccpd2019_hard_blue"

        # Gather all image paths
        all_files = sorted(os.listdir(subdir_path))
        img_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not img_files:
            continue

        total_scanned += len(img_files)

        # Parse quad for each file and filter invalid
        valid = []
        for fname in img_files:
            rel_path = f"{subdir}/{fname}"
            quad = parse_ccpd_quad_from_path(rel_path)
            if quad is not None:
                valid.append((str(subdir_path / fname), quad))

        if not valid:
            continue

        # Shuffle and split
        random.shuffle(valid)
        n = len(valid)
        n_train = int(n * 0.85)
        n_val = int(n * 0.10)
        # n_test = n - n_train - n_val

        train_set = valid[:n_train]
        val_set = valid[n_train:n_train + n_val]
        test_set = valid[n_train + n_val:]

        for (img_path, quad) in train_set:
            results["train"][slice_name].append((img_path, quad))
        for (img_path, quad) in val_set:
            results["val"][slice_name].append((img_path, quad))
        for (img_path, quad) in test_set:
            results["test"][slice_name].append((img_path, quad))

    # Report
    for split_name in ["train", "val", "test"]:
        counts = {k: len(v) for k, v in results[split_name].items()}
        print(f"  CCPD2019 {split_name}: {counts}")

    return results


# ── CCPD2020 green ────────────────────────────────────────────────────
def build_ccpd2020(output_dir: Path, link_mode: str = "symlink"):
    """
    Process CCPD2020 green dataset.

    Computes difficulty score per image, splits into easy/mid/hard
    using percentile-based thresholds.

    Returns dict: {split: {slice_name: [(img_path, quad, text), ...]}}
    """
    results = {}

    for split_name, csv_path in [("train", CCPD2020_CSV_TRAIN), ("test", CCPD2020_CSV_TEST)]:
        if not csv_path.exists():
            print(f"  [SKIP] CCPD2020 CSV {csv_path} not found")
            continue

        rows = []
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["img_path"]
                if not os.path.isfile(img_path):
                    continue
                quad = parse_ccpd_quad_from_path(img_path)
                if quad is None:
                    continue
                text = row.get("text", "")
                # Get image size
                try:
                    w, h = get_image_size(img_path)
                except Exception:
                    continue
                score = compute_quad_difficulty_score(quad)
                rows.append((img_path, quad, text, score, w, h))

        results[split_name] = rows
        print(f"  CCPD2020 {split_name}: {len(rows)} images loaded")

    # Compute difficulty thresholds from training set
    train_rows = results.get("train", [])
    if train_rows:
        scores = np.array([r[3] for r in train_rows])
        p40 = np.percentile(scores, 40)
        p70 = np.percentile(scores, 70)
        print(f"  CCPD2020 difficulty thresholds: easy<{p40:.3f}, mid<{p70:.3f}, hard>={p70:.3f}")

        def classify(score):
            if score < p40:
                return "ccpd2020_green_easy"
            elif score < p70:
                return "ccpd2020_green_mid"
            return "ccpd2020_green_hard"

        # Build slice-mapped result
        mapped = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
        for split_name, rows in results.items():
            for img_path, quad, text, score, w, h in rows:
                slice_name = classify(score)
                mapped[split_name][slice_name].append((img_path, quad, text))

        for split_name in ["train", "test"]:
            counts = {k: len(v) for k, v in mapped[split_name].items()}
            print(f"  CCPD2020 {split_name}: {counts}")

        return mapped
    else:
        return {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}


# ── CRPD ──────────────────────────────────────────────────────────────
def build_crpd(output_dir: Path, link_mode: str = "symlink"):
    """
    Process CRPD dataset.

    class 0 -> crpd_blue
    class 1 -> crpd_yellow
    class 2,3 -> crpd_multi_complex

    Returns dict: {split: {slice_name: [(img_path, quad, text), ...]}}
    """
    CRPD_CLASS_MAP = {0: "crpd_blue", 1: "crpd_yellow", 2: "crpd_multi_complex", 3: "crpd_multi_complex"}
    results = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

    for subset in ["single", "double", "multi"]:
        subset_dir = CRPD_ROOT / f"CRPD_{subset}"
        if not subset_dir.exists():
            continue

        for split_name in ["train", "val", "test"]:
            img_dir = subset_dir / split_name / "images"
            label_dir = subset_dir / split_name / "labels"
            if not img_dir.exists() or not label_dir.exists():
                continue

            for label_file in sorted(os.listdir(label_dir)):
                if not label_file.endswith(".txt"):
                    continue
                label_path = os.path.join(label_dir, label_file)
                img_name = label_file.replace(".txt", ".jpg")
                img_path = os.path.join(img_dir, img_name)

                if not os.path.isfile(img_path):
                    continue

                # Parse image size
                try:
                    w, h = get_image_size(img_path)
                except Exception:
                    continue

                lines = open(label_path).read().strip().splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 10:
                        continue
                    coords = list(map(float, parts[:8]))
                    cls_id = int(parts[8])
                    text = " ".join(parts[9:]) if len(parts) > 9 else ""

                    slice_name = CRPD_CLASS_MAP.get(cls_id, "crpd_multi_complex")
                    quad = np.array(coords, dtype=np.float32).reshape(4, 2)

                    results[split_name][slice_name].append((img_path, quad, text))

        # Report per-subset counts
        for split_name in ["train", "val", "test"]:
            if results[split_name]:
                total = sum(len(v) for v in results[split_name].values())
                if total > 0:
                    print(f"  CRPD_{subset} {split_name}: {total} labels")

    return results


# ── Write labels and image lists ──────────────────────────────────────
def write_labels_and_lists(output_dir: Path, all_data: dict, link_mode: str):
    """
    Write YOLO pose labels and create per-slice image lists.

    all_data: {dataset_key: {split: {slice_name: [(img_path, quad, text), ...]}}}

    Also writes image_sources.tsv:
        new_filename<tab>original_absolute_path
    This mapping lets the cloud setup script recreate symlinks without having
    to search source directories by the renamed filenames.
    """
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    lists_dir = output_dir / "lists"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    lists_dir.mkdir(parents=True, exist_ok=True)

    # Flatten per-slice per-split
    slice_images = defaultdict(lambda: defaultdict(list))  # {split: {slice: [img_path]}}

    # Source mapping: new_img_name -> original_abs_path
    source_map = {}

    # Assign unique short ID per image to avoid name collisions
    img_counter = 0

    for dataset_key, split_data in all_data.items():
        for split_name, slice_dict in split_data.items():
            for slice_name, records in slice_dict.items():
                for record in records:
                    # Handle both (img_path, quad) and (img_path, quad, text) formats
                    if len(record) == 2:
                        img_path, quad = record
                        text = ""
                    else:
                        img_path, quad, text = record
                    # Get image size
                    try:
                        w, h = get_image_size(img_path)
                    except Exception:
                        continue

                    # Unique filename
                    img_counter += 1
                    short_id = f"{SLICE_TAGS.get(slice_name, 'unk')}_{img_counter:08d}"
                    src_ext = Path(img_path).suffix
                    dst_img_name = f"{short_id}{src_ext}"
                    dst_label_name = f"{short_id}.txt"

                    # Record source mapping (for cloud portability)
                    source_map[f"{split_name}/{dst_img_name}"] = str(Path(img_path).resolve())

                    # Symlink/copy image
                    dst_img = images_dir / split_name / dst_img_name
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    make_link(img_path, str(dst_img), link_mode)

                    # Write label
                    dst_label = labels_dir / split_name / dst_label_name
                    dst_label.parent.mkdir(parents=True, exist_ok=True)
                    label_line = quad_to_yolo_pose_line(quad, w, h)
                    dst_label.write_text(label_line + "\n")

                    slice_images[split_name][slice_name].append(str(dst_img))

    # Write per-slice lists
    for split_name in slice_images:
        for slice_name, paths in slice_images[split_name].items():
            list_file = lists_dir / f"{split_name}_{slice_name}.txt"
            list_file.write_text("\n".join(sorted(paths)) + "\n")

    # Write image_sources.tsv for cloud portability
    tsv_path = output_dir / "image_sources.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("rel_path\tsource_path\n")  # header
        for new_key in sorted(source_map):
            f.write(f"{new_key}\t{source_map[new_key]}\n")
    n_sources = len(source_map)
    print(f"  Written: {tsv_path} ({n_sources} mappings)")

    # Report stats
    for split_name in ["train", "val", "test"]:
        if split_name in slice_images:
            total = sum(len(v) for v in slice_images[split_name].values())
            per_slice = {k: len(v) for k, v in slice_images[split_name].items()}
            print(f"  {split_name}: {total} total {per_slice}")

    return slice_images


def make_link(src: str, dst: str, mode: str):
    """Create a symlink or copy."""
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        rel = os.path.relpath(src, os.path.dirname(dst))
        os.symlink(rel, dst)


# ── Build weighted train.txt ──────────────────────────────────────────
def build_weighted_train_list(slice_images: dict, output_dir: Path,
                              epoch_target: int = 100000):
    """
    Build a single train.txt with weighted sampling to match SLICE_RATIOS.
    Uses paths relative to output_dir for portability.
    """
    train_slices = slice_images.get("train", {})
    if not train_slices:
        print("  [WARN] No training images found!")
        return

    # Compute total available per slice
    available = {slice_name: len(paths) for slice_name, paths in train_slices.items()}

    print("\n  Weighted sampling plan (target per epoch = {}):".format(epoch_target))
    print(f"  {'Slice':<25} {'Avail':>8} {'Target%':>8} {'TargetN':>8} {'Repeats':>8}")

    selected_abs = []
    for slice_name in sorted(SLICE_RATIOS.keys()):
        ratio = SLICE_RATIOS[slice_name]
        target_n = int(epoch_target * ratio)
        avail = available.get(slice_name, 0)
        if avail == 0:
            print(f"  {slice_name:<25} {0:>8} {ratio*100:>7.1f}% {target_n:>8} {'N/A':>8} (NO DATA)")
            continue
        repeats = math.ceil(target_n / avail) if avail > 0 else 0
        pool = train_slices[slice_name]
        sampled = (pool * repeats)[:target_n]
        selected_abs.extend(sampled)
        print(f"  {slice_name:<25} {avail:>8} {ratio*100:>7.1f}% {target_n:>8} {repeats:>8}")

    # Shuffle
    random.shuffle(selected_abs)

    # Write with relative paths
    train_rel = []
    for abs_path in selected_abs:
        rel = os.path.relpath(abs_path, output_dir)
        train_rel.append(rel)

    train_txt = output_dir / "train.txt"
    train_txt.write_text("\n".join(train_rel) + "\n")
    print(f"\n  Written: {train_txt} ({len(train_rel)} images, relative paths)")


def build_val_test_lists(slice_images: dict, output_dir: Path):
    """Write val.txt and test.txt with relative paths. Cap sizes for practical use."""
    for split_name, max_count in [("val", 3000), ("test", 10000)]:
        if split_name not in slice_images:
            continue
        all_paths = []
        for paths in slice_images[split_name].values():
            all_paths.extend(paths)
        if not all_paths:
            continue
        random.shuffle(all_paths)
        capped = all_paths[:max_count]
        rel_lines = [os.path.relpath(p, output_dir) for p in capped]
        out_file = output_dir / f"{split_name}.txt"
        out_file.write_text("\n".join(rel_lines) + "\n")
        print(f"  Written: {out_file} ({len(rel_lines)} images, capped to {max_count})")


# ── Write dataset.yaml ────────────────────────────────────────────────
def write_dataset_yaml(output_dir: Path):
    """Write Ultralytics-compatible dataset YAML with portable path."""
    yaml_path = output_dir / "dataset.yaml"
    # Use '.' so paths are relative to the YAML itself — portable across machines
    content = """# True-Quad Pose Dataset
# Generated by build_true_quad_pose_dataset.py

path: .
train: train.txt
val: val.txt

nc: 1
kpt_shape: [4, 3]
names:
  0: plate
"""
    yaml_path.write_text(content)
    print(f"  Written: {yaml_path}")
    return yaml_path


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build YOLOv8-pose dataset for true-quad plate detection.")
    parser.add_argument("--output-dir", type=str,
                        default=str(LPRNET_ROOT / "datasets" / "plate_true_quad_pose"),
                        help="Output dataset directory")
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "copy"],
                        help="How to store images in output dir")
    parser.add_argument("--epoch-target", type=int, default=100000,
                        help="Target number of training images per epoch for weighted sampling")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Building True-Quad Pose Dataset")
    print("=" * 70)

    # Step 1: Parse all datasets
    print("\n[1/4] Parsing CCPD2019...")
    ccpd2019 = build_ccpd2019(output_dir, args.link_mode)

    print("\n[2/4] Parsing CCPD2020 green...")
    ccpd2020 = build_ccpd2020(output_dir, args.link_mode)

    print("\n[3/4] Parsing CRPD...")
    crpd = build_crpd(output_dir, args.link_mode)

    # Combine
    all_data = {
        "ccpd2019": ccpd2019,
        "ccpd2020": ccpd2020,
        "crpd": crpd,
    }

    # Step 4: Write labels, symlinks, and per-slice lists
    print("\n[4/4] Writing labels and dataset structure...")
    slice_images = write_labels_and_lists(output_dir, all_data, args.link_mode)

    # Build weighted train.txt
    build_weighted_train_list(slice_images, output_dir, args.epoch_target)

    # Build val.txt and test.txt
    build_val_test_lists(slice_images, output_dir)

    # Write dataset.yaml
    write_dataset_yaml(output_dir)

    # Summary
    total_labels = sum(
        len(v) for split_data in all_data.values()
        for split_dict in split_data.values()
        for v in split_dict.values()
    )
    print(f"\n{'=' * 70}")
    print(f"DONE. Total labels processed: {total_labels}")
    print(f"Output: {output_dir.resolve()}")
    print(f"  dataset.yaml: {output_dir / 'dataset.yaml'}")
    print(f"  train.txt: {output_dir / 'train.txt'}")
    print(f"  val.txt: {output_dir / 'val.txt'}")
    print(f"  test.txt: {output_dir / 'test.txt'}")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
