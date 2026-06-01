#!/usr/bin/env python3
"""Generate comprehensive QA sheets for special_split_v2 dataset."""
import csv
import random
from pathlib import Path
import cv2
import numpy as np

# Config
DATE_TAG = "20260601"
MANIFEST_DIR = Path(f"/home/wzzz/LPRNet/scripts/special_gen/manifests_rebased/special_split_v2_{DATE_TAG}")
DATASET_DIR = Path(f"/home/wzzz/LPRNet/scripts/special_gen/datasets/special_ccpd2019_base_cvreplace_v2_{DATE_TAG}")
QA_DIR = DATASET_DIR / "qa_v2"
PROJECT_ROOT = Path("/home/wzzz/LPRNet")
rng = random.Random(20260602)

QA_DIR.mkdir(parents=True, exist_ok=True)

def make_pose_crop(image_bgr, quad, out_w=94, out_h=24):
    src = np.asarray(quad, dtype=np.float32)
    dst = np.float32([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def read_quad_from_row(row):
    """Extract quad from manifest row."""
    return [[float(row["quad_1x"]), float(row["quad_1y"])],
            [float(row["quad_2x"]), float(row["quad_2y"])],
            [float(row["quad_3x"]), float(row["quad_3y"])],
            [float(row["quad_4x"]), float(row["quad_4y"])]]

def build_qa_grid(rows, title, n_samples=36, cols=6):
    """Build QA grid with OCR crops and labels."""
    selected = rng.sample(rows, min(n_samples, len(rows)))
    n = len(selected)
    cols = min(cols, n)
    rows_grid = (n + cols - 1) // cols
    
    tile_w, tile_h = 188, 48  # 2x upscaled 94x24
    label_h = 40
    gap = 3
    margin = 10
    
    canvas_w = cols * tile_w + (cols - 1) * gap + margin * 2
    canvas_h = margin + 30 + rows_grid * (tile_h + label_h + gap)
    canvas = np.full((int(canvas_h), int(canvas_w), 3), 35, dtype=np.uint8)
    
    cv2.putText(canvas, title, (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    
    for idx, row in enumerate(selected):
        if idx >= n:
            break
        r = idx // cols
        c = idx % cols
        x0 = margin + c * (tile_w + gap)
        y0 = int(margin + 30 + r * (tile_h + label_h + gap))
        
        # Load image and crop
        img_path = row["img_path"]
        if not Path(img_path).is_absolute():
            img_path = str(PROJECT_ROOT / img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        quad = read_quad_from_row(row)
        ocrin = make_pose_crop(img, quad)
        tile = cv2.resize(ocrin, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        canvas[y0:y0+tile_h, x0:x0+tile_w] = tile
        
        # Label
        text = row["text"]
        cv2.putText(canvas, text, (x0, y0 + tile_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 200, 180), 1)
        family = row["family"]
        split = row["split"]
        cv2.putText(canvas, f"{family}/{split}", (x0, y0 + tile_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)
    
    return canvas

def build_clean_vs_hard_grid(clean_rows, hard_rows, title, n_pairs=20):
    """Side-by-side comparison of clean vs hard for same-family images."""
    selected = rng.sample(list(zip(clean_rows, hard_rows)), min(n_pairs, len(clean_rows), len(hard_rows)))
    n = len(selected)
    cols = 2
    rows_grid = n
    
    tile_w, tile_h = 188, 48
    label_h = 20
    gap = 3
    margin = 10
    header_h = 50
    
    canvas_w = margin*2 + cols*(tile_w + gap)
    canvas_h = header_h + rows_grid * (tile_h + label_h + gap)
    canvas = np.full((int(canvas_h), int(canvas_w), 3), 35, dtype=np.uint8)
    
    cv2.putText(canvas, title + "  [CLEAN left | HARD right]", (margin, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, f"Each row: same text type, clean vs hard-degraded", (margin, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    
    for idx, (clean_row, hard_row) in enumerate(selected):
        y0 = int(header_h + idx * (tile_h + label_h + gap))
        
        for col, row in enumerate([clean_row, hard_row]):
            x0 = margin + col * (tile_w + gap)
            img_path = row["img_path"]
            if not Path(img_path).is_absolute():
                img_path = str(PROJECT_ROOT / img_path)
            img = cv2.imread(img_path)
            if img is None:
                continue
            quad = read_quad_from_row(row)
            ocrin = make_pose_crop(img, quad)
            tile = cv2.resize(ocrin, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
            canvas[y0:y0+tile_h, x0:x0+tile_w] = tile
            
            # Short label
            text = row["text"][:10]
            cv2.putText(canvas, text, (x0, y0 + tile_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)
    
    return canvas

# Load manifests
manifests = {
    "police": {},
    "embassy": {},
}

for family in ["police", "embassy"]:
    for split in ["train", "val_clean", "val_hard"]:
        mf = MANIFEST_DIR / f"{split}_{family}.csv"
        with mf.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifests[family][split] = list(reader)
        print(f"[{family:8s}] [{split:10s}] {len(manifests[family][split]):6d} rows")

# Generate QA grids
print("\nGenerating QA grids (36 samples each)...")
for family in ["police", "embassy"]:
    for split in ["train", "val_clean", "val_hard"]:
        rows = manifests[family][split]
        canvas = build_qa_grid(rows, f"special_v2_{DATE_TAG} {family} {split}", n_samples=36)
        qa_path = QA_DIR / f"qa_{split}_{family}_36.jpg"
        cv2.imwrite(str(qa_path), canvas)
        print(f"  {qa_path}")

# Generate clean vs hard comparison
print("\nGenerating clean vs hard comparison (20 pairs each)...")
for family in ["police", "embassy"]:
    clean_rows = manifests[family]["val_clean"]
    hard_rows = manifests[family]["val_hard"]
    canvas = build_clean_vs_hard_grid(clean_rows, hard_rows, f"special_v2_{DATE_TAG} {family} clean_vs_hard")
    qa_path = QA_DIR / f"qa_clean_vs_hard_{family}.jpg"
    cv2.imwrite(str(qa_path), canvas)
    print(f"  {qa_path}")

print(f"\nAll QA sheets in: {QA_DIR}")
print("Done.")
