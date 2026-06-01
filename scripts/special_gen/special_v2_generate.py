#!/usr/bin/env python3
"""
Special plate v2 data generation and audit.
- Police: train 500/province, val_clean 50/province, val_hard 50/province
- Embassy: train 10,000, val_clean 1,000, val_hard 1,000
- No base image overlap across splits
- val_hard applies board-like degradation
Output: datasets/ and manifests_rebased/ under scripts/special_gen/
"""
import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
import sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lib_special_plate_renderer import SpecialPlateRenderer
from plate_number import provinces

# ── Constants ──
DEFAULT_POSE_JSONL = Path("/home/wzzz/LPRNet/datasets/ccpd2019_base_posquads_20260509/pose_quads.jsonl")
ALL_PROVINCES = provinces          # 31 provinces
OUTPUT_FIELDS = [
    "img_path", "text", "family", "source", "split",
    "preprocess_group", "has_quad", "can_parse_ccpd_geom",
    "can_perspective", "quad_source", "bbox_source",
    "quad_1x", "quad_1y", "quad_2x", "quad_2y",
    "quad_3x", "quad_3y", "quad_4x", "quad_4y",
    "ocr_crop_mode", "ocr_resize_mode", "ocr_resize_kernel",
    "ocr_preproc", "ocr_channel_order", "ocr_quad_pad_ratio",
]

# Base image pool sizes needed:
# Police: train(15,500) + val_clean(1,550) + val_hard(1,550) = 18,600
# Embassy: train(10,000) + val_clean(1,000) + val_hard(1,000) = 12,000
# Total unique base images: 18,600 (police) + 12,000 (embassy) = 30,600
# But we can reuse the same base across families for train, just not across splits
# Since police and embassy are different families, they CAN share base images
# Only constraint: train / val_clean / val_hard MUST have different base images
# So total unique bases needed: max(train,val_clean,val_hard across families)
# = max(25,500, 2,550, 2,550) = 25,500
# Available: 199,996 — plenty

# ── Data volume targets ──
TARGETS = {
    "police": {
        "train": 500,
        "val_clean": 50,
        "val_hard": 50,
    },
    "embassy": {
        "train": 10000,
        "val_clean": 1000,
        "val_hard": 1000,
    },
}

# ── Helpers ──

def parse_args():
    ap = argparse.ArgumentParser(description="Generate special plate v2 dataset with splits.")
    ap.add_argument("--pose-jsonl", type=Path, default=DEFAULT_POSE_JSONL)
    ap.add_argument("--date-tag", default=None,
                    help="Date tag for output dirs (default: auto)")
    ap.add_argument("--skip-generate", action="store_true",
                    help="Skip image generation, only produce manifests/audit")
    ap.add_argument("--val-source-count", type=int, default=3000,
                    help="Base images reserved for val splits")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: tiny counts")
    ap.add_argument("--seed", type=int, default=20260601)
    return ap.parse_args()


def ensure_dirs(*paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def cv_imwrite(path, image, quality=95):
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    path.write_bytes(buf.tobytes())


def estimate_sharpness(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_noise(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_std = cv2.boxFilter(gray, -1, (7, 7), normalize=False)
    local_mean = cv2.boxFilter(gray, -1, (7, 7), normalize=True)
    local_var = np.clip(local_std - local_mean * local_mean * 49, 0, None)
    return float(np.median(np.sqrt(local_var)))


# ── Style transfer (from v1, unchanged) ──

def transfer_style_l_only(rendered_bgr, source_patch_bgr):
    h, w = rendered_bgr.shape[:2]
    if source_patch_bgr.shape[:2] != (h, w):
        source_patch_bgr = cv2.resize(source_patch_bgr, (w, h), interpolation=cv2.INTER_AREA)
    rendered_lab = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source_patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_rendered = rendered_lab[:, :, 0]
    l_source = source_lab[:, :, 0]
    mean_r, std_r = l_rendered.mean(), l_rendered.std() + 1e-6
    mean_s, std_s = l_source.mean(), l_source.std() + 1e-6
    ratio = max(0.3, min(3.0, std_s / std_r))
    rendered_lab[:, :, 0] = np.clip((l_rendered - mean_r) * ratio + mean_s, 0, 255)
    result = cv2.cvtColor(rendered_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    source_hsv = cv2.cvtColor(source_patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    src_v_low = cv2.resize(
        cv2.resize(source_hsv[:, :, 2], (8, 4), interpolation=cv2.INTER_AREA),
        (w, h), interpolation=cv2.INTER_LINEAR,
    )
    res_v_low = cv2.resize(
        cv2.resize(result_hsv[:, :, 2], (8, 4), interpolation=cv2.INTER_AREA),
        (w, h), interpolation=cv2.INTER_LINEAR,
    )
    result_hsv[:, :, 2] = np.clip(result_hsv[:, :, 2] + (src_v_low - res_v_low) * 0.5, 0, 255)
    result = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return apply_capture_finish(result, source_patch_bgr)


def apply_capture_finish(rendered_bgr, source_patch_bgr):
    h, w = rendered_bgr.shape[:2]
    if source_patch_bgr.shape[:2] != (h, w):
        source_patch_bgr = cv2.resize(source_patch_bgr, (w, h), interpolation=cv2.INTER_AREA)
    result = rendered_bgr.copy()
    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    source_hsv = cv2.cvtColor(source_patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    src_v_low = cv2.resize(
        cv2.resize(source_hsv[:, :, 2], (8, 4), interpolation=cv2.INTER_AREA),
        (w, h), interpolation=cv2.INTER_LINEAR,
    )
    res_v_low = cv2.resize(
        cv2.resize(result_hsv[:, :, 2], (8, 4), interpolation=cv2.INTER_AREA),
        (w, h), interpolation=cv2.INTER_LINEAR,
    )
    result_hsv[:, :, 2] = np.clip(result_hsv[:, :, 2] + (src_v_low - res_v_low) * 0.35, 0, 255)
    result = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    sharp_result = estimate_sharpness(result)
    sharp_source = estimate_sharpness(source_patch_bgr)
    if sharp_result > sharp_source * 1.25:
        ksize = 3
        while ksize <= 11 and estimate_sharpness(result) > sharp_source * 1.10:
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)
            ksize += 2
    noise_source = estimate_noise(source_patch_bgr)
    noise_result = estimate_noise(result)
    if noise_source > noise_result * 1.05:
        noise_amt = min(max(noise_source - noise_result, 1.0), 18.0)
        noise_map = np.random.randn(h, w, 3).astype(np.float32) * noise_amt
        result = np.clip(result.astype(np.float32) + noise_map, 0, 255).astype(np.uint8)
    quality = 88 if sharp_source > 60 else 82
    ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        result = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return result


# ── Color guard (from v1, unchanged) ──

def color_guard_metrics(image_bgr, family):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    total = float(image_bgr.shape[0] * image_bgr.shape[1])
    yellow_ratio = float(((s > 40) & (v > 60) & (h >= 10) & (h <= 40)).sum()) / total
    white_ratio = float(((s < 40) & (v > 150)).sum()) / total
    dark_ratio = float((v < 80).sum()) / total
    red_ratio = float((((h <= 10) | (h >= 170)) & (s > 80) & (v > 80)).sum()) / total
    return {
        "yellow_ratio": yellow_ratio, "white_ratio": white_ratio,
        "dark_ratio": dark_ratio, "red_ratio": red_ratio,
        "ok": family_guard_ok(family, yellow_ratio, white_ratio, dark_ratio, red_ratio),
    }


def family_guard_ok(family, yellow_ratio, white_ratio, dark_ratio, red_ratio):
    if family == "police":
        return white_ratio >= 0.18 and red_ratio >= 0.003
    if family == "embassy":
        return dark_ratio >= 0.25 and red_ratio >= 0.003
    return False


# ── Pose crop ──

def make_pose_crop(image_bgr, pose_quad, out_w=94, out_h=24):
    src = np.asarray(pose_quad, dtype=np.float32)
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image_bgr, matrix, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


# ── Degradation pipeline for val_hard ──

def apply_hard_degradation(bgr_img, rng):
    """Apply board-like degradations to simulate difficult capture conditions."""
    result = bgr_img.copy()
    h, w = result.shape[:2]

    # 1. Gaussian blur (mild)
    if rng.random() < 0.7:
        ksize = rng.choice([3, 5])
        sigma = rng.uniform(0.3, 1.0)
        result = cv2.GaussianBlur(result, (ksize, ksize), sigma)

    # 2. JPEG compression
    if rng.random() < 0.8:
        quality = rng.randint(60, 90)
        ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            result = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # 3. Brightness/contrast adjustment
    if rng.random() < 0.7:
        brightness = rng.uniform(-40, 40)    # ±40
        contrast = rng.uniform(0.7, 1.3)     # 0.7x-1.3x
        result = cv2.convertScaleAbs(result, alpha=contrast, beta=brightness)

    # 4. Local exposure variation (vignette-like)
    if rng.random() < 0.4:
        # Create a random gradient mask
        mask = np.ones((h, w), dtype=np.float32) * rng.uniform(0.7, 1.0)
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        dist = dist / max(h, w) * rng.uniform(1.0, 2.5)
        mask = np.clip(mask - dist * 0.15, 0.6, 1.0)
        result = (result.astype(np.float32) * mask[:, :, None]).clip(0, 255).astype(np.uint8)

    # 5. Noise (shot noise)
    if rng.random() < 0.5:
        noise = np.random.randn(h, w, 3).astype(np.float32) * rng.uniform(2, 8)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 6. Slight color shift
    if rng.random() < 0.3:
        result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        result_hsv[:, :, 0] = (result_hsv[:, :, 0] + rng.uniform(-10, 10)) % 180
        result = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


def apply_quad_jitter(pose_quad, rng, max_px=4):
    """Apply small random perturbation to quad points to simulate detection jitter."""
    jittered = np.array(pose_quad, dtype=np.float32).copy()
    for i in range(4):
        jittered[i, 0] += rng.uniform(-max_px, max_px)
        jittered[i, 1] += rng.uniform(-max_px, max_px)
    return jittered


def warped_quad_crop(image_bgr, pose_quad, out_w=94, out_h=24):
    """Standard pose-based warp crop (same as training preprocessing)."""
    src = np.asarray(pose_quad, dtype=np.float32)
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image_bgr, matrix, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


# ── Data loading and splitting ──

def load_pose_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def make_manifest_row(out_path, text, family, split_name, pose_quad, source_tag):
    return {
        "img_path": str(out_path),
        "text": text,
        "family": family,
        "source": f"special_ccpd2019_base_cvreplace_v2__{family}",
        "split": split_name,
        "preprocess_group": "ccpd_board",
        "has_quad": "1",
        "can_parse_ccpd_geom": "0",
        "can_perspective": "1",
        "quad_source": "pose_yolov8n_ccpd2019_base",
        "bbox_source": "pose_yolov8n_ccpd2019_base",
        "quad_1x": f"{pose_quad[0][0]:.1f}",
        "quad_1y": f"{pose_quad[0][1]:.1f}",
        "quad_2x": f"{pose_quad[1][0]:.1f}",
        "quad_2y": f"{pose_quad[1][1]:.1f}",
        "quad_3x": f"{pose_quad[2][0]:.1f}",
        "quad_3y": f"{pose_quad[2][1]:.1f}",
        "quad_4x": f"{pose_quad[3][0]:.1f}",
        "quad_4y": f"{pose_quad[3][1]:.1f}",
        "ocr_crop_mode": "obb_warp",
        "ocr_resize_mode": "letterbox",
        "ocr_resize_kernel": "nn",
        "ocr_preproc": "none",
        "ocr_channel_order": "bgr",
        "ocr_quad_pad_ratio": "0.0",
    }


def build_schedule_from_pool(pool, count, rng):
    """Draw `count` items from pool with cycling if needed, without replacement per call."""
    if not pool or count <= 0:
        return []
    # Shuffle a copy and take first `count` (allowing cycling for large counts)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    cycles = math.ceil(count / len(shuffled))
    schedule = []
    for _ in range(cycles):
        rng.shuffle(shuffled)
        schedule.extend(shuffled)
    return schedule[:count]


def split_pools(pose_rows, val_count, rng):
    """Split pose rows into train, val_clean, val_hard pools (mutually exclusive)."""
    rows = list(pose_rows)
    rng.shuffle(rows)
    # Reserve val_clean pools (3-way: we need val_clean and val_hard to each have their own)
    # Actually, we need 3 non-overlapping pools: train, val_clean, val_hard
    # We'll split: first val_count → val_clean, next val_count → val_hard, rest → train
    total_val = val_count * 2  # need val_count for clean + val_count for hard
    val_pool = rows[:total_val]
    train_pool = rows[total_val:]
    val_clean_pool = val_pool[:val_count]
    val_hard_pool = val_pool[val_count:]
    return train_pool, val_clean_pool, val_hard_pool


# ── Main generation ──

def generate_family_split(
    split_name, pool, family, work_items, image_dir, renderer,
    skip_generate, rng_seed, apply_degrade=False
):
    """Generate images for one family × split combination."""
    rng = random.Random(rng_seed)
    schedule = build_schedule_from_pool(pool, len(work_items), rng)
    rows = []
    counts = Counter()
    skip_counts = Counter()
    guard_stats = []
    src_rect = None

    for src_rec, target_text in zip(schedule, work_items):
        img_path = src_rec["img_path"]
        img = cv2.imread(img_path)
        if img is None:
            skip_counts["read_fail"] += 1
            continue

        gt_quad_raw = np.array(src_rec["gt_quad"], dtype=np.float32)
        gt_quad = np.array([gt_quad_raw[2], gt_quad_raw[3], gt_quad_raw[0], gt_quad_raw[1]], dtype=np.float32)
        pose_quad = np.array(src_rec["pose_quad"], dtype=np.float32)
        text = target_text
        rendered = renderer.render_plate(text, family)
        plate_h, plate_w = rendered.image_bgr.shape[:2]
        if src_rect is None or src_rect.shape[0] != 4 or src_rect[2][0] != plate_w - 1 or src_rect[2][1] != plate_h - 1:
            src_rect = np.float32([[0, 0], [plate_w - 1, 0], [plate_w - 1, plate_h - 1], [0, plate_h - 1]])

        try:
            h, w = img.shape[:2]
            forward = cv2.getPerspectiveTransform(gt_quad.astype(np.float32), src_rect)
            source_patch = cv2.warpPerspective(
                img, forward, (plate_w, plate_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            transferred = transfer_style_l_only(rendered.image_bgr, source_patch)
            metrics = color_guard_metrics(transferred, family)
            if not metrics["ok"]:
                raw_metrics = color_guard_metrics(rendered.image_bgr, family)
                if raw_metrics["ok"]:
                    transferred = apply_capture_finish(rendered.image_bgr, source_patch)
                    metrics = raw_metrics
                    skip_counts["color_guard_fallback_raw"] += 1
                else:
                    skip_counts["color_guard"] += 1
                    guard_stats.append(metrics)
                    continue
            guard_stats.append(metrics)

            inverse = cv2.getPerspectiveTransform(src_rect, gt_quad.astype(np.float32))
            warped = cv2.warpPerspective(
                transferred, inverse, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [gt_quad.round().astype(np.int32)], 255)
            mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
            result = (
                img.astype(np.float32) * (1.0 - mask[:, :, None])
                + warped.astype(np.float32) * mask[:, :, None]
            ).clip(0, 255).astype(np.uint8)

            # For val_hard, add degradations
            if apply_degrade:
                result = apply_hard_degradation(result, rng)

            stem = Path(img_path).stem
            family_tag = {"police": "pol", "embassy": "emb"}[family]
            out_path = image_dir / f"{stem}_{family_tag}_{text}.jpg"
            if not skip_generate:
                cv_imwrite(out_path, result)
            rows.append(make_manifest_row(out_path, text, family, split_name, pose_quad, family))
            counts[family] += 1

        except Exception as e:
            skip_counts["exception"] += 1
            continue

    return rows, counts, skip_counts, guard_stats


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_qa_sheet(samples, qa_path, title="QA"):
    """Build a QA preview grid image."""
    if not samples:
        return
    n = len(samples)
    cols = 5
    rows_n = (n + cols - 1) // cols
    tile_w, tile_h = 188, 48   # downsized 94x24 * 2
    gap = 4
    canvas_w = cols * tile_w + (cols - 1) * gap + 20
    canvas_h = rows_n * (tile_h + 30) + (rows_n - 1) * gap + 40
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    for idx, (ocrin, text, family, split) in enumerate(samples):
        if idx >= n:
            break
        r = idx // cols
        c = idx % cols
        x0 = 10 + c * (tile_w + gap)
        y0 = 40 + r * (tile_h + 30 + gap)

        # Resize OCR crop to tile
        tile = cv2.resize(ocrin, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

        # Label below
        label = f"{family}:{text}"
        cv2.putText(canvas, label, (x0, y0 + tile_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(canvas, split, (x0, y0 + tile_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

    cv_imwrite(qa_path, canvas, 92)


# ── Audit ──

def audit_manifests(manifest_dir, dataset_dir, date_tag):
    """Analyze generated manifests and produce summary stats."""
    print("\n" + "=" * 60)
    print(f"AUDIT REPORT — special_split_v2_{date_tag}")
    print("=" * 60)

    all_manifests = sorted(manifest_dir.glob("*.csv"))
    print(f"\nManifests found: {len(all_manifests)}")
    for m in all_manifests:
        size = m.stat().st_size
        print(f"  {m.name}: {size:,} bytes")

    # Load all manifests and aggregate stats
    all_rows = {}  # split_name -> list of rows
    all_base_ids = {}  # split_name -> set of base image ids

    for mf_path in all_manifests:
        with mf_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split_key = row["split"]
                if split_key not in all_rows:
                    all_rows[split_key] = []
                    all_base_ids[split_key] = set()
                all_rows[split_key].append(row)
                # Extract base image id from img_path
                stem = Path(row["img_path"]).stem
                # The base id is the CCPD2019-style prefix (before _pol_ or _emb_)
                base_id = stem.rsplit("_", 2)[0]
                all_base_ids[split_key].add(base_id)

    # Per-split counts
    print("\n--- Per-Split Counts ---")
    for split_name in sorted(all_rows.keys()):
        rows = all_rows[split_name]
        families = Counter(r["family"] for r in rows)
        print(f"  {split_name}: {len(rows)} total")
        for fam, cnt in sorted(families.items()):
            print(f"    {fam}: {cnt}")

    # Base image overlap check
    print("\n--- Base Image Overlap Check ---")
    split_names = sorted(all_base_ids.keys())
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i + 1:]:
            overlap = all_base_ids[s1] & all_base_ids[s2]
            ratio = len(overlap) / max(len(all_base_ids[s1] | all_base_ids[s2]), 1) * 100
            status = "❌ OVERLAP" if overlap else "✅ clean"
            print(f"  {s1} vs {s2}: {len(overlap)} shared bases ({ratio:.2f}%) {status}")

    # Police province balance
    print("\n--- Police Province Balance ---")
    police_rows = [r for s in all_rows.values() for r in s if r["family"] == "police"]
    prov_counts = Counter()
    for r in police_rows:
        # Extract province (first character of text)
        text = r["text"]
        if text:
            prov = text[0]
            prov_counts[(r["split"], prov)] += 1

    for split_name in sorted(all_rows.keys()):
        print(f"  {split_name}:")
        split_provs = [(p, c) for (s, p), c in prov_counts.items() if s == split_name]
        split_provs.sort(key=lambda x: x[0])
        counts = [c for _, c in split_provs]
        if counts:
            print(f"    min={min(counts)}, max={max(counts)}, mean={sum(counts)/len(counts):.1f}")
            # Show full table only if not too many
            for p, c in split_provs:
                print(f"    {p}: {c}")

    # Embassy digit distribution
    print("\n--- Embassy Digit Distribution ---")
    emb_rows = [r for s in all_rows.values() for r in s if r["family"] == "embassy"]
    digit_counts = Counter()
    for r in emb_rows:
        text = r["text"]
        for ch in text:
            if ch.isdigit():
                digit_counts[(r["split"], ch)] += 1
    for split_name in sorted(all_rows.keys()):
        split_digits = [(d, c) for (s, d), c in digit_counts.items() if s == split_name]
        split_digits.sort(key=lambda x: x[0])
        min_c = min(c for _, c in split_digits) if split_digits else 0
        max_c = max(c for _, c in split_digits) if split_digits else 0
        total = sum(c for _, c in split_digits)
        if total > 0:
            print(f"  {split_name}: total={total}, min={min_c}, max={max_c}, mean={total/len(split_digits):.1f}")

    # Text format validation
    print("\n--- Text Format Check ---")
    format_errors = Counter()
    for r in police_rows:
        text = r["text"]
        if not (len(text) == 7 and text[0] in ALL_PROVINCES and text[-1] == "警"):
            format_errors[f"police_bad_format: {text}"] += 1
    for r in emb_rows:
        text = r["text"]
        if not (len(text) == 7 and text[0] == "使" and text[1:].isdigit()):
            format_errors[f"embassy_bad_format: {text}"] += 1
    if format_errors:
        print(f"  FORMAT ERRORS: {len(format_errors)}")
        for err, cnt in format_errors.most_common(10):
            print(f"    {err}: {cnt}")
    else:
        print("  All texts match expected format ✅")

    # Source field check
    print("\n--- Source Field Check ---")
    sources = Counter()
    for s in all_rows.values():
        for r in s:
            sources[r["source"]] += 1
    for src, cnt in sorted(sources.items()):
        print(f"  {src}: {cnt}")

    # Missing files check
    print("\n--- Missing Image Files ---")
    missing = 0
    for s in all_rows.values():
        for r in s:
            p = Path(r["img_path"])
            if not p.is_absolute():
                # Check relative to dataset_dir or PROJECT_ROOT
                candidate = dataset_dir.parent.parent / r["img_path"]
                if not candidate.exists():
                    missing += 1
            elif not p.exists():
                missing += 1
    if missing == 0:
        print("  All images present ✅")
    else:
        print(f"  MISSING: {missing} images ❌")

    print("\n" + "=" * 60)
    print("END OF AUDIT REPORT")
    print("=" * 60)


# ── Main ──

def main():
    args = parse_args()
    date_tag = args.date_tag or time.strftime("%Y%m%d")
    master_seed = args.seed
    rng = random.Random(master_seed)

    # Smoke mode: tiny counts for testing
    global TARGETS
    if args.smoke:
        TARGETS = {
            "police": {"train": 5, "val_clean": 2, "val_hard": 2},
            "embassy": {"train": 10, "val_clean": 3, "val_hard": 3},
        }
    output_root = _HERE  # scripts/special_gen/
    dataset_dir = output_root / "datasets" / f"special_ccpd2019_base_cvreplace_v2_{date_tag}"
    manifest_dir = output_root / "manifests_rebased" / f"special_split_v2_{date_tag}"
    image_dirs = {
        "train": dataset_dir / "images" / "train",
        "val_clean": dataset_dir / "images" / "val_clean",
        "val_hard": dataset_dir / "images" / "val_hard",
    }
    qa_dir = dataset_dir / "qa"
    ensure_dirs(dataset_dir, manifest_dir, qa_dir, *image_dirs.values())

    renderer = SpecialPlateRenderer(_HERE)
    pose_rows = load_pose_rows(args.pose_jsonl)
    print(f"Loaded {len(pose_rows)} pose rows from {args.pose_jsonl}")

    # Split pools: train, val_clean, val_hard — mutually exclusive
    train_pool, val_clean_pool, val_hard_pool = split_pools(
        pose_rows, args.val_source_count, rng
    )
    print(f"Pools: train={len(train_pool)}, val_clean={len(val_clean_pool)}, val_hard={len(val_hard_pool)}")

    all_rows = {}       # (family, split) -> rows
    all_skips = {}      # (family, split) -> skip_counts
    all_counts = {}     # (family, split) -> counts
    qa_samples = []     # list of (ocrin_crop, text, family, split)

    total_expected = 0

    for family in ["police", "embassy"]:
        targets = TARGETS[family]
        for split_name, count in targets.items():
            if family == "police":
                per_province = count  # e.g. 500/50/50 per province
                total_count = per_province * len(ALL_PROVINCES)
            else:
                total_count = count

            total_expected += total_count

            # Build work items
            work_items = []
            if family == "police":
                for prov in ALL_PROVINCES:
                    for _ in range(per_province):
                        letter = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
                        middle = "".join(rng.choice("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(4))
                        text = prov + letter + middle + "警"
                        work_items.append(text)
            else:  # embassy
                for _ in range(total_count):
                    digits = "".join(rng.choice("0123456789") for _ in range(6))
                    text = "使" + digits
                    work_items.append(text)

            # Select pool
            pool_map = {"train": train_pool, "val_clean": val_clean_pool, "val_hard": val_hard_pool}
            pool = pool_map[split_name]
            apply_degrade = (split_name == "val_hard")

            rng_seed = hash((master_seed, family, split_name)) & 0x7FFFFFFF
            rows, counts, skips, guard_stats = generate_family_split(
                split_name=split_name,
                pool=pool,
                family=family,
                work_items=work_items,
                image_dir=image_dirs[split_name],
                renderer=renderer,
                skip_generate=args.skip_generate,
                rng_seed=rng_seed,
                apply_degrade=apply_degrade,
            )

            key = (family, split_name)
            all_rows[key] = rows
            all_skips[key] = skips
            all_counts[key] = counts

            # Collect QA samples (take first 15 per category)
            for row in rows[:15]:
                p = Path(row["img_path"])
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None and row["text"]:
                        pose_quad = [
                            [float(row["quad_1x"]), float(row["quad_1y"])],
                            [float(row["quad_2x"]), float(row["quad_2y"])],
                            [float(row["quad_3x"]), float(row["quad_3y"])],
                            [float(row["quad_4x"]), float(row["quad_4y"])],
                        ]
                        ocrin = make_pose_crop(img, np.array(pose_quad, dtype=np.float32))
                        qa_samples.append((ocrin, row["text"], family, split_name))

            gen_count = counts.get(family, 0)
            print(f"  [{family:8s}] [{split_name:10s}] target={total_count:5d} generated={gen_count:5d} "
                  f"skipped={dict(skips)} guard_ok={len([s for s in guard_stats if s['ok']])}/{len(guard_stats)}")

            # Write manifest
            mf_path = manifest_dir / f"{split_name}_{family}.csv"
            write_manifest(mf_path, rows)
            print(f"    Manifest: {mf_path}")

    # Write combined manifests (for convenience)
    print(f"\nExpected total: {total_expected}")
    print(f"Actual generated: {sum(c.get(f, 0) for (f, _), c in all_counts.items() for _ in [1])}")

    # Write summary file
    summary = {
        "date_tag": date_tag,
        "targets": TARGETS,
        "generated": {f"{k[0]}_{k[1]}": dict(v) for k, v in all_counts.items()},
        "skips": {f"{k[0]}_{k[1]}": dict(v) for k, v in all_skips.items()},
        "pool_sizes": {
            "train": len(train_pool),
            "val_clean": len(val_clean_pool),
            "val_hard": len(val_hard_pool),
        },
    }
    summary_path = manifest_dir / "generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary_path}")

    # Build QA sheets
    print("\nGenerating QA sheets...")
    for split_name in ["train", "val_clean", "val_hard"]:
        for family in ["police", "embassy"]:
            samples = [(o, t, f, s) for o, t, f, s in qa_samples if f == family and s == split_name]
            if samples:
                qa_path = qa_dir / f"qa_{split_name}_{family}.jpg"
                title = f"special_v2_{date_tag} {family} {split_name}"
                build_qa_sheet(samples, qa_path, title)
                print(f"  QA: {qa_path} ({len(samples)} samples)")

    # Run audit
    audit_manifests(manifest_dir, dataset_dir, date_tag)
    print("\nDone.")


if __name__ == "__main__":
    main()
