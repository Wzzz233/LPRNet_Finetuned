#!/usr/bin/env python3
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

from lib_special_plate_renderer import SpecialPlateRenderer
from plate_number import provinces


ROOT = Path(__file__).resolve().parent
DEFAULT_POSE_JSONL = Path("/home/wzzz/LPRNet/datasets/ccpd2019_base_posquads_20260509/pose_quads.jsonl")
DATE_TAG = "20260517"
ALL_PROVINCES = provinces
OUTPUT_FIELDS = [
    "img_path",
    "text",
    "family",
    "source",
    "split",
    "preprocess_group",
    "has_quad",
    "can_parse_ccpd_geom",
    "can_perspective",
    "quad_source",
    "bbox_source",
    "quad_1x",
    "quad_1y",
    "quad_2x",
    "quad_2y",
    "quad_3x",
    "quad_3y",
    "quad_4x",
    "quad_4y",
    "ocr_crop_mode",
    "ocr_resize_mode",
    "ocr_resize_kernel",
    "ocr_preproc",
    "ocr_channel_order",
    "ocr_quad_pad_ratio",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Generate special-plate cvreplace data on CCPD2019 base.")
    ap.add_argument("--pose-jsonl", type=Path, default=DEFAULT_POSE_JSONL)
    ap.add_argument("--date-tag", default=DATE_TAG)
    ap.add_argument("--output-root", type=Path, default=ROOT)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-yellow-per-province", type=int, default=200)
    ap.add_argument("--val-yellow-per-province", type=int, default=20)
    ap.add_argument("--train-police-per-province", type=int, default=120)
    ap.add_argument("--val-police-per-province", type=int, default=10)
    ap.add_argument("--train-embassy-total", type=int, default=3000)
    ap.add_argument("--val-embassy-total", type=int, default=300)
    ap.add_argument("--val-source-count", type=int, default=2000)
    return ap.parse_args()


def ensure_dirs(*paths):
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def cv_imwrite(path, image):
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
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
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    res_v_low = cv2.resize(
        cv2.resize(result_hsv[:, :, 2], (8, 4), interpolation=cv2.INTER_AREA),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
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
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    res_v_low = cv2.resize(
        cv2.resize(result_hsv[:, :, 2], (8, 4), interpolation=cv2.INTER_AREA),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
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


def make_pose_crop(image_bgr, pose_quad, out_w=94, out_h=24):
    src = np.asarray(pose_quad, dtype=np.float32)
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image_bgr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def degrade_for_qa(image_bgr):
    blurred = cv2.GaussianBlur(image_bgr, (3, 3), 0.45)
    ok, buf = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 84])
    if ok:
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return blurred


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
        "yellow_ratio": yellow_ratio,
        "white_ratio": white_ratio,
        "dark_ratio": dark_ratio,
        "red_ratio": red_ratio,
        "ok": family_guard_ok(family, yellow_ratio, white_ratio, dark_ratio, red_ratio),
    }


def family_guard_ok(family, yellow_ratio, white_ratio, dark_ratio, red_ratio):
    if family == "yellow_single":
        return yellow_ratio >= 0.20 and red_ratio <= 0.02
    if family == "police":
        return white_ratio >= 0.18 and red_ratio >= 0.003
    if family == "embassy":
        return dark_ratio >= 0.25 and red_ratio >= 0.003
    return False


def load_pose_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def split_source_rows(rows, val_source_count):
    rng = random.Random(20260517)
    rows = list(rows)
    rng.shuffle(rows)
    val_rows = rows[:val_source_count]
    train_rows = rows[val_source_count:]
    return train_rows, val_rows


def build_schedule(pool, count, rng):
    if not pool or count <= 0:
        return []
    shuffled = list(pool)
    rng.shuffle(shuffled)
    cycles = math.ceil(count / len(shuffled))
    schedule = []
    for _ in range(cycles):
        schedule.extend(shuffled)
    return schedule[:count]


def family_assignments(args):
    if args.smoke:
        return {
            "yellow_single": {"train": [(prov, 3) for prov in ALL_PROVINCES], "val": [(prov, 1) for prov in ALL_PROVINCES]},
            "police": {"train": [(prov, 2) for prov in ALL_PROVINCES], "val": [(prov, 1) for prov in ALL_PROVINCES]},
            "embassy": {"train": [(None, 20)], "val": [(None, 10)]},
        }
    return {
        "yellow_single": {
            "train": [(prov, args.train_yellow_per_province) for prov in ALL_PROVINCES],
            "val": [(prov, args.val_yellow_per_province) for prov in ALL_PROVINCES],
        },
        "police": {
            "train": [(prov, args.train_police_per_province) for prov in ALL_PROVINCES],
            "val": [(prov, args.val_police_per_province) for prov in ALL_PROVINCES],
        },
        "embassy": {
            "train": [(None, args.train_embassy_total)],
            "val": [(None, args.val_embassy_total)],
        },
    }


def iter_targets(family_cfg):
    targets = []
    for province, count in family_cfg:
        for _ in range(count):
            targets.append(province)
    return targets


def make_manifest_row(out_path, text, family, split_name, pose_quad):
    return {
        "img_path": str(out_path),
        "text": text,
        "family": family,
        "source": f"special_ccpd2019_base_cvreplace_v1__{family}",
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


def build_qa_sheet(samples, qa_dir):
    if not samples:
        return
    for old_file in qa_dir.glob("qa_*.jpg"):
        old_file.unlink()

    for idx, sample in enumerate(samples[:18]):
        src = cv2.imread(sample["source_img"])
        out = cv2.imread(sample["generated_img"])
        pose_crop = make_pose_crop(out, np.array(sample["pose_quad"], dtype=np.float32))
        src_crop = make_pose_crop(src, np.array(sample["pose_quad"], dtype=np.float32)) if src is not None else None
        if src is None or out is None or pose_crop is None or src_crop is None:
            continue

        top_crop = degrade_for_qa(pose_crop)
        top_crop = cv2.resize(top_crop, (376, 96), interpolation=cv2.INTER_NEAREST)
        bottom_crop = cv2.resize(src_crop, (376, 96), interpolation=cv2.INTER_NEAREST)
        canvas_h = top_crop.shape[0] + bottom_crop.shape[0] + 54
        canvas_w = top_crop.shape[1] + 20
        canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)
        canvas[22 : 22 + top_crop.shape[0], 10 : 10 + top_crop.shape[1]] = top_crop
        y2 = 32 + top_crop.shape[0]
        canvas[y2 : y2 + bottom_crop.shape[0], 10 : 10 + bottom_crop.shape[1]] = bottom_crop
        cv2.putText(
            canvas,
            f'{sample["family"]} {sample["text"]} special-board-view',
            (10, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            canvas,
            "source-blue-view",
            (10, y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
        )
        cv_imwrite(qa_dir / f"qa_{idx:02d}_{sample['family']}.jpg", canvas)


def generate_split(split_name, pool, work_items, image_dir, renderer, skip_generate):
    rng = random.Random(20260517 if split_name == "train" else 20260518)
    schedule = build_schedule(pool, len(work_items), rng)
    rows = []
    qa_samples = []
    counts = Counter()
    skip_counts = Counter()
    guard_stats = []
    src_rect = None

    for src_rec, (family, province) in zip(schedule, work_items):
        img_path = src_rec["img_path"]
        img = cv2.imread(img_path)
        if img is None:
            skip_counts["read_fail"] += 1
            continue

        gt_quad_raw = np.array(src_rec["gt_quad"], dtype=np.float32)
        gt_quad = np.array([gt_quad_raw[2], gt_quad_raw[3], gt_quad_raw[0], gt_quad_raw[1]], dtype=np.float32)
        pose_quad = np.array(src_rec["pose_quad"], dtype=np.float32)
        text = renderer.sample_text(family, province=province)
        rendered = renderer.render_plate(text, family)
        plate_h, plate_w = rendered.image_bgr.shape[:2]
        if src_rect is None or src_rect.shape[0] != 4 or src_rect[2][0] != plate_w - 1 or src_rect[2][1] != plate_h - 1:
            src_rect = np.float32([[0, 0], [plate_w - 1, 0], [plate_w - 1, plate_h - 1], [0, plate_h - 1]])

        try:
            h, w = img.shape[:2]
            forward = cv2.getPerspectiveTransform(gt_quad.astype(np.float32), src_rect)
            source_patch = cv2.warpPerspective(
                img,
                forward,
                (plate_w, plate_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            transferred = transfer_style_l_only(rendered.image_bgr, source_patch)
            metrics = color_guard_metrics(transferred, family)
            metrics["family"] = family
            if not metrics["ok"]:
                raw_metrics = color_guard_metrics(rendered.image_bgr, family)
                raw_metrics["family"] = family
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
                transferred,
                inverse,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [gt_quad.round().astype(np.int32)], 255)
            mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
            result = (
                img.astype(np.float32) * (1.0 - mask[:, :, None]) + warped.astype(np.float32) * mask[:, :, None]
            ).clip(0, 255).astype(np.uint8)

            stem = Path(img_path).stem
            family_tag = {"yellow_single": "ys", "police": "pol", "embassy": "emb"}[family]
            out_path = image_dir / f"{stem}_{family_tag}_{text}.jpg"
            if not skip_generate:
                cv_imwrite(out_path, result)
            rows.append(make_manifest_row(out_path, text, family, split_name, pose_quad))
            counts[family] += 1

            if len(qa_samples) < 6:
                qa_samples.append(
                    {
                        "source_img": img_path,
                        "generated_img": str(out_path),
                        "family": family,
                        "text": text,
                        "pose_quad": pose_quad.tolist(),
                    }
                )
        except Exception:
            skip_counts["exception"] += 1
            continue

    return rows, counts, skip_counts, guard_stats, qa_samples


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_guard(metrics):
    if not metrics:
        return {}
    summary = {}
    for key in ["yellow_ratio", "white_ratio", "dark_ratio", "red_ratio"]:
        values = [m[key] for m in metrics]
        summary[key] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def main():
    args = parse_args()
    rng = random.Random(20260517)
    output_root = args.output_root
    dataset_dir = output_root / "datasets" / f"special_ccpd2019_base_cvreplace_v1_{args.date_tag}"
    manifest_dir = output_root / "manifests_rebased" / f"special_ccpd2019_base_cvreplace_v1_{args.date_tag}"
    qa_dir = dataset_dir / "qa"
    train_image_dir = dataset_dir / "images" / "train"
    val_image_dir = dataset_dir / "images" / "val"
    ensure_dirs(dataset_dir, manifest_dir, qa_dir, train_image_dir, val_image_dir)

    renderer = SpecialPlateRenderer(ROOT)
    pose_rows = load_pose_rows(args.pose_jsonl)
    train_pool, val_pool = split_source_rows(pose_rows, args.val_source_count)
    family_cfg = family_assignments(args)

    all_train_rows = []
    all_val_rows = []
    split_counts = {"train": Counter(), "val": Counter()}
    split_skips = {"train": Counter(), "val": Counter()}
    guard_summary = {"train": defaultdict(list), "val": defaultdict(list)}
    qa_samples = []

    t0 = time.time()
    for split_name, pool, image_dir in [("train", train_pool, train_image_dir), ("val", val_pool, val_image_dir)]:
        work_items = []
        for family, cfg in family_cfg.items():
            for province in iter_targets(cfg[split_name]):
                work_items.append((family, province))
        random.Random(202605170 if split_name == "train" else 202605171).shuffle(work_items)

        rows, counts, skips, metrics, samples = generate_split(
            split_name=split_name,
            pool=pool,
            work_items=work_items,
            image_dir=image_dir,
            renderer=renderer,
            skip_generate=args.skip_generate,
        )
        if split_name == "train":
            all_train_rows.extend(rows)
        else:
            all_val_rows.extend(rows)
        split_counts[split_name].update(counts)
        split_skips[split_name].update(skips)
        for metric in metrics:
            guard_summary[split_name][metric["family"]].append(metric)
        qa_samples.extend(samples)

    train_manifest = manifest_dir / "train_special_base_cvreplace.csv"
    val_manifest = manifest_dir / "val_special_base_cvreplace.csv"
    write_manifest(train_manifest, all_train_rows)
    write_manifest(val_manifest, all_val_rows)
    build_qa_sheet(qa_samples, qa_dir)

    meta = {
        "mode": "smoke" if args.smoke else "full",
        "pose_jsonl": str(args.pose_jsonl),
        "source_pool": {"train": len(train_pool), "val": len(val_pool)},
        "family_counts": {"train": dict(split_counts["train"]), "val": dict(split_counts["val"])},
        "skip_counts": {"train": dict(split_skips["train"]), "val": dict(split_skips["val"])},
        "guard_summary": {
            split: {family: summarize_guard(metrics) for family, metrics in family_map.items()}
            for split, family_map in guard_summary.items()
        },
        "manifests": {"train": str(train_manifest), "val": str(val_manifest)},
        "elapsed_sec": time.time() - t0,
    }
    with (dataset_dir / "generation_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
