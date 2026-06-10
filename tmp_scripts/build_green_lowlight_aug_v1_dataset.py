#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/wzzz/LPRNet")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_data import (  # noqa: E402
    box_to_quad,
    clamp_box,
    clip_quad_to_image,
    compute_expand_crop_box,
    compute_match_ytrim_crop,
    compute_ocr_crop_box,
    estimate_ocr_occ_ratio,
    parse_ccpd_bbox_from_name,
    parse_ccpd_quad_from_name,
    prepare_board_ocr_input_bgr888,
    prepare_board_ocr_input_from_quad_bgr888,
    quad_to_box,
)

DEFAULT_SOURCE = ROOT / "manifests_rebased/a_ratio_sweep_20260510/train_A_ratio_r50.csv"
DEFAULT_TEST = ROOT / "manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv"
DEFAULT_OUT_DATASET = ROOT / "datasets/green_lowlight_aug_v1_20260610"
DEFAULT_OUT_MANIFEST = ROOT / "manifests_rebased/green_lowlight_aug_v1_20260610"
DEFAULT_REAL_DUMP = Path("/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/green_dark")


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames)


def write_manifest(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_ppm(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    payload = np.ascontiguousarray(img).astype(np.uint8).copy()
    # load_data.read_ppm_p6_payload skips whitespace after the header; avoid
    # a dark first pixel being consumed as extra header whitespace.
    if payload.size and int(payload.reshape(-1)[0]) in (9, 10, 13, 32):
        payload.reshape(-1)[0] = 11
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(payload.tobytes())




def read_ppm_p6_exact(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if not blob.startswith(b"P6"):
        raise RuntimeError(f"unsupported ppm format: {path}")
    i = 2
    tokens = []
    n = len(blob)
    while len(tokens) < 3:
        while i < n and blob[i] in b" \t\r\n":
            i += 1
        if i < n and blob[i] == ord("#"):
            while i < n and blob[i] not in b"\r\n":
                i += 1
            continue
        j = i
        while j < n and blob[j] not in b" \t\r\n":
            j += 1
        tokens.append(blob[i:j].decode("ascii"))
        i = j
    w, h, maxv = map(int, tokens)
    if maxv != 255:
        raise RuntimeError(f"unsupported ppm max value {maxv} in {path}")
    if i < n and blob[i] in b" \t\r\n":
        i += 1
    payload = np.frombuffer(blob[i:], dtype=np.uint8)
    if payload.size != w * h * 3:
        raise RuntimeError(f"invalid ppm payload size in {path}: expect {w*h*3}, got {payload.size}")
    return payload.reshape(h, w, 3).copy()

def resolve_img(root: Path, raw: str):
    p = Path(raw)
    return p if p.is_absolute() else root / p


def row_quad(row, img_path, img_w, img_h, geometry_source="filename_first"):
    keys = ["quad_1x", "quad_1y", "quad_2x", "quad_2y", "quad_3x", "quad_3y", "quad_4x", "quad_4y"]

    def filename_quad():
        q = parse_ccpd_quad_from_name(row.get("img_rel_path") or row.get("img_path") or "")
        if q is None:
            q = parse_ccpd_quad_from_name(str(img_path))
        return clip_quad_to_image(q, img_w, img_h) if q is not None else None

    if geometry_source == "filename_first":
        q = filename_quad()
        if q is not None:
            return q, "ccpd_filename"

    try:
        if all(row.get(k) not in (None, "") for k in keys):
            q = np.asarray(
                [
                    (float(row["quad_1x"]), float(row["quad_1y"])),
                    (float(row["quad_2x"]), float(row["quad_2y"])),
                    (float(row["quad_3x"]), float(row["quad_3y"])),
                    (float(row["quad_4x"]), float(row["quad_4y"])),
                ],
                dtype=np.float32,
            )
            return clip_quad_to_image(q, img_w, img_h), row.get("quad_source") or "manifest_quad"
    except (ValueError, TypeError):
        pass

    q = filename_quad()
    return (q, "ccpd_filename") if q is not None else (None, "")


def board_ocr_from_row(root: Path, row, img_size=(94, 24), geometry_source="filename_first", resize_kernel_override=""):
    img_path = resolve_img(root, row["img_path"])
    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"failed to read {img_path}")
    img_h, img_w = image.shape[:2]
    quad, resolved_quad_source = row_quad(row, img_path, img_w, img_h, geometry_source=geometry_source)
    bbox = parse_ccpd_bbox_from_name(row.get("img_rel_path") or row.get("img_path") or "")
    if bbox is None:
        bbox = parse_ccpd_bbox_from_name(str(img_path))
    if bbox is None and quad is not None:
        bbox = quad_to_box(quad, img_w, img_h)
    if bbox is None:
        raise RuntimeError(f"cannot parse bbox for {row.get('img_path')}")
    bbox = clamp_box(bbox, img_w, img_h)

    crop_mode = row.get("ocr_crop_mode") or "obb_warp"
    resize_mode = row.get("ocr_resize_mode") or "letterbox"
    resize_kernel = resize_kernel_override or row.get("ocr_resize_kernel") or "nn"
    preproc_mode = row.get("ocr_preproc") or "none"
    channel_order = row.get("ocr_channel_order") or "bgr"
    min_occ_ratio = float(row.get("ocr_min_occ_ratio") or 0.90)
    quad_pad_ratio = float(row.get("ocr_quad_pad_ratio") or 0.0)

    if crop_mode == "obb_warp" and quad is None:
        quad = box_to_quad(bbox)
    if crop_mode == "obb_warp" and quad is not None:
        out, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
            image,
            quad,
            img_size[0],
            img_size[1],
            resize_mode,
            resize_kernel,
            preproc_mode,
            channel_order,
            quad_pad_ratio=quad_pad_ratio,
        )
        return out, {"resolved_quad_source": resolved_quad_source, "resolved_resize_kernel": resize_kernel}

    crop_box = compute_ocr_crop_box(bbox, img_w, img_h, crop_mode)
    occ = estimate_ocr_occ_ratio(crop_box.w, crop_box.h, img_size[0], img_size[1], resize_mode)
    if min_occ_ratio > 0.0 and occ < min_occ_ratio and crop_mode != "tight":
        if crop_mode == "match":
            recrop = compute_match_ytrim_crop(crop_box, img_size[0], img_size[1], min_occ_ratio)
            if recrop is not None:
                crop_box = clamp_box(recrop, img_w, img_h)
        else:
            crop_box = compute_expand_crop_box(bbox, img_w, img_h, 0.08, 0.16)
    crop_bgr = image[crop_box.y1 : crop_box.y2 + 1, crop_box.x1 : crop_box.x2 + 1]
    out, _ = prepare_board_ocr_input_bgr888(
        crop_bgr, img_size[0], img_size[1], resize_mode, resize_kernel, preproc_mode, channel_order
    )
    return out, {"resolved_quad_source": resolved_quad_source or "bbox_crop", "resolved_resize_kernel": resize_kernel}


def lowlight_transform(img_bgr: np.ndarray, rng: np.random.Generator, profile: str, style: str = "edge_preserve") -> np.ndarray:
    x = img_bgr.astype(np.float32)
    gray = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_mean = float(gray.mean())
    if style == "green_dark_match":
        # Match observed green_dark ocrin more closely: dark but not flat,
        # with green channel dominant and red channel suppressed.
        if profile == "ultra":
            target = rng.uniform(13.0, 18.0)
            contrast = rng.uniform(0.42, 0.68)
            noise_sigma = rng.uniform(0.25, 0.85)
            crush_prob = 0.08
        elif profile == "deep":
            target = rng.uniform(18.0, 25.5)
            contrast = rng.uniform(0.48, 0.76)
            noise_sigma = rng.uniform(0.35, 1.05)
            crush_prob = 0.06
        else:
            target = rng.uniform(25.5, 33.0)
            contrast = rng.uniform(0.55, 0.88)
            noise_sigma = rng.uniform(0.45, 1.25)
            crush_prob = 0.04
    elif style == "edge_preserve":
        # Real green_dark dumps are dark, but local character edges remain visible.
        # Avoid crushing already tiny 94x24 strokes into a flat field.
        if profile == "ultra":
            target = rng.uniform(11.5, 16.5)
            contrast = rng.uniform(0.32, 0.55)
            noise_sigma = rng.uniform(0.35, 1.1)
            crush_prob = 0.18
        elif profile == "deep":
            target = rng.uniform(16.5, 23.0)
            contrast = rng.uniform(0.38, 0.65)
            noise_sigma = rng.uniform(0.45, 1.3)
            crush_prob = 0.12
        else:
            target = rng.uniform(23.0, 31.0)
            contrast = rng.uniform(0.45, 0.78)
            noise_sigma = rng.uniform(0.55, 1.6)
            crush_prob = 0.06
    else:
        if profile == "ultra":
            target = rng.uniform(8.0, 12.5)
            contrast = rng.uniform(0.10, 0.22)
            noise_sigma = rng.uniform(0.6, 1.8)
            crush_prob = 0.72
        elif profile == "deep":
            target = rng.uniform(12.5, 18.0)
            contrast = rng.uniform(0.14, 0.30)
            noise_sigma = rng.uniform(0.8, 2.2)
            crush_prob = 0.52
        else:
            target = rng.uniform(18.0, 29.0)
            contrast = rng.uniform(0.20, 0.42)
            noise_sigma = rng.uniform(1.0, 2.8)
            crush_prob = 0.25

    y = (x - gray_mean) * contrast + target
    h, w = y.shape[:2]
    gx = np.linspace(rng.uniform(0.80, 1.05), rng.uniform(0.86, 1.18), w, dtype=np.float32)[None, :, None]
    gy = np.linspace(rng.uniform(0.86, 1.12), rng.uniform(0.84, 1.10), h, dtype=np.float32)[:, None, None]
    y = y * gx * gy
    if style == "green_dark_match":
        color = rng.normal([1.02, 1.18, 0.68], [0.05, 0.06, 0.07], size=(1, 1, 3)).astype(np.float32)
        y = y * np.clip(color, 0.45, 1.35)
    elif rng.random() < 0.60:
        # Very slight color bias. This keeps the dump-like RGB channel imbalance without changing labels.
        color = rng.normal(1.0, 0.035, size=(1, 1, 3)).astype(np.float32)
        y = y * color
    y += rng.normal(0.0, noise_sigma, size=y.shape).astype(np.float32)
    if rng.random() < crush_prob:
        threshold = rng.uniform(5.0, 14.0)
        floor = rng.uniform(0.0, 3.5)
        y = np.where(y < threshold, floor, y)
    quant_prob = 0.03 if style == "green_dark_match" else (0.08 if style == "edge_preserve" else 0.28)
    if rng.random() < quant_prob:
        step = float(rng.choice([1.0, 2.0, 3.0]))
        y = np.round(y / step) * step
    # Keep the pool concentrated on the intended low-light target instead of
    # letting high-contrast source plates leak back into normal brightness.
    y = y + (target - float(np.mean(y)))
    y = np.clip(y, 0, 255)
    if style in ("edge_preserve", "green_dark_match"):
        y = np.clip(y + (target - float(np.mean(y))), 0, 255)
    return y.astype(np.uint8)


def image_stats(img: np.ndarray):
    g = img.mean(axis=2).astype(np.float32)
    return {
        "mean": float(g.mean()),
        "p5": float(np.percentile(g, 5)),
        "p50": float(np.percentile(g, 50)),
        "p95": float(np.percentile(g, 95)),
        "dark30": float((g < 30).mean()),
        "bright220": float((g > 220).mean()),
    }


def bucket_mean(mean):
    if mean < 14:
        return "08-14_ultra"
    if mean < 22:
        return "14-22_deep"
    return "22-30_bridge"


def select_weighted(rows, count, seed, split_tag):
    rng = random.Random(seed)
    by_prov = defaultdict(list)
    for row in rows:
        if row.get("text"):
            by_prov[row["text"][0]].append(row)
    provinces = sorted(by_prov)
    # Province-balanced with replacement. Rare provinces are intentionally repeated because the source manifest is Anhui-heavy.
    selected = []
    per = count // len(provinces)
    rem = count % len(provinces)
    for i, prov in enumerate(provinces):
        n = per + (1 if i < rem else 0)
        bucket = by_prov[prov]
        for j in range(n):
            row = rng.choice(bucket)
            selected.append((row, prov, j))
    rng.shuffle(selected)
    return selected


def make_lowlight_rows(
    root,
    source_rows,
    header,
    out_dataset,
    out_manifest_dir,
    count,
    seed,
    split_name,
    start_idx=0,
    geometry_source="filename_first",
    lowlight_style="edge_preserve",
    resize_kernel_override="",
):
    selected = select_weighted(source_rows, count, seed, split_name)
    rng_py = random.Random(seed + 17)
    rng_np = np.random.default_rng(seed + 31)
    out_rows = []
    stat_rows = []
    profile_counts = Counter()
    province_counts = Counter()
    for i, (row, prov, repeat_idx) in enumerate(selected):
        u = rng_py.random()
        if u < 0.72:
            profile = "ultra"
        elif u < 0.92:
            profile = "deep"
        else:
            profile = "bridge"
        profile_counts[profile] += 1
        province_counts[prov] += 1
        board, board_meta = board_ocr_from_row(
            root,
            row,
            geometry_source=geometry_source,
            resize_kernel_override=resize_kernel_override,
        )
        img = lowlight_transform(board, rng_np, profile, style=lowlight_style)
        idx = start_idx + i
        rel = Path("datasets") / out_dataset.name / "images" / split_name / f"lowlight_{split_name}_{idx:06d}.ppm"
        abs_path = root / rel
        write_ppm(abs_path, img)

        nr = {k: row.get(k, "") for k in header}
        nr["img_path"] = str(rel)
        nr["text"] = row["text"]
        nr["family"] = "green8"
        nr["source"] = "green_lowlight_v1"
        nr["split"] = "train" if split_name == "train" else "test"
        nr["preprocess_group"] = "board_dump"
        nr["ocr_crop_mode"] = "board_dump"
        nr["ocr_resize_mode"] = "none"
        nr["ocr_resize_kernel"] = "none"
        nr["ocr_preproc"] = "none"
        nr["ocr_channel_order"] = row.get("ocr_channel_order") or "bgr"
        nr["ocr_quad_pad_ratio"] = "0.0"
        nr["quad_source"] = board_meta.get("resolved_quad_source", nr.get("quad_source", ""))
        out_rows.append(nr)

        sr = {
            "idx": idx,
            "split": split_name,
            "profile": profile,
            "province": prov,
            "source_img_path": row.get("img_path", ""),
            "img_path": str(rel),
            "text": row["text"],
            "source_repeat_idx": repeat_idx,
            "resolved_quad_source": board_meta.get("resolved_quad_source", ""),
            "resolved_resize_kernel": board_meta.get("resolved_resize_kernel", ""),
            "lowlight_style": lowlight_style,
        }
        sr.update(image_stats(img))
        sr["mean_bucket"] = bucket_mean(sr["mean"])
        stat_rows.append(sr)
        if (i + 1) % 5000 == 0:
            print(f"[GEN {split_name}] {i+1}/{count}", flush=True)
    return out_rows, stat_rows, profile_counts, province_counts


def build_contact_sheet(path: Path, root: Path, stat_rows, per_group=8):
    items = []
    for group in ["08-14_ultra", "14-22_deep", "22-30_bridge"]:
        subset = [r for r in stat_rows if r.get("mean_bucket") == group]
        subset = sorted(subset, key=lambda r: (r["province"], r["text"], r["idx"]))
        items.extend(subset[:per_group])
    thumbs = []
    for r in items:
        img = Image.open(root / r["img_path"]).convert("RGB").resize((282, 72), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (282, 108), "white")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((3, 75), f"{r['text']} {r['mean_bucket']} m={r['mean']:.1f}", fill=(0, 0, 0))
        draw.text((3, 91), f"{r['province']} d30={r['dark30']*100:.0f}% {Path(r['img_path']).name}", fill=(0, 0, 0))
        thumbs.append(canvas)
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 282, rows * 108), (230, 230, 230))
    for i, img in enumerate(thumbs):
        sheet.paste(img, ((i % cols) * 282, (i // cols) * 108))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=95)


def write_real_dump_manifest(path: Path, header, dump_dir: Path, gt: str, root: Path, out_dataset: Path):
    rows = []
    if not dump_dir.exists():
        return rows
    for idx, ppm in enumerate(sorted(dump_dir.glob("ocrin_*.ppm"))):
        img = read_ppm_p6_exact(ppm)
        rel = Path("datasets") / out_dataset.name / "images" / "real_dump" / f"real_green_dark_{idx:04d}.ppm"
        write_ppm(root / rel, img)
        row = {k: "" for k in header}
        row["img_path"] = str(rel)
        row["text"] = gt
        row["family"] = "green8"
        row["source"] = "board_green_dark"
        row["split"] = "test"
        row["preprocess_group"] = "board_dump"
        row["ocr_crop_mode"] = "board_dump"
        row["ocr_resize_mode"] = "none"
        row["ocr_resize_kernel"] = "none"
        row["ocr_preproc"] = "none"
        row["ocr_channel_order"] = "bgr"
        row["ocr_quad_pad_ratio"] = "0.0"
        rows.append(row)
    if rows:
        write_manifest(path, header, rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE))
    parser.add_argument("--test-manifest", default=str(DEFAULT_TEST))
    parser.add_argument("--dataset-root", default=str(ROOT))
    parser.add_argument("--out-dataset", default=str(DEFAULT_OUT_DATASET))
    parser.add_argument("--out-manifest-dir", default=str(DEFAULT_OUT_MANIFEST))
    parser.add_argument("--train-count", type=int, default=36000)
    parser.add_argument("--heldout-count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--real-dump-dir", default=str(DEFAULT_REAL_DUMP))
    parser.add_argument("--real-dump-gt", default="京ADA5396")
    parser.add_argument("--geometry-source", default="filename_first", choices=["filename_first", "manifest_first"])
    parser.add_argument("--lowlight-style", default="edge_preserve", choices=["edge_preserve", "green_dark_match", "legacy"])
    parser.add_argument("--resize-kernel-override", default="")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out_dataset = Path(args.out_dataset)
    out_manifest_dir = Path(args.out_manifest_dir)
    source_rows, header = load_rows(Path(args.source_manifest))
    train_rows = [r for r in source_rows if r.get("split") == "train" and r.get("family") == "green8"]
    train_low, train_stats, train_profiles, train_prov = make_lowlight_rows(
        root, train_rows, header, out_dataset, out_manifest_dir, args.train_count, args.seed, "train", 0,
        geometry_source=args.geometry_source, lowlight_style=args.lowlight_style,
        resize_kernel_override=args.resize_kernel_override,
    )
    heldout_low, heldout_stats, heldout_profiles, heldout_prov = make_lowlight_rows(
        root, train_rows, header, out_dataset, out_manifest_dir, args.heldout_count, args.seed + 100000, "heldout", 0,
        geometry_source=args.geometry_source, lowlight_style=args.lowlight_style,
        resize_kernel_override=args.resize_kernel_override,
    )

    write_manifest(out_manifest_dir / f"lowlight_train_pool_{len(train_low)}.csv", header, train_low)
    write_manifest(out_manifest_dir / f"lowlight_heldout_{len(heldout_low)}.csv", header, heldout_low)
    variant_ratios = {"lowlight10": 0.10, "lowlight15": 0.15, "lowlight25": 0.25}
    variants = {}
    base_n = len(source_rows)
    for name, ratio in variant_ratios.items():
        # n is chosen so lowlight rows are approximately ratio of the final mixed training manifest.
        n = int(round((base_n * ratio) / max(1e-6, 1.0 - ratio)))
        n = min(n, len(train_low))
        variants[name] = n
        write_manifest(out_manifest_dir / f"train_v1_{name}.csv", header, list(source_rows) + train_low[:n])
    real_rows = write_real_dump_manifest(
        out_manifest_dir / "real_green_dark_39.csv", header, Path(args.real_dump_dir), args.real_dump_gt, root, out_dataset
    )

    stat_header = list(train_stats[0].keys())
    with (out_manifest_dir / "lowlight_stats.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stat_header)
        writer.writeheader()
        writer.writerows(train_stats + heldout_stats)

    build_contact_sheet(out_manifest_dir / "lowlight_contact_sheet.jpg", root, train_stats)

    train_means = np.asarray([r["mean"] for r in train_stats], dtype=np.float32)
    train_dark30 = np.asarray([r["dark30"] for r in train_stats], dtype=np.float32)
    held_means = np.asarray([r["mean"] for r in heldout_stats], dtype=np.float32)
    summary = {
        "source_manifest": str(Path(args.source_manifest)),
        "test_manifest": str(Path(args.test_manifest)),
        "out_dataset": str(out_dataset),
        "out_manifest_dir": str(out_manifest_dir),
        "train_count": len(train_low),
        "heldout_count": len(heldout_low),
        "real_dump_count": len(real_rows),
        "train_profile_counts": dict(train_profiles),
        "heldout_profile_counts": dict(heldout_profiles),
        "train_province_count": len(train_prov),
        "heldout_province_count": len(heldout_prov),
        "train_mean_min": float(train_means.min()),
        "train_mean_p25": float(np.percentile(train_means, 25)),
        "train_mean_p50": float(np.percentile(train_means, 50)),
        "train_mean_p75": float(np.percentile(train_means, 75)),
        "train_mean_max": float(train_means.max()),
        "train_dark30_avg": float(train_dark30.mean()),
        "heldout_mean_p50": float(np.percentile(held_means, 50)),
        "manifests": {
            "train_pool": str(out_manifest_dir / f"lowlight_train_pool_{len(train_low)}.csv"),
            "heldout": str(out_manifest_dir / f"lowlight_heldout_{len(heldout_low)}.csv"),
            "real_dump": str(out_manifest_dir / "real_green_dark_39.csv"),
            **{name: str(out_manifest_dir / f"train_v1_{name}.csv") for name in variants},
        },
        "geometry_source": args.geometry_source,
        "lowlight_style": args.lowlight_style,
        "resize_kernel_override": args.resize_kernel_override,
        "notes": [
            "Training rows are generated only from the R50 green train manifest; real dump is validation only.",
            "Lowlight rows use board_dump 94x24 PPM payloads and should be read without additional crop/resize.",
            "Province-balanced sampling intentionally repeats rare-province source rows when the source manifest is province-skewed.",
        ],
    }
    (out_manifest_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_manifest_dir / "test_source.txt").write_text(str(Path(args.test_manifest)) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
