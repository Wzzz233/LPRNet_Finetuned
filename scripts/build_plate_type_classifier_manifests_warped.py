#!/usr/bin/env python3
"""Build corrected 6-class plate type classifier manifests.

This version excludes CBLPRD and git_plate, keeps quad fields, and treats
real board fc224 embassy dumps as final holdout only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from plate_type_classifier_common import CLASS_MAP, CLASS_NAMES, ROOT, load_plate_bgr, plate_color_scores

DATE = "20260602"
OUT_DIR = ROOT / "manifests_rebased" / f"plate_type_classifier_6cls_warped_nocrop_{DATE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = random.Random(20260602)

FIELDS = [
    "img_path", "label", "label_name", "plate_text", "source", "source_split", "crop_mode",
    "original_w", "original_h",
    "quad_1x", "quad_1y", "quad_2x", "quad_2y", "quad_3x", "quad_3y", "quad_4x", "quad_4y",
    "is_synthetic", "is_gan", "base_id", "sample_weight", "notes",
]

records: List[Dict[str, str]] = []


def md5_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def img_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return 0, 0
        return img.shape[1], img.shape[0]


def quad_from_row(row: Dict[str, str]) -> Optional[List[float]]:
    vals: List[float] = []
    for i in range(1, 5):
        x = row.get(f"quad_{i}x", "")
        y = row.get(f"quad_{i}y", "")
        if x == "" or y == "":
            return None
        try:
            vals.extend([float(x), float(y)])
        except ValueError:
            return None
    return vals


def add_record(
    path: Path,
    label_name: str,
    plate_text: str,
    source: str,
    source_split: str,
    crop_mode: str,
    original_w: int,
    original_h: int,
    quad: Optional[List[float]] = None,
    is_synthetic: bool = False,
    is_gan: bool = False,
    base_id: Optional[str] = None,
    sample_weight: float = 1.0,
    notes: str = "",
) -> None:
    if not path.exists():
        return
    if base_id is None:
        base_id = md5_id(str(path))
    row: Dict[str, str] = {
        "img_path": str(path),
        "label": str(CLASS_MAP[label_name]),
        "label_name": label_name,
        "plate_text": plate_text,
        "source": source,
        "source_split": source_split,
        "crop_mode": crop_mode,
        "original_w": str(original_w),
        "original_h": str(original_h),
        "is_synthetic": "1" if is_synthetic else "0",
        "is_gan": "1" if is_gan else "0",
        "base_id": base_id,
        "sample_weight": f"{sample_weight:.2f}",
        "notes": notes,
    }
    for i in range(4):
        if quad is not None and len(quad) == 8:
            row[f"quad_{i + 1}x"] = f"{quad[2 * i]:.2f}"
            row[f"quad_{i + 1}y"] = f"{quad[2 * i + 1]:.2f}"
        else:
            row[f"quad_{i + 1}x"] = ""
            row[f"quad_{i + 1}y"] = ""
    records.append(row)


def label_from_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("使"):
        return "embassy"
    if text.endswith("警"):
        return "police"
    if text.startswith("领") or "港" in text or "澳" in text or text.endswith("学") or text.endswith("挂"):
        return "other"
    if len(text) == 8:
        return "green"
    if len(text) == 7:
        return "blue"
    return "other"


def label_from_warp_color(path: Path, quad: List[float], fallback_text: str) -> str:
    row = {"img_path": str(path), "crop_mode": "perspective_warp"}
    for i in range(4):
        row[f"quad_{i + 1}x"] = str(quad[2 * i])
        row[f"quad_{i + 1}y"] = str(quad[2 * i + 1])
    plate = load_plate_bgr(row)
    scores = plate_color_scores(plate)
    if scores["yellow"] > 0.28 and scores["yellow"] > scores["blue"] * 1.15:
        return "yellow"
    if scores["blue"] > 0.25 and scores["blue"] >= scores["yellow"]:
        return "blue"
    return label_from_text(fallback_text)


def scan_ccpd2019() -> None:
    src = ROOT / "manifests_rebased" / "blue_ccpd2019_tilt_db_challenge_posquad_20260508" / "all_posquad.csv"
    if not src.exists():
        print("[skip] CCPD2019 manifest missing")
        return
    n = 0
    with open(src, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if n >= 8000:
                break
            path = ROOT / row["img_path"]
            quad = quad_from_row(row)
            if quad is None or not path.exists():
                continue
            w, h = img_size(path)
            if w == 0:
                continue
            text = row.get("text", "")
            add_record(path, "blue", text, "ccpd2019", row.get("split", ""), "perspective_warp", w, h, quad=quad)
            n += 1
    print(f"ccpd2019: {n}", flush=True)


def scan_ccpd2020() -> None:
    src = ROOT / "manifests_rebased" / "ccpd2020_green_real_20260509" / "train_ccpd2020_green_real.csv"
    if not src.exists():
        print("[skip] CCPD2020 manifest missing")
        return
    n = 0
    with open(src, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            path = ROOT / row["img_path"]
            quad = quad_from_row(row)
            if quad is None or not path.exists():
                continue
            w, h = img_size(path)
            if w == 0:
                continue
            add_record(path, "green", row.get("text", ""), "ccpd2020", "train", "perspective_warp", w, h, quad=quad)
            n += 1
    print(f"ccpd2020: {n}", flush=True)


def scan_crpd_all() -> None:
    base = ROOT / "datasets" / "CRPD_all"
    limits = {
        "CRPD_single": {"train": 2500, "val": 700, "test": 700},
        "CRPD_double": {"train": 1200, "val": 300, "test": 300},
        "CRPD_multi": {"train": 600, "val": 150, "test": 150},
    }
    total = 0
    color_stats = Counter()
    for layout, split_limits in limits.items():
        source = f"crpd_{layout.split('_')[-1].lower()}"
        for split in ["train", "val", "test"]:
            limit = split_limits[split]
            seen_split = 0
            img_dir = base / layout / split / "images"
            lbl_dir = base / layout / split / "labels"
            if not img_dir.is_dir() or not lbl_dir.is_dir():
                continue
            for lbl_path in sorted(lbl_dir.glob("*.txt")):
                if seen_split >= limit:
                    break
                img_path = img_dir / f"{lbl_path.stem}.jpg"
                if not img_path.exists():
                    continue
                lines = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                if not lines:
                    continue
                parts = lines[0].split()
                if len(parts) < 10:
                    continue
                try:
                    quad = [float(x) for x in parts[:8]]
                except ValueError:
                    continue
                text = parts[-1]
                w, h = img_size(img_path)
                if w == 0:
                    continue
                label = label_from_warp_color(img_path, quad, text)
                if label not in ("blue", "yellow", "police", "other"):
                    label = "other"
                color_stats[f"{split}:{label}"] += 1
                base_id = md5_id(f"{img_path}:{parts[0]}:{parts[1]}:{text}")
                add_record(img_path, label, text, source, split, "perspective_warp", w, h, quad=quad, base_id=base_id)
                seen_split += 1
                total += 1
    print(f"crpd_all: {total} {dict(color_stats)}", flush=True)

def scan_special_v2() -> None:
    manifest_dir = ROOT / "manifests_rebased" / "special_split_v2_20260601"
    total = 0
    for split in ["train", "val_clean", "val_hard"]:
        for label_name in ["police", "embassy"]:
            path = manifest_dir / f"{split}_{label_name}.csv"
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    img_path_text = row.get("img_path", "")
                    if img_path_text.startswith("scripts/special_gen/datasets/"):
                        rel = img_path_text.split("scripts/special_gen/datasets/", 1)[1]
                        img_path = ROOT / "datasets" / rel
                    else:
                        img_path = ROOT / img_path_text
                    quad = quad_from_row(row)
                    if quad is None or not img_path.exists():
                        continue
                    w, h = img_size(img_path)
                    if w == 0:
                        continue
                    add_record(
                        img_path, label_name, row.get("text", ""), "special_v2", split,
                        "perspective_warp", w, h, quad=quad, is_synthetic=True,
                        sample_weight=2.0, notes="real_board_aug_on_train_loader",
                    )
                    total += 1
    print(f"special_v2: {total}", flush=True)


def scan_real_embassy_holdout() -> None:
    roots = [
        Path("/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/embassy"),
        Path("/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/embassy_2"),
    ]
    n = 0
    for folder in roots:
        for path in sorted(folder.glob("fc224_*.ppm")):
            w, h = img_size(path)
            if w == 0:
                continue
            add_record(
                path, "embassy", "REAL_EMBASSY", "real_embassy_board_dump", "final_holdout",
                "board_fc224", w, h, is_synthetic=False, base_id=md5_id(str(path)),
                notes="user_real_embassy_fc224_not_for_training",
            )
            n += 1
    print(f"real_embassy_holdout: {n}", flush=True)


def split_records() -> Dict[str, List[Dict[str, str]]]:
    splits = {"train": [], "val_clean": [], "val_hard": [], "val_cross_source": [], "final_holdout": []}
    train_ids = set()
    for row in records:
        src = row["source"]
        src_split = row["source_split"]
        bid = row["base_id"]
        if src == "real_embassy_board_dump":
            splits["final_holdout"].append(row)
            continue
        if src == "special_v2":
            target = {"train": "train", "val_clean": "val_clean", "val_hard": "val_hard"}.get(src_split, "train")
        elif src.startswith("crpd_"):
            target = {"train": "train", "val": "val_clean", "test": "val_hard"}.get(src_split, "train")
        else:
            bucket = int(bid[:8], 16) % 20
            target = "train" if bucket < 14 else "val_clean" if bucket < 16 else "val_hard" if bucket < 18 else "val_cross_source"
        if target == "train":
            train_ids.add(bid)
        splits[target].append(row)

    for name in ["val_clean", "val_hard", "val_cross_source"]:
        before = len(splits[name])
        splits[name] = [r for r in splits[name] if r["base_id"] not in train_ids]
        removed = before - len(splits[name])
        if removed:
            print(f"removed train overlap from {name}: {removed}")
    return splits


def write_outputs(splits: Dict[str, List[Dict[str, str]]]) -> None:
    for name, rows in splits.items():
        with open(OUT_DIR / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}.csv: {len(rows)}")

    with open(OUT_DIR / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_MAP, f, ensure_ascii=False, indent=2)

    stats = {}
    for name, rows in splits.items():
        by_class = Counter(r["label_name"] for r in rows)
        by_source = defaultdict(Counter)
        for row in rows:
            by_source[row["source"]][row["label_name"]] += 1
        stats[name] = {
            "total": len(rows),
            "by_class": dict(by_class),
            "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
        }
    with open(OUT_DIR / "source_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    summary = {
        "date": DATE,
        "class_map": CLASS_MAP,
        "output_dir": str(OUT_DIR),
        "excluded_sources": ["datasets/CRPD_raw_ccpd_board_v1", "datasets/git_plate", "datasets/CBLPRD-330k_v1"],
        "sources_used": sorted({r["source"] for r in records}),
        "splits": {k: len(v) for k, v in splits.items()},
        "notes": [
            "All full-frame sources require perspective_warp or are excluded.",
            "real_embassy_board_dump is final_holdout only.",
            "special_v2 train samples are augmented in the training loader to resemble real board fc224 dumps.",
        ],
    }
    with open(OUT_DIR / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    scan_ccpd2019()
    scan_ccpd2020()
    scan_crpd_all()
    scan_special_v2()
    scan_real_embassy_holdout()
    splits = split_records()
    write_outputs(splits)
    print(f"written: {OUT_DIR}")


if __name__ == "__main__":
    main()

