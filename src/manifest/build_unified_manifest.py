#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from load_data import parse_ccpd_bbox_from_name, parse_ccpd_quad_from_name

MANIFEST_FIELDS = [
    "img_path",
    "img_rel_path",
    "dataset_name",
    "split",
    "text",
    "plate_len",
    "family",
    "sub_type",
    "source",
    "is_real",
    "need_tilt_aug",
    "preprocess_group",
    "has_bbox",
    "has_quad",
    "can_parse_ccpd_geom",
    "can_perspective",
    "bbox_source",
    "quad_source",
    "ocr_channel_order",
    "ocr_crop_mode",
    "ocr_resize_mode",
    "ocr_resize_kernel",
    "ocr_preproc",
    "ocr_min_occ_ratio",
    "ocr_quad_pad_ratio",
]


def infer_preprocess_group(rel_path: str, dataset_name: str) -> str:
    rel = (rel_path or "").replace("\\", "/")
    ds = (dataset_name or "").lower()
    if ds == "board_dump" or rel.lower().endswith(".ppm"):
        return "board_dump"
    # 关键约束：除 CCPD 类数据外，其它只有车牌文本标签的数据即便文件名看起来像带坐标，
    # 也不能默认认为具备可用的 bbox/quad 透视信息。
    if ds in {"ccpd2019", "ccpd2019_hard_tilt", "ccpd2020_green"}:
        if parse_ccpd_bbox_from_name(rel) is not None or parse_ccpd_quad_from_name(rel) is not None:
            return "ccpd_board"
    return "plain_plate"


def build_row_from_label_entry(
    img_root: str,
    rel_path: str,
    text: str,
    split: str,
    family: str,
    sub_type: str,
    source: str,
    is_real: int,
    need_tilt_aug: int,
    dataset_name: str,
    ocr_channel_order: str = "bgr",
    ocr_crop_mode: str = "match",
    ocr_resize_mode: str = "letterbox",
    ocr_resize_kernel: str = "nn",
    ocr_preproc: str = "none",
    ocr_min_occ_ratio: float = 0.90,
    ocr_quad_pad_ratio: float = 0.0,
) -> Dict[str, object]:
    rel_norm = str(Path(rel_path)).replace("\\", "/")
    img_path = str((Path(img_root) / rel_norm).resolve())
    preprocess_group = infer_preprocess_group(rel_norm, dataset_name)
    bbox = parse_ccpd_bbox_from_name(rel_norm) if preprocess_group == "ccpd_board" else None
    quad = parse_ccpd_quad_from_name(rel_norm) if preprocess_group == "ccpd_board" else None
    has_bbox = 1 if bbox is not None else 0
    has_quad = 1 if quad is not None else 0
    can_parse_ccpd_geom = 1 if preprocess_group == "ccpd_board" and (has_bbox or has_quad) else 0
    can_perspective = 1 if preprocess_group == "ccpd_board" and has_quad else 0

    if preprocess_group == "board_dump":
        crop_mode = "board_dump"
        min_occ = 1.0
        quad_pad = 0.0
    elif preprocess_group == "plain_plate":
        crop_mode = "plain_plate"
        min_occ = 1.0
        quad_pad = 0.0
    else:
        crop_mode = ocr_crop_mode
        min_occ = ocr_min_occ_ratio
        quad_pad = ocr_quad_pad_ratio

    return {
        "img_path": img_path,
        "img_rel_path": rel_norm,
        "dataset_name": dataset_name,
        "split": split,
        "text": text,
        "plate_len": len(text),
        "family": family,
        "sub_type": sub_type,
        "source": source,
        "is_real": int(is_real),
        "need_tilt_aug": int(need_tilt_aug),
        "preprocess_group": preprocess_group,
        "has_bbox": has_bbox,
        "has_quad": has_quad,
        "can_parse_ccpd_geom": can_parse_ccpd_geom,
        "can_perspective": can_perspective,
        "bbox_source": "ccpd_filename" if has_bbox else "none",
        "quad_source": "ccpd_filename" if has_quad else "none",
        "ocr_channel_order": ocr_channel_order,
        "ocr_crop_mode": crop_mode,
        "ocr_resize_mode": ocr_resize_mode,
        "ocr_resize_kernel": ocr_resize_kernel,
        "ocr_preproc": ocr_preproc,
        "ocr_min_occ_ratio": min_occ,
        "ocr_quad_pad_ratio": quad_pad,
    }


def read_label_txt(path: str) -> List[tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, text = line.split(maxsplit=1)
            rows.append((rel_path, text))
    return rows


def write_manifest(rows: Iterable[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    dataset_counter = Counter()
    preprocess_counter = Counter()
    family_counter = Counter()
    split_counter = Counter()
    for row in rows:
        dataset_counter[row["dataset_name"]] += 1
        preprocess_counter[row["preprocess_group"]] += 1
        family_counter[row["family"]] += 1
        split_counter[row["split"]] += 1
    return {
        "sample_count": len(rows),
        "datasets": dict(sorted(dataset_counter.items())),
        "preprocess_groups": dict(sorted(preprocess_counter.items())),
        "families": dict(sorted(family_counter.items())),
        "splits": dict(sorted(split_counter.items())),
    }


def _dataset_specs_from_args(args: argparse.Namespace) -> List[Dict[str, object]]:
    specs = []
    if args.ccpd2019_root and args.ccpd2019_train and args.ccpd2019_val and args.ccpd2019_test:
        specs.extend([
            dict(dataset_name="ccpd2019", img_root=args.ccpd2019_root, txt=args.ccpd2019_train, split="train", family="normal7", sub_type="blue", source="real", is_real=1, need_tilt_aug=1),
            dict(dataset_name="ccpd2019", img_root=args.ccpd2019_root, txt=args.ccpd2019_val, split="val", family="normal7", sub_type="blue", source="real", is_real=1, need_tilt_aug=1),
            dict(dataset_name="ccpd2019", img_root=args.ccpd2019_root, txt=args.ccpd2019_test, split="test", family="normal7", sub_type="blue", source="real", is_real=1, need_tilt_aug=1),
        ])
    if args.ccpd2019_hard_tilt_root and args.ccpd2019_hard_tilt_train and args.ccpd2019_hard_tilt_val and args.ccpd2019_hard_tilt_test:
        specs.extend([
            dict(dataset_name="ccpd2019_hard_tilt", img_root=args.ccpd2019_hard_tilt_root, txt=args.ccpd2019_hard_tilt_train, split="train", family="normal7", sub_type="blue", source="real", is_real=1, need_tilt_aug=1),
            dict(dataset_name="ccpd2019_hard_tilt", img_root=args.ccpd2019_hard_tilt_root, txt=args.ccpd2019_hard_tilt_val, split="val", family="normal7", sub_type="blue", source="real", is_real=1, need_tilt_aug=1),
            dict(dataset_name="ccpd2019_hard_tilt", img_root=args.ccpd2019_hard_tilt_root, txt=args.ccpd2019_hard_tilt_test, split="test", family="normal7", sub_type="blue", source="real", is_real=1, need_tilt_aug=1),
        ])
    if args.ccpd2020_green_root and args.ccpd2020_green_train and args.ccpd2020_green_val and args.ccpd2020_green_test:
        specs.extend([
            dict(dataset_name="ccpd2020_green", img_root=args.ccpd2020_green_root, txt=args.ccpd2020_green_train, split="train", family="green8", sub_type="green", source="real", is_real=1, need_tilt_aug=1),
            dict(dataset_name="ccpd2020_green", img_root=args.ccpd2020_green_root, txt=args.ccpd2020_green_val, split="val", family="green8", sub_type="green", source="real", is_real=1, need_tilt_aug=1),
            dict(dataset_name="ccpd2020_green", img_root=args.ccpd2020_green_root, txt=args.ccpd2020_green_test, split="test", family="green8", sub_type="green", source="real", is_real=1, need_tilt_aug=1),
        ])
    if args.targeted_green_root and args.targeted_green_train and args.targeted_green_val:
        specs.extend([
            dict(dataset_name="targeted_green_missing_18", img_root=args.targeted_green_root, txt=args.targeted_green_train, split="train", family="green8", sub_type="green", source="synthetic_full", is_real=0, need_tilt_aug=1),
            dict(dataset_name="targeted_green_missing_18", img_root=args.targeted_green_root, txt=args.targeted_green_val, split="val", family="green8", sub_type="green", source="synthetic_full", is_real=0, need_tilt_aug=1),
        ])
    if args.board_dump_root and args.board_dump_txt:
        specs.append(dict(dataset_name="board_dump", img_root=args.board_dump_root, txt=args.board_dump_txt, split="eval", family="normal7", sub_type="blue", source="board_dump", is_real=1, need_tilt_aug=0))
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified OCR manifest with explicit geometry/perspective capability fields.")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-json", default="")

    p.add_argument("--ccpd2019-root", default="")
    p.add_argument("--ccpd2019-train", default="")
    p.add_argument("--ccpd2019-val", default="")
    p.add_argument("--ccpd2019-test", default="")

    p.add_argument("--ccpd2019-hard-tilt-root", default="")
    p.add_argument("--ccpd2019-hard-tilt-train", default="")
    p.add_argument("--ccpd2019-hard-tilt-val", default="")
    p.add_argument("--ccpd2019-hard-tilt-test", default="")

    p.add_argument("--ccpd2020-green-root", default="")
    p.add_argument("--ccpd2020-green-train", default="")
    p.add_argument("--ccpd2020-green-val", default="")
    p.add_argument("--ccpd2020-green-test", default="")

    p.add_argument("--targeted-green-root", default="")
    p.add_argument("--targeted-green-train", default="")
    p.add_argument("--targeted-green-val", default="")

    p.add_argument("--board-dump-root", default="")
    p.add_argument("--board-dump-txt", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = _dataset_specs_from_args(args)
    rows: List[Dict[str, object]] = []
    for spec in specs:
        for rel_path, text in read_label_txt(spec["txt"]):
            rows.append(
                build_row_from_label_entry(
                    img_root=spec["img_root"],
                    rel_path=rel_path,
                    text=text,
                    split=spec["split"],
                    family=spec["family"],
                    sub_type=spec["sub_type"],
                    source=spec["source"],
                    is_real=spec["is_real"],
                    need_tilt_aug=spec["need_tilt_aug"],
                    dataset_name=spec["dataset_name"],
                )
            )
    out_csv = Path(args.out_csv)
    write_manifest(rows, out_csv)
    summary = summarize_rows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
