#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLOv8-OBB plate dataset from CCPD splits.")
    parser.add_argument("--dataset_root", default="./CCPD2019", help="CCPD dataset root directory.")
    parser.add_argument("--split_dir", default=None, help="Directory that contains train.txt/val.txt/test.txt.")
    parser.add_argument(
        "--output_dir",
        default="./prepared_labels/ccpd2019_yolov8_obb",
        help="Output dataset directory containing images/, labels/, and dataset.yaml.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Split names to generate.")
    parser.add_argument(
        "--link-mode",
        default="symlink",
        choices=["symlink", "copy"],
        help="How to populate output images directory.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Optional cap for smoke testing. 0 means no limit.",
    )
    return parser.parse_args()


def normalize_rel_path(rel_path: str) -> str:
    return Path(rel_path.strip().replace("\\", "/")).as_posix()


def parse_ccpd_quad_from_name(rel_path: str) -> np.ndarray:
    stem = Path(rel_path).stem
    parts = stem.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected CCPD filename format: {rel_path}")
    points_text = parts[3]
    points = []
    for item in points_text.split("_"):
        xs, ys = item.split("&", 1)
        points.append((float(xs), float(ys)))
    if len(points) != 4:
        raise ValueError(f"Unexpected CCPD quad length in: {rel_path}")
    return np.asarray(points, dtype=np.float32)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    quad = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = quad.sum(axis=1)
    diffs = np.diff(quad, axis=1).reshape(-1)
    ordered[0] = quad[np.argmin(sums)]
    ordered[2] = quad[np.argmax(sums)]
    ordered[1] = quad[np.argmin(diffs)]
    ordered[3] = quad[np.argmax(diffs)]
    return ordered


def make_image_link(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "copy":
        shutil.copy2(src, dst)
        return
    rel_src = os.path.relpath(src, dst.parent)
    os.symlink(rel_src, dst)


def quad_to_yolo_obb_line(quad: np.ndarray, img_w: int, img_h: int, cls_id: int = 0) -> str:
    quad = order_quad_points(quad)
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: {(img_w, img_h)}")
    coords = []
    for x, y in quad:
        coords.append(f"{x / float(img_w):.6f}")
        coords.append(f"{y / float(img_h):.6f}")
    return f"{cls_id} " + " ".join(coords)


def prepare_split(dataset_root: Path, split_file: Path, output_dir: Path, split_name: str, link_mode: str, limit: int):
    image_root = output_dir / "images" / split_name
    label_root = output_dir / "labels" / split_name
    manifest_path = output_dir / f"{split_name}.txt"

    lines = split_file.read_text(encoding="utf-8").splitlines()
    manifest_rows = []
    valid = 0
    skipped = 0

    for raw_line in lines:
        rel_path = normalize_rel_path(raw_line)
        if not rel_path:
            continue
        src_img = dataset_root / rel_path
        if not src_img.exists():
            skipped += 1
            continue

        image = cv2.imread(str(src_img))
        if image is None:
            skipped += 1
            continue
        img_h, img_w = image.shape[:2]
        quad = parse_ccpd_quad_from_name(rel_path)
        label_line = quad_to_yolo_obb_line(quad, img_w, img_h)

        dst_img = image_root / rel_path
        dst_label = label_root / Path(rel_path).with_suffix(".txt")
        make_image_link(src_img, dst_img, link_mode)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text(label_line + "\n", encoding="utf-8")
        manifest_rows.append(str(dst_img.absolute()))
        valid += 1

        if limit > 0 and valid >= limit:
            break

    manifest_path.write_text("\n".join(manifest_rows) + ("\n" if manifest_rows else ""), encoding="utf-8")
    return valid, skipped, manifest_path


def write_dataset_yaml(output_dir: Path, split_names):
    yaml_path = output_dir / "dataset.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        f"train: {output_dir.resolve() / 'train.txt'}",
        f"val: {output_dir.resolve() / 'val.txt'}",
    ]
    if "test" in split_names:
        lines.append(f"test: {output_dir.resolve() / 'test.txt'}")
    lines.extend(
        [
            "names:",
            "  0: plate",
        ]
    )
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    split_dir = Path(args.split_dir) if args.split_dir else dataset_root / "splits"
    output_dir = Path(args.output_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    split_names = []
    for split_name in args.splits:
        split_file = split_dir / f"{split_name}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        valid, skipped, manifest = prepare_split(
            dataset_root,
            split_file,
            output_dir,
            split_name,
            args.link_mode,
            args.limit_per_split,
        )
        split_names.append(split_name)
        print(
            f"[OK] {split_name}: wrote {valid} image links and labels, "
            f"skipped={skipped}, manifest={manifest}"
        )

    yaml_path = write_dataset_yaml(output_dir, split_names)
    print(f"[OK] dataset yaml: {yaml_path.resolve()}")
    print(f"[Done] output_dir={output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
