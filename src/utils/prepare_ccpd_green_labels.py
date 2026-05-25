#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_ccpd_splits import ADS, PROVINCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate label txt files for CCPD green plates (8-char) from filename encoding."
    )
    parser.add_argument(
        "--dataset-root",
        default="./CCPD2020/ccpd_green",
        help="Root directory of CCPD green dataset, expected subdirs: train/val/test.",
    )
    parser.add_argument(
        "--output-dir",
        default="./prepared_labels/ccpd2020_green",
        help="Output directory for generated label txt files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split names to generate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any file fails to decode.",
    )
    return parser.parse_args()


def decode_green_plate_from_stem(stem: str) -> str:
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"unexpected CCPD filename format: {stem}")

    codes = [int(item) for item in parts[4].split("_")]
    if len(codes) != 8:
        raise ValueError(f"expected 8 encoded symbols for green plate, got {len(codes)}: {stem}")

    province_idx = codes[0]
    if not 0 <= province_idx < len(PROVINCES):
        raise ValueError(f"invalid province index {province_idx}: {stem}")

    plate_chars = [PROVINCES[province_idx]]
    for code in codes[1:]:
        if not 0 <= code < len(ADS):
            raise ValueError(f"invalid char index {code}: {stem}")
        plate_chars.append(ADS[code])
    return "".join(plate_chars)


def summarize(rows):
    province = Counter()
    lengths = Counter()
    for _, text in rows:
        if text:
            province[text[0]] += 1
            lengths[len(text)] += 1
    return {
        "sample_count": len(rows),
        "province_distribution": dict(sorted(province.items())),
        "length_distribution": dict(sorted(lengths.items())),
    }


def generate_split(dataset_root: Path, split: str):
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"split dir not found: {split_dir}")

    rows = []
    skipped = []
    for img_path in sorted(split_dir.glob("*.jpg")):
        rel_path = Path(split) / img_path.name
        try:
            plate = decode_green_plate_from_stem(img_path.stem)
            rows.append((rel_path.as_posix(), plate))
        except Exception as exc:  # pylint: disable=broad-except
            skipped.append({"file": rel_path.as_posix(), "reason": str(exc)})
    return rows, skipped


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rel_path, plate in rows:
            f.write(f"{rel_path} {plate}\n")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    report = {
        "dataset_root": str(dataset_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "strict": bool(args.strict),
        "splits": {},
        "total_samples": 0,
        "total_skipped": 0,
    }

    for split in args.splits:
        rows, skipped = generate_split(dataset_root, split)
        out_file = output_dir / f"{split}_labels.txt"
        write_rows(out_file, rows)

        split_report = summarize(rows)
        split_report["output_file"] = str(out_file)
        split_report["skipped_count"] = len(skipped)
        if skipped:
            split_report["skipped_preview"] = skipped[:20]
        report["splits"][split] = split_report
        report["total_samples"] += len(rows)
        report["total_skipped"] += len(skipped)

        print(
            f"[OK] {split}: wrote {len(rows)} labels to {out_file} "
            f"(skipped={len(skipped)})"
        )

    stats_path = output_dir / "split_stats.json"
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[Done] stats={stats_path}")

    if args.strict and report["total_skipped"] > 0:
        print(f"[FAIL] strict mode: skipped={report['total_skipped']}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
