#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CCPD2020 green labels with targeted green labels and enforce strict split hygiene."
    )
    parser.add_argument("--ccpd-train", required=True)
    parser.add_argument("--ccpd-val", required=True)
    parser.add_argument("--ccpd-test", required=True)
    parser.add_argument("--targeted-train", default="")
    parser.add_argument("--targeted-val", default="")
    parser.add_argument("--targeted-test", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--image-roots",
        default="",
        help="Comma-separated image roots used to verify rel-path resolvability (e.g. ccpd_root,repo_root).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if invalid rows, leakage, or missing files are found.")
    return parser.parse_args()


def normalize_rel_path(raw_rel_path: str, source: str) -> str:
    rel = (raw_rel_path or "").strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]

    if source.startswith("targeted"):
        repo_prefix = "repo_license_plate_generator/"
        if rel.startswith(repo_prefix):
            rel = rel[len(repo_prefix):]

        marker = "targeted_green_missing_18/"
        idx = rel.find(marker)
        if idx >= 0:
            rel = rel[idx:]
        elif rel.startswith(("train/", "val/", "test/")):
            rel = f"targeted_green_missing_18/{rel}"

    rel = os.path.normpath(rel).replace("\\", "/")
    return rel


def read_rows(path: Path, source: str):
    rows = []
    invalid = []
    if not path:
        return rows, invalid
    if not path.exists():
        raise FileNotFoundError(f"label file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                invalid.append({"line": line_no, "reason": "missing label text", "content": line})
                continue
            rel_path, text = parts
            rel_path = normalize_rel_path(rel_path, source)
            if not rel_path or rel_path.startswith("../"):
                invalid.append({"line": line_no, "reason": "invalid rel path", "content": line})
                continue
            text = text.strip()
            if len(text) != 8:
                invalid.append(
                    {"line": line_no, "reason": f"non-8-char label ({len(text)})", "rel_path": rel_path, "text": text}
                )
                continue
            rows.append((rel_path, text))
    return rows, invalid


def dedupe_rows(rows):
    out = []
    seen = set()
    for rel_path, text in rows:
        if rel_path in seen:
            continue
        seen.add(rel_path)
        out.append((rel_path, text))
    return out


def path_set(rows):
    return {p for p, _ in rows}


def summarize(rows):
    province = Counter()
    lengths = Counter()
    for _, text in rows:
        if text:
            province[text[0]] += 1
            lengths[len(text)] += 1
    major_province, major_count = (None, 0)
    if province:
        major_province, major_count = max(province.items(), key=lambda kv: kv[1])
    total = len(rows)
    return {
        "sample_count": total,
        "province_distribution": dict(sorted(province.items())),
        "length_distribution": dict(sorted(lengths.items())),
        "major_province": major_province,
        "major_province_count": major_count,
        "major_province_ratio": (major_count / total) if total else 0.0,
    }


def split_missing(rows, image_roots):
    if not image_roots:
        return rows, []
    kept = []
    missing = []
    for rel_path, text in rows:
        filename = os.path.basename(rel_path)
        ok = False
        for root in image_roots:
            candidate_a = root / rel_path
            candidate_b = root / filename
            if candidate_a.exists() or candidate_b.exists():
                ok = True
                break
        if ok:
            kept.append((rel_path, text))
        else:
            missing.append((rel_path, text))
    return kept, missing


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rel_path, text in rows:
            f.write(f"{rel_path} {text}\n")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ccpd_train, invalid_ccpd_train = read_rows(Path(args.ccpd_train), "ccpd_train")
    ccpd_val, invalid_ccpd_val = read_rows(Path(args.ccpd_val), "ccpd_val")
    ccpd_test, invalid_ccpd_test = read_rows(Path(args.ccpd_test), "ccpd_test")

    targeted_train, invalid_targeted_train = read_rows(Path(args.targeted_train), "targeted_train") if args.targeted_train else ([], [])
    targeted_val, invalid_targeted_val = read_rows(Path(args.targeted_val), "targeted_val") if args.targeted_val else ([], [])
    targeted_test, invalid_targeted_test = read_rows(Path(args.targeted_test), "targeted_test") if args.targeted_test else ([], [])

    train_rows = dedupe_rows(ccpd_train + targeted_train)
    val_rows = dedupe_rows(ccpd_val + targeted_val)
    test_rows = dedupe_rows(ccpd_test + targeted_test)

    test_paths = path_set(test_rows)
    val_paths = path_set(val_rows)
    before_train = len(train_rows)
    before_val = len(val_rows)
    train_rows = [row for row in train_rows if row[0] not in val_paths and row[0] not in test_paths]
    val_rows = [row for row in val_rows if row[0] not in test_paths]
    train_rows = dedupe_rows(train_rows)
    val_rows = dedupe_rows(val_rows)
    test_rows = dedupe_rows(test_rows)

    image_roots = []
    if args.image_roots.strip():
        image_roots = [Path(item.strip()) for item in args.image_roots.split(",") if item.strip()]
        image_roots = [p for p in image_roots if p.exists()]

    train_rows, train_missing = split_missing(train_rows, image_roots)
    val_rows, val_missing = split_missing(val_rows, image_roots)
    test_rows, test_missing = split_missing(test_rows, image_roots)

    overlap_report = {
        "train_val": len(path_set(train_rows) & path_set(val_rows)),
        "train_test": len(path_set(train_rows) & path_set(test_rows)),
        "val_test": len(path_set(val_rows) & path_set(test_rows)),
    }

    invalid_rows = (
        invalid_ccpd_train
        + invalid_ccpd_val
        + invalid_ccpd_test
        + invalid_targeted_train
        + invalid_targeted_val
        + invalid_targeted_test
    )
    missing_count = len(train_missing) + len(val_missing) + len(test_missing)

    report = {
        "strict": bool(args.strict),
        "source_files": {
            "ccpd_train": str(Path(args.ccpd_train)),
            "ccpd_val": str(Path(args.ccpd_val)),
            "ccpd_test": str(Path(args.ccpd_test)),
            "targeted_train": str(Path(args.targeted_train)) if args.targeted_train else "",
            "targeted_val": str(Path(args.targeted_val)) if args.targeted_val else "",
            "targeted_test": str(Path(args.targeted_test)) if args.targeted_test else "",
        },
        "source_counts": {
            "ccpd_train": len(ccpd_train),
            "ccpd_val": len(ccpd_val),
            "ccpd_test": len(ccpd_test),
            "targeted_train": len(targeted_train),
            "targeted_val": len(targeted_val),
            "targeted_test": len(targeted_test),
        },
        "dropped_for_leak_guard": {
            "train_removed": before_train - len(train_rows),
            "val_removed": before_val - len(val_rows),
        },
        "missing_after_root_resolve": {
            "train": len(train_missing),
            "val": len(val_missing),
            "test": len(test_missing),
        },
        "invalid_row_count": len(invalid_rows),
        "invalid_preview": invalid_rows[:20],
        "missing_preview": {
            "train": train_missing[:20],
            "val": val_missing[:20],
            "test": test_missing[:20],
        },
        "overlap": overlap_report,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "test": summarize(test_rows),
        "output_files": {
            "train_labels": str((out_dir / "train_labels.txt").resolve()),
            "val_labels": str((out_dir / "val_labels.txt").resolve()),
            "test_labels": str((out_dir / "test_labels.txt").resolve()),
        },
    }

    if overlap_report["train_val"] > 0 or overlap_report["train_test"] > 0 or overlap_report["val_test"] > 0:
        raise RuntimeError(f"leakage detected: {overlap_report}")

    if len(train_rows) == 0 or len(val_rows) == 0 or len(test_rows) == 0:
        raise RuntimeError("empty split after merge/filtering")

    write_rows(out_dir / "train_labels.txt", train_rows)
    write_rows(out_dir / "val_labels.txt", val_rows)
    write_rows(out_dir / "test_labels.txt", test_rows)
    (out_dir / "merge_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.strict and (len(invalid_rows) > 0 or missing_count > 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
