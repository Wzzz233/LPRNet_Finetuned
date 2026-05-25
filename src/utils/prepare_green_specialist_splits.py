#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare strict train/val/test splits for green specialist OCR training."
    )
    parser.add_argument("--green-train", required=True)
    parser.add_argument("--green-val", required=True)
    parser.add_argument("--green-test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260320)

    # Legacy outputs kept for compatibility with existing scripts.
    parser.add_argument("--train-cap-per-province", type=int, default=2200)
    parser.add_argument("--train-max-major-ratio", type=float, default=0.90)
    parser.add_argument("--val-balanced-per-province", type=int, default=120)
    parser.add_argument("--test-balanced-per-province", type=int, default=260)

    # New v2 protocol.
    parser.add_argument("--train-balanced-v2-per-province", type=int, default=50)
    parser.add_argument("--train-full-capped-v2-per-province", type=int, default=300)
    parser.add_argument("--train-full-capped-v2-max-major-ratio", type=float, default=0.35)
    parser.add_argument("--val-balanced-v2-per-province", type=int, default=20)
    parser.add_argument("--test-balanced-v2-per-province", type=int, default=40)
    return parser.parse_args()


def read_label_rows(path: Path):
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, text = line.split(maxsplit=1)
            if rel_path in seen:
                continue
            seen.add(rel_path)
            rows.append((rel_path, text))
    return rows


def write_label_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rel_path, text in rows:
            f.write(f"{rel_path} {text}\n")


def path_set(rows):
    return {p for p, _ in rows}


def group_by_province(rows):
    buckets = defaultdict(list)
    for rel_path, text in rows:
        province = text[0] if text else "?"
        buckets[province].append((rel_path, text))
    return buckets


def sample_by_province_cap(rows, cap, seed):
    if cap <= 0:
        return list(rows)
    buckets = group_by_province(rows)
    rng = random.Random(seed)
    out = []
    for province in sorted(buckets):
        chunk = list(buckets[province])
        rng.shuffle(chunk)
        out.extend(chunk[:cap])
    rng.shuffle(out)
    return out


def limit_major_ratio(rows, max_major_ratio, seed):
    if max_major_ratio <= 0.0 or max_major_ratio >= 1.0:
        return list(rows)
    if len(rows) <= 1:
        return list(rows)

    rng = random.Random(seed)
    buckets = {province: list(chunk) for province, chunk in group_by_province(rows).items()}
    for province in buckets:
        rng.shuffle(buckets[province])

    while True:
        total = sum(len(chunk) for chunk in buckets.values())
        if total <= 1:
            break
        major_province = max(buckets.keys(), key=lambda k: len(buckets[k]))
        major_count = len(buckets[major_province])
        other_count = total - major_count
        if other_count <= 0:
            break
        allowed_major = int((other_count * max_major_ratio) / (1.0 - max_major_ratio))
        allowed_major = max(1, allowed_major)
        if major_count <= allowed_major:
            break
        buckets[major_province] = buckets[major_province][:allowed_major]

    out = []
    for province in sorted(buckets):
        out.extend(buckets[province])
    rng.shuffle(out)
    return out


def dedupe_rows(rows):
    out = []
    seen = set()
    for rel_path, text in rows:
        if rel_path in seen:
            continue
        seen.add(rel_path)
        out.append((rel_path, text))
    return out


def overlap(a_rows, b_rows):
    return len(path_set(a_rows) & path_set(b_rows))


def major_province_stats(rows):
    counts = Counter()
    for _, text in rows:
        if text:
            counts[text[0]] += 1
    if not counts:
        return {"province": None, "count": 0, "ratio": 0.0}
    province, count = max(counts.items(), key=lambda kv: kv[1])
    total = sum(counts.values())
    return {"province": province, "count": count, "ratio": (count / total) if total else 0.0}


def summarize(rows):
    province = Counter()
    lengths = Counter()
    for _, text in rows:
        if text:
            province[text[0]] += 1
            lengths[len(text)] += 1
    major = major_province_stats(rows)
    return {
        "sample_count": len(rows),
        "province_distribution": dict(sorted(province.items())),
        "length_distribution": dict(sorted(lengths.items())),
        "major_province": major["province"],
        "major_province_count": major["count"],
        "major_province_ratio": major["ratio"],
    }


def source_counts(*named_rows):
    return {name: len(rows) for name, rows in named_rows}


def ensure_no_overlap(report_name, rows_a, rows_b):
    ov = overlap(rows_a, rows_b)
    if ov != 0:
        raise RuntimeError(f"leakage detected in {report_name}: {ov}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_label_rows(Path(args.green_train))
    val_rows = read_label_rows(Path(args.green_val))
    test_rows = read_label_rows(Path(args.green_test))

    if overlap(train_rows, val_rows) > 0 or overlap(train_rows, test_rows) > 0 or overlap(val_rows, test_rows) > 0:
        raise RuntimeError("green split leakage detected in source labels")

    train_bal = sample_by_province_cap(train_rows, args.train_cap_per_province, args.seed + 1)
    train_bal = limit_major_ratio(train_bal, args.train_max_major_ratio, args.seed + 2)
    val_bal = sample_by_province_cap(val_rows, args.val_balanced_per_province, args.seed + 3)
    test_bal = sample_by_province_cap(test_rows, args.test_balanced_per_province, args.seed + 4)

    train_bal_v2 = sample_by_province_cap(train_rows, args.train_balanced_v2_per_province, args.seed + 11)
    train_full_capped_v2 = sample_by_province_cap(train_rows, args.train_full_capped_v2_per_province, args.seed + 12)
    train_full_capped_v2 = limit_major_ratio(
        train_full_capped_v2,
        args.train_full_capped_v2_max_major_ratio,
        args.seed + 13,
    )
    train_stagec_v2 = dedupe_rows(train_full_capped_v2 + train_bal_v2)
    val_bal_v2 = sample_by_province_cap(val_rows, args.val_balanced_v2_per_province, args.seed + 14)
    test_bal_v2 = sample_by_province_cap(test_rows, args.test_balanced_v2_per_province, args.seed + 15)

    files = {
        "train_full": out_dir / "train_green_full_labels.txt",
        "train_balanced": out_dir / "train_green_balanced_labels.txt",
        "val_full": out_dir / "val_green_full_labels.txt",
        "val_balanced": out_dir / "val_green_balanced_labels.txt",
        "test_full": out_dir / "test_green_full_labels.txt",
        "test_balanced": out_dir / "test_green_balanced_labels.txt",
        "train_balanced_v2": out_dir / "train_green_balanced_v2_labels.txt",
        "train_full_capped_v2": out_dir / "train_green_full_capped_v2_labels.txt",
        "train_stagec_v2": out_dir / "train_green_stagec_v2_labels.txt",
        "val_balanced_v2": out_dir / "val_green_balanced_v2_labels.txt",
        "test_balanced_v2": out_dir / "test_green_balanced_v2_labels.txt",
    }
    for key, rows in (
        ("train_full", train_rows),
        ("train_balanced", train_bal),
        ("val_full", val_rows),
        ("val_balanced", val_bal),
        ("test_full", test_rows),
        ("test_balanced", test_bal),
        ("train_balanced_v2", train_bal_v2),
        ("train_full_capped_v2", train_full_capped_v2),
        ("train_stagec_v2", train_stagec_v2),
        ("val_balanced_v2", val_bal_v2),
        ("test_balanced_v2", test_bal_v2),
    ):
        write_label_rows(files[key], rows)

    leak_checks = {
        "trainfull_valfull": overlap(train_rows, val_rows),
        "trainfull_testfull": overlap(train_rows, test_rows),
        "valfull_testfull": overlap(val_rows, test_rows),
        "trainbal_valfull": overlap(train_bal, val_rows),
        "trainbal_testfull": overlap(train_bal, test_rows),
        "train_balanced_v2_valfull": overlap(train_bal_v2, val_rows),
        "train_balanced_v2_testfull": overlap(train_bal_v2, test_rows),
        "train_full_capped_v2_valfull": overlap(train_full_capped_v2, val_rows),
        "train_full_capped_v2_testfull": overlap(train_full_capped_v2, test_rows),
        "train_stagec_v2_valfull": overlap(train_stagec_v2, val_rows),
        "train_stagec_v2_testfull": overlap(train_stagec_v2, test_rows),
    }
    if any(v > 0 for v in leak_checks.values()):
        raise RuntimeError(f"leakage detected after balancing: {leak_checks}")

    ensure_no_overlap("val_balanced_v2 vs train", val_bal_v2, train_rows)
    ensure_no_overlap("test_balanced_v2 vs train", test_bal_v2, train_rows)
    ensure_no_overlap("val_balanced_v2 vs test", val_bal_v2, test_rows)
    ensure_no_overlap("test_balanced_v2 vs val", test_bal_v2, val_rows)

    report = {
        "seed": args.seed,
        "caps": {
            "train_cap_per_province": args.train_cap_per_province,
            "train_max_major_ratio": args.train_max_major_ratio,
            "val_balanced_per_province": args.val_balanced_per_province,
            "test_balanced_per_province": args.test_balanced_per_province,
            "train_balanced_v2_per_province": args.train_balanced_v2_per_province,
            "train_full_capped_v2_per_province": args.train_full_capped_v2_per_province,
            "train_full_capped_v2_max_major_ratio": args.train_full_capped_v2_max_major_ratio,
            "val_balanced_v2_per_province": args.val_balanced_v2_per_province,
            "test_balanced_v2_per_province": args.test_balanced_v2_per_province,
        },
        "source_counts": source_counts(
            ("train_source", train_rows),
            ("val_source", val_rows),
            ("test_source", test_rows),
        ),
        "train_full": summarize(train_rows),
        "train_balanced": summarize(train_bal),
        "val_full": summarize(val_rows),
        "val_balanced": summarize(val_bal),
        "test_full": summarize(test_rows),
        "test_balanced": summarize(test_bal),
        "train_balanced_v2": summarize(train_bal_v2),
        "train_full_capped_v2": summarize(train_full_capped_v2),
        "train_stagec_v2": summarize(train_stagec_v2),
        "val_balanced_v2": summarize(val_bal_v2),
        "test_balanced_v2": summarize(test_bal_v2),
        "overlap": leak_checks,
        "output_files": {k: str(v) for k, v in files.items()},
    }

    report_path = out_dir / "split_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
