#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path


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


def merge_unique_rows(base_rows, extra_rows, exclude_paths=None):
    merged = list(base_rows)
    seen = path_set(base_rows)
    exclude_paths = exclude_paths or set()
    added = 0
    skipped_dup = 0
    skipped_excluded = 0
    for rel_path, text in extra_rows:
        if rel_path in exclude_paths:
            skipped_excluded += 1
            continue
        if rel_path in seen:
            skipped_dup += 1
            continue
        merged.append((rel_path, text))
        seen.add(rel_path)
        added += 1
    return merged, {
        "added": added,
        "skipped_duplicate": skipped_dup,
        "skipped_excluded": skipped_excluded,
    }


def sample_normal_rows(pool_rows, keep_count, seed):
    if keep_count < 0:
        raise ValueError("keep_count must be >= 0")
    if keep_count > len(pool_rows):
        raise RuntimeError(f"normal pool not enough: need={keep_count}, have={len(pool_rows)}")
    rng = random.Random(seed)
    idx = list(range(len(pool_rows)))
    rng.shuffle(idx)
    picked = [pool_rows[i] for i in idx[:keep_count]]
    return picked


def build_mix(hard_rows, normal_pool_rows, hard_ratio, seed):
    if hard_ratio <= 0.0 or hard_ratio > 1.0:
        raise ValueError("hard_ratio must be in (0, 1]")
    hard_count = len(hard_rows)
    if hard_count <= 0:
        raise RuntimeError("hard train rows is empty")
    normal_keep = int(round(hard_count * (1.0 - hard_ratio) / hard_ratio))
    normal_rows = sample_normal_rows(normal_pool_rows, normal_keep, seed)
    merged = list(hard_rows) + list(normal_rows)
    rng = random.Random(seed + 13)
    rng.shuffle(merged)
    return merged, normal_rows


def ensure_no_overlap(train_rows, val_rows, test_rows, tag):
    train_set = path_set(train_rows)
    val_set = path_set(val_rows)
    test_set = path_set(test_rows)
    ov_train_val = len(train_set & val_set)
    ov_train_test = len(train_set & test_set)
    ov_val_test = len(val_set & test_set)
    if ov_train_val > 0 or ov_train_test > 0 or ov_val_test > 0:
        raise RuntimeError(
            f"[{tag}] leakage detected: train_val={ov_train_val} train_test={ov_train_test} val_test={ov_val_test}"
        )
    return {
        "train_val": ov_train_val,
        "train_test": ov_train_test,
        "val_test": ov_val_test,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build hard-tilt/normal mixed training labels with strict leakage checks.")
    parser.add_argument("--hard-train", required=True)
    parser.add_argument("--hard-val", required=True)
    parser.add_argument("--hard-test", required=True)
    parser.add_argument("--normal-train", required=True)
    parser.add_argument("--hardcase-train", default="", help="optional extra hard-case train labels to merge into hard train pool")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stagea-hard-ratio", type=float, default=0.70)
    parser.add_argument("--stageb-hard-ratio", type=float, default=0.85)
    parser.add_argument("--stagec-hard-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260318)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hard_train_base = read_label_rows(Path(args.hard_train))
    hard_val = read_label_rows(Path(args.hard_val))
    hard_test = read_label_rows(Path(args.hard_test))
    normal_train = read_label_rows(Path(args.normal_train))
    hardcase_train = []
    if args.hardcase_train:
        hardcase_path = Path(args.hardcase_train)
        if not hardcase_path.exists():
            raise FileNotFoundError(f"hardcase train labels not found: {hardcase_path}")
        hardcase_train = read_label_rows(hardcase_path)

    hard_all_non_train = path_set(hard_val) | path_set(hard_test)
    hard_train, hardcase_merge = merge_unique_rows(hard_train_base, hardcase_train, exclude_paths=hard_all_non_train)
    if len(hard_train) == 0:
        raise RuntimeError("hard train pool is empty after merge/filter")

    normal_pool = [row for row in normal_train if row[0] not in hard_all_non_train]
    if len(normal_pool) == 0:
        raise RuntimeError("normal pool is empty after filtering against hard val/test")

    mix_a, normal_a = build_mix(hard_train, normal_pool, args.stagea_hard_ratio, args.seed)
    mix_b, normal_b = build_mix(hard_train, normal_pool, args.stageb_hard_ratio, args.seed + 7)
    mix_c, normal_c = build_mix(hard_train, normal_pool, args.stagec_hard_ratio, args.seed + 17)

    overlap_a = ensure_no_overlap(mix_a, hard_val, hard_test, "stageA")
    overlap_b = ensure_no_overlap(mix_b, hard_val, hard_test, "stageB")
    overlap_c = ensure_no_overlap(mix_c, hard_val, hard_test, "stageC")

    stagea_path = out_dir / "train_mix_stageA_labels.txt"
    stageb_path = out_dir / "train_mix_stageB_labels.txt"
    stagec_path = out_dir / "train_mix_stageC_labels.txt"
    val_path = out_dir / "val_hard_labels.txt"
    test_path = out_dir / "test_hard_labels.txt"

    write_label_rows(stagea_path, mix_a)
    write_label_rows(stageb_path, mix_b)
    write_label_rows(stagec_path, mix_c)
    write_label_rows(val_path, hard_val)
    write_label_rows(test_path, hard_test)

    report = {
        "hard_pool": {
            "base_hard_train_count": len(hard_train_base),
            "hardcase_train_count": len(hardcase_train),
            "hard_train_merged_count": len(hard_train),
            "hardcase_merge": hardcase_merge,
        },
        "stageA": {
            "hard_ratio_target": args.stagea_hard_ratio,
            "train_total": len(mix_a),
            "hard_count": len(hard_train),
            "normal_count": len(normal_a),
            "overlap": overlap_a,
        },
        "stageB": {
            "hard_ratio_target": args.stageb_hard_ratio,
            "train_total": len(mix_b),
            "hard_count": len(hard_train),
            "normal_count": len(normal_b),
            "overlap": overlap_b,
        },
        "stageC": {
            "hard_ratio_target": args.stagec_hard_ratio,
            "train_total": len(mix_c),
            "hard_count": len(hard_train),
            "normal_count": len(normal_c),
            "overlap": overlap_c,
        },
        "val_total": len(hard_val),
        "test_total": len(hard_test),
        "paths": {
            "train_stageA": str(stagea_path),
            "train_stageB": str(stageb_path),
            "train_stageC": str(stagec_path),
            "val_hard": str(val_path),
            "test_hard": str(test_path),
        },
    }
    (out_dir / "mix_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
