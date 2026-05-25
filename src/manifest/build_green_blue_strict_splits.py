#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict green+blue training splits with leakage checks and province balancing."
    )
    parser.add_argument("--green-train", required=True)
    parser.add_argument("--green-val", required=True)
    parser.add_argument("--green-test", required=True)
    parser.add_argument(
        "--blue-train",
        required=True,
        help="Comma-separated blue train label txt paths.",
    )
    parser.add_argument("--blue-val-normal", required=True)
    parser.add_argument("--blue-test-normal", required=True)
    parser.add_argument("--blue-val-hard", required=True)
    parser.add_argument("--blue-test-hard", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260320)

    parser.add_argument("--stagea-green-ratio", type=float, default=0.60)
    parser.add_argument("--stageb-green-ratio", type=float, default=0.80)
    parser.add_argument("--stagec-green-ratio", type=float, default=0.70)

    parser.add_argument("--green-train-cap-per-province", type=int, default=900)
    parser.add_argument("--blue-train-cap-per-province", type=int, default=1400)
    parser.add_argument("--green-train-max-major-ratio", type=float, default=0.40)
    parser.add_argument("--blue-train-max-major-ratio", type=float, default=0.60)
    parser.add_argument("--green-val-balanced-per-province", type=int, default=80)
    parser.add_argument("--green-test-balanced-per-province", type=int, default=220)
    parser.add_argument("--blue-val-gate-per-province", type=int, default=120)
    return parser.parse_args()


def csv_paths(s: str):
    out = []
    for item in s.split(","):
        item = item.strip()
        if item:
            out.append(Path(item))
    return out


def read_label_rows(paths):
    rows = []
    seen = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"label file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
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


def path_set(rows):
    return {p for p, _ in rows}


def filter_out_paths(rows, exclude_paths):
    return [row for row in rows if row[0] not in exclude_paths]


def sample_by_province_cap(rows, cap, seed):
    if cap <= 0:
        return list(rows)
    buckets = defaultdict(list)
    for rel_path, text in rows:
        province = text[0] if text else "?"
        buckets[province].append((rel_path, text))
    rng = random.Random(seed)
    out = []
    for province in sorted(buckets):
        chunk = buckets[province]
        rng.shuffle(chunk)
        out.extend(chunk[:cap])
    rng.shuffle(out)
    return out


def limit_major_ratio(rows, max_major_ratio, seed):
    if max_major_ratio <= 0.0 or max_major_ratio >= 1.0:
        return list(rows)
    if len(rows) <= 1:
        return list(rows)

    buckets = defaultdict(list)
    for rel_path, text in rows:
        province = text[0] if text else "?"
        buckets[province].append((rel_path, text))
    major_province = max(buckets.keys(), key=lambda k: len(buckets[k]))
    major_rows = buckets[major_province]
    major_count = len(major_rows)
    total = len(rows)
    other_count = total - major_count
    if other_count <= 0:
        return list(rows)

    allowed_major = int((other_count * max_major_ratio) / (1.0 - max_major_ratio))
    if allowed_major < 1:
        allowed_major = 1
    if major_count <= allowed_major:
        return list(rows)

    rng = random.Random(seed)
    rng.shuffle(major_rows)
    kept_major = major_rows[:allowed_major]
    out = []
    for province, chunk in buckets.items():
        if province == major_province:
            out.extend(kept_major)
        else:
            out.extend(chunk)
    rng.shuffle(out)
    return out


def build_stage_mix(green_rows, blue_rows, green_ratio, seed):
    if green_ratio <= 0.0 or green_ratio > 1.0:
        raise ValueError(f"invalid green ratio: {green_ratio}")
    if len(green_rows) == 0:
        raise RuntimeError("green train pool is empty")
    blue_need = int(round(len(green_rows) * (1.0 - green_ratio) / green_ratio))
    if blue_need > len(blue_rows):
        raise RuntimeError(
            f"blue train pool not enough: need={blue_need}, have={len(blue_rows)}, "
            f"green_count={len(green_rows)}, green_ratio={green_ratio}"
        )
    rng = random.Random(seed)
    idx = list(range(len(blue_rows)))
    rng.shuffle(idx)
    blue_pick = [blue_rows[i] for i in idx[:blue_need]]
    merged = list(green_rows) + blue_pick
    rng.shuffle(merged)
    return merged, {"green_count": len(green_rows), "blue_count": blue_need, "total": len(merged)}


def dedupe_rows(rows):
    out = []
    seen = set()
    for row in rows:
        if row[0] in seen:
            continue
        seen.add(row[0])
        out.append(row)
    return out


def pairwise_overlap(name_to_rows):
    keys = sorted(name_to_rows)
    report = {}
    for a, b in itertools.combinations(keys, 2):
        ov = len(path_set(name_to_rows[a]) & path_set(name_to_rows[b]))
        report[f"{a}__{b}"] = ov
    return report


def check_train_eval_leak(train_name_to_rows, eval_name_to_rows):
    leaks = {}
    for train_name, train_rows in sorted(train_name_to_rows.items()):
        train_paths = path_set(train_rows)
        for eval_name, eval_rows in sorted(eval_name_to_rows.items()):
            ov = len(train_paths & path_set(eval_rows))
            leaks[f"{train_name}__{eval_name}"] = ov
    return leaks


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    green_train = read_label_rows([Path(args.green_train)])
    green_val = read_label_rows([Path(args.green_val)])
    green_test = read_label_rows([Path(args.green_test)])

    blue_train = read_label_rows(csv_paths(args.blue_train))
    blue_val_normal = read_label_rows([Path(args.blue_val_normal)])
    blue_test_normal = read_label_rows([Path(args.blue_test_normal)])
    blue_val_hard = read_label_rows([Path(args.blue_val_hard)])
    blue_test_hard = read_label_rows([Path(args.blue_test_hard)])

    eval_holdout_paths = (
        path_set(green_val)
        | path_set(green_test)
        | path_set(blue_val_normal)
        | path_set(blue_test_normal)
        | path_set(blue_val_hard)
        | path_set(blue_test_hard)
    )
    green_train_filtered = filter_out_paths(green_train, eval_holdout_paths)
    blue_train_filtered = filter_out_paths(blue_train, eval_holdout_paths)

    green_train_bal = sample_by_province_cap(
        green_train_filtered, args.green_train_cap_per_province, args.seed + 101
    )
    green_train_bal = limit_major_ratio(
        green_train_bal, args.green_train_max_major_ratio, args.seed + 107
    )
    blue_train_bal = sample_by_province_cap(
        blue_train_filtered, args.blue_train_cap_per_province, args.seed + 102
    )
    blue_train_bal = limit_major_ratio(
        blue_train_bal, args.blue_train_max_major_ratio, args.seed + 108
    )
    if len(green_train_bal) == 0:
        raise RuntimeError("empty green train pool after filtering/balancing")
    if len(blue_train_bal) == 0:
        raise RuntimeError("empty blue train pool after filtering/balancing")

    green_val_bal = sample_by_province_cap(
        green_val, args.green_val_balanced_per_province, args.seed + 103
    )
    green_test_bal = sample_by_province_cap(
        green_test, args.green_test_balanced_per_province, args.seed + 104
    )
    blue_val_normal_bal = sample_by_province_cap(
        blue_val_normal, args.blue_val_gate_per_province, args.seed + 105
    )
    blue_val_hard_bal = sample_by_province_cap(
        blue_val_hard, args.blue_val_gate_per_province, args.seed + 106
    )

    val_gate = dedupe_rows(green_val_bal + blue_val_normal_bal + blue_val_hard_bal)

    train_a, stage_a_meta = build_stage_mix(
        green_train_bal, blue_train_bal, args.stagea_green_ratio, args.seed + 201
    )
    train_b, stage_b_meta = build_stage_mix(
        green_train_bal, blue_train_bal, args.stageb_green_ratio, args.seed + 202
    )
    train_c, stage_c_meta = build_stage_mix(
        green_train_bal, blue_train_bal, args.stagec_green_ratio, args.seed + 203
    )

    files = {
        "train_green_balanced": out_dir / "train_green_balanced_labels.txt",
        "train_blue_balanced": out_dir / "train_blue_balanced_labels.txt",
        "train_stageA": out_dir / "train_mix_stageA_labels.txt",
        "train_stageB": out_dir / "train_mix_stageB_labels.txt",
        "train_stageC": out_dir / "train_mix_stageC_labels.txt",
        "val_gate": out_dir / "val_gate_labels.txt",
        "val_green_full": out_dir / "val_green_full_labels.txt",
        "test_green_full": out_dir / "test_green_full_labels.txt",
        "val_green_balanced": out_dir / "val_green_balanced_labels.txt",
        "test_green_balanced": out_dir / "test_green_balanced_labels.txt",
        "val_blue_normal": out_dir / "val_blue_normal_labels.txt",
        "test_blue_normal": out_dir / "test_blue_normal_labels.txt",
        "val_blue_hard": out_dir / "val_blue_hard_labels.txt",
        "test_blue_hard": out_dir / "test_blue_hard_labels.txt",
    }

    write_label_rows(files["train_green_balanced"], green_train_bal)
    write_label_rows(files["train_blue_balanced"], blue_train_bal)
    write_label_rows(files["train_stageA"], train_a)
    write_label_rows(files["train_stageB"], train_b)
    write_label_rows(files["train_stageC"], train_c)
    write_label_rows(files["val_gate"], val_gate)
    write_label_rows(files["val_green_full"], green_val)
    write_label_rows(files["test_green_full"], green_test)
    write_label_rows(files["val_green_balanced"], green_val_bal)
    write_label_rows(files["test_green_balanced"], green_test_bal)
    write_label_rows(files["val_blue_normal"], blue_val_normal)
    write_label_rows(files["test_blue_normal"], blue_test_normal)
    write_label_rows(files["val_blue_hard"], blue_val_hard)
    write_label_rows(files["test_blue_hard"], blue_test_hard)

    overlap_report = check_train_eval_leak(
        {
            "train_stageA": train_a,
            "train_stageB": train_b,
            "train_stageC": train_c,
        },
        {
            "val_gate": val_gate,
            "green_val_full": green_val,
            "green_test_full": green_test,
            "blue_val_normal": blue_val_normal,
            "blue_test_normal": blue_test_normal,
            "blue_val_hard": blue_val_hard,
            "blue_test_hard": blue_test_hard,
        },
    )
    leak_items = {k: v for k, v in overlap_report.items() if v > 0}
    if leak_items:
        raise RuntimeError(f"leakage detected: {leak_items}")

    report = {
        "seed": args.seed,
        "stage_green_ratios": {
            "stageA": args.stagea_green_ratio,
            "stageB": args.stageb_green_ratio,
            "stageC": args.stagec_green_ratio,
        },
        "caps": {
            "green_train_cap_per_province": args.green_train_cap_per_province,
            "blue_train_cap_per_province": args.blue_train_cap_per_province,
            "green_train_max_major_ratio": args.green_train_max_major_ratio,
            "blue_train_max_major_ratio": args.blue_train_max_major_ratio,
            "green_val_balanced_per_province": args.green_val_balanced_per_province,
            "green_test_balanced_per_province": args.green_test_balanced_per_province,
            "blue_val_gate_per_province": args.blue_val_gate_per_province,
        },
        "pools": {
            "green_train_raw": summarize(green_train),
            "green_train_filtered": summarize(green_train_filtered),
            "green_train_balanced": summarize(green_train_bal),
            "blue_train_raw": summarize(blue_train),
            "blue_train_filtered": summarize(blue_train_filtered),
            "blue_train_balanced": summarize(blue_train_bal),
        },
        "eval_sets": {
            "val_gate": summarize(val_gate),
            "val_green_full": summarize(green_val),
            "test_green_full": summarize(green_test),
            "val_green_balanced": summarize(green_val_bal),
            "test_green_balanced": summarize(green_test_bal),
            "val_blue_normal": summarize(blue_val_normal),
            "test_blue_normal": summarize(blue_test_normal),
            "val_blue_hard": summarize(blue_val_hard),
            "test_blue_hard": summarize(blue_test_hard),
        },
        "train_stage": {
            "stageA": {"mix_meta": stage_a_meta, "summary": summarize(train_a)},
            "stageB": {"mix_meta": stage_b_meta, "summary": summarize(train_b)},
            "stageC": {"mix_meta": stage_c_meta, "summary": summarize(train_c)},
        },
        "overlap": overlap_report,
        "output_files": {k: str(v) for k, v in files.items()},
    }

    report_path = out_dir / "split_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
