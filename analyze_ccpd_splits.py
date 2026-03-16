#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from pathlib import Path


def read_labels(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, text = line.split(maxsplit=1)
            rows.append((rel_path, text))
    return rows


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


def main():
    parser = argparse.ArgumentParser(description="Analyze train/val/test CCPD label split statistics.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    train_rows = read_labels(Path(args.train))
    val_rows = read_labels(Path(args.val))
    test_rows = read_labels(Path(args.test))

    train_set = {p for p, _ in train_rows}
    val_set = {p for p, _ in val_rows}
    test_set = {p for p, _ in test_rows}

    report = {
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "test": summarize(test_rows),
        "overlap": {
            "train_val": len(train_set & val_set),
            "train_test": len(train_set & test_set),
            "val_test": len(val_set & test_set),
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
