#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


def read_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"metrics not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Acceptance gate for green uplift + blue no-regression.")
    parser.add_argument("--baseline-blue-hard", required=True)
    parser.add_argument("--new-blue-hard", required=True)
    parser.add_argument("--baseline-blue-normal", required=True)
    parser.add_argument("--new-blue-normal", required=True)
    parser.add_argument("--baseline-green-full", required=True)
    parser.add_argument("--new-green-full", required=True)
    parser.add_argument("--baseline-green-balanced", required=True)
    parser.add_argument("--new-green-balanced", required=True)
    parser.add_argument("--blue-drop-max-pp", type=float, default=0.0)
    parser.add_argument("--green-gain-min-pp", type=float, default=0.01)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    b_blue_h = float(read_json(args.baseline_blue_hard).get("exact_plate_acc", 0.0))
    n_blue_h = float(read_json(args.new_blue_hard).get("exact_plate_acc", 0.0))
    b_blue_n = float(read_json(args.baseline_blue_normal).get("exact_plate_acc", 0.0))
    n_blue_n = float(read_json(args.new_blue_normal).get("exact_plate_acc", 0.0))
    b_green_f = float(read_json(args.baseline_green_full).get("exact_plate_acc", 0.0))
    n_green_f = float(read_json(args.new_green_full).get("exact_plate_acc", 0.0))
    b_green_b = float(read_json(args.baseline_green_balanced).get("exact_plate_acc", 0.0))
    n_green_b = float(read_json(args.new_green_balanced).get("exact_plate_acc", 0.0))

    blue_hard_drop_pp = (b_blue_h - n_blue_h) * 100.0
    blue_normal_drop_pp = (b_blue_n - n_blue_n) * 100.0
    green_full_gain_pp = (n_green_f - b_green_f) * 100.0
    green_balanced_gain_pp = (n_green_b - b_green_b) * 100.0

    pass_blue_hard = blue_hard_drop_pp <= args.blue_drop_max_pp
    pass_blue_normal = blue_normal_drop_pp <= args.blue_drop_max_pp
    pass_green_full = green_full_gain_pp >= args.green_gain_min_pp
    pass_green_balanced = green_balanced_gain_pp >= args.green_gain_min_pp
    passed = pass_blue_hard and pass_blue_normal and pass_green_full and pass_green_balanced

    report = {
        "baseline_blue_hard_exact": b_blue_h,
        "new_blue_hard_exact": n_blue_h,
        "blue_hard_drop_pp": blue_hard_drop_pp,
        "baseline_blue_normal_exact": b_blue_n,
        "new_blue_normal_exact": n_blue_n,
        "blue_normal_drop_pp": blue_normal_drop_pp,
        "blue_drop_max_pp": args.blue_drop_max_pp,
        "baseline_green_full_exact": b_green_f,
        "new_green_full_exact": n_green_f,
        "green_full_gain_pp": green_full_gain_pp,
        "baseline_green_balanced_exact": b_green_b,
        "new_green_balanced_exact": n_green_b,
        "green_balanced_gain_pp": green_balanced_gain_pp,
        "green_gain_min_pp": args.green_gain_min_pp,
        "pass_blue_hard": pass_blue_hard,
        "pass_blue_normal": pass_blue_normal,
        "pass_green_full": pass_green_full,
        "pass_green_balanced": pass_green_balanced,
        "passed": passed,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
