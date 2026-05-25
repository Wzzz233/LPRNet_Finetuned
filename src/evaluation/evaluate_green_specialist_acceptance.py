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
    parser = argparse.ArgumentParser(description="Acceptance for green specialist model.")
    parser.add_argument("--baseline-green-full", required=True)
    parser.add_argument("--new-green-full", required=True)
    parser.add_argument("--baseline-green-balanced", required=True)
    parser.add_argument("--new-green-balanced", required=True)
    parser.add_argument("--green-full-gain-min-pp", type=float, default=0.20)
    parser.add_argument("--green-balanced-gain-min-pp", type=float, default=0.20)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    b_f = read_json(args.baseline_green_full)
    n_f = read_json(args.new_green_full)
    b_b = read_json(args.baseline_green_balanced)
    n_b = read_json(args.new_green_balanced)

    b_f_exact = float(b_f.get("exact_plate_acc", 0.0))
    n_f_exact = float(n_f.get("exact_plate_acc", 0.0))
    b_b_exact = float(b_b.get("exact_plate_acc", 0.0))
    n_b_exact = float(n_b.get("exact_plate_acc", 0.0))

    full_gain_pp = (n_f_exact - b_f_exact) * 100.0
    bal_gain_pp = (n_b_exact - b_b_exact) * 100.0

    pass_full = full_gain_pp >= args.green_full_gain_min_pp
    pass_bal = bal_gain_pp >= args.green_balanced_gain_min_pp
    passed = pass_full and pass_bal

    report = {
        "baseline_green_full_exact": b_f_exact,
        "new_green_full_exact": n_f_exact,
        "green_full_gain_pp": full_gain_pp,
        "green_full_gain_min_pp": args.green_full_gain_min_pp,
        "baseline_green_balanced_exact": b_b_exact,
        "new_green_balanced_exact": n_b_exact,
        "green_balanced_gain_pp": bal_gain_pp,
        "green_balanced_gain_min_pp": args.green_balanced_gain_min_pp,
        "passed_full": pass_full,
        "passed_balanced": pass_bal,
        "passed": passed,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
