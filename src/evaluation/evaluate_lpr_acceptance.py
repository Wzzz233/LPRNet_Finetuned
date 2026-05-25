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


def rel_gain(new_v, base_v):
    if base_v <= 0:
        return 1.0 if new_v > 0 else 0.0
    return (new_v - base_v) / base_v


def main():
    ap = argparse.ArgumentParser(description="Check acceptance gates for tilt-focused OCR fine-tune.")
    ap.add_argument("--baseline-hard-test", required=True)
    ap.add_argument("--new-hard-test", required=True)
    ap.add_argument("--baseline-normal-test", required=True)
    ap.add_argument("--new-normal-test", required=True)
    ap.add_argument("--hard-rel-gain-min", type=float, default=0.10)
    ap.add_argument("--normal-drop-max-pp", type=float, default=1.5)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    b_h = read_json(args.baseline_hard_test)
    n_h = read_json(args.new_hard_test)
    b_n = read_json(args.baseline_normal_test)
    n_n = read_json(args.new_normal_test)

    b_h_exact = float(b_h.get("exact_plate_acc", 0.0))
    n_h_exact = float(n_h.get("exact_plate_acc", 0.0))
    b_n_exact = float(b_n.get("exact_plate_acc", 0.0))
    n_n_exact = float(n_n.get("exact_plate_acc", 0.0))

    hard_gain = rel_gain(n_h_exact, b_h_exact)
    normal_drop_pp = (b_n_exact - n_n_exact) * 100.0

    pass_hard = hard_gain >= args.hard_rel_gain_min
    pass_normal = normal_drop_pp <= args.normal_drop_max_pp
    passed = pass_hard and pass_normal

    report = {
        "baseline_hard_exact": b_h_exact,
        "new_hard_exact": n_h_exact,
        "hard_rel_gain": hard_gain,
        "hard_rel_gain_min": args.hard_rel_gain_min,
        "baseline_normal_exact": b_n_exact,
        "new_normal_exact": n_n_exact,
        "normal_drop_pp": normal_drop_pp,
        "normal_drop_max_pp": args.normal_drop_max_pp,
        "pass_hard_gain": pass_hard,
        "pass_normal_drop": pass_normal,
        "passed": passed,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
