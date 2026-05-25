#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


def read_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze baseline weights and evaluation protocol into a single JSON record.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--train-entry", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--eval-json", action="append", default=[])
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    weight_path = Path(args.weights)
    if not weight_path.exists():
        raise FileNotFoundError(f"weights not found: {weight_path}")

    eval_bundle = {}
    for item in args.eval_json:
        if "=" not in item:
            raise ValueError(f"--eval-json expects alias=path, got: {item}")
        alias, path = item.split("=", 1)
        eval_bundle[alias] = read_json(path)

    report = {
        "name": args.name,
        "role": args.role,
        "weights": str(weight_path.resolve()),
        "train_entry": args.train_entry,
        "notes": args.notes,
        "evaluations": eval_bundle,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
