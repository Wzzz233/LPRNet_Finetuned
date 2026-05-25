#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/wzzz/LPRNet")
for p in [ROOT / "src", ROOT / "src/evaluation", ROOT / "src/training", ROOT / "src/utils"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_lpr_detailed import decode_logits  # noqa: E402
from load_data import CHARS, read_ppm_p6_payload  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa: E402
from train_LPRNet import _select_family_logits_from_dict  # noqa: E402


def load_model(path, device):
    state = torch.load(str(path), map_location=device)
    net, _ = build_lprnet_multihead_from_state_dict(
        state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    return net.to(device).eval()


def read_dump_rows(dump_dir):
    idx = dump_dir / "index.csv"
    if idx.exists():
        with idx.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        out = []
        for i, r in enumerate(rows):
            sid = int(r.get("sample_id") or r.get("idx") or i)
            frame = int(r.get("frame_id") or r.get("frame") or 0)
            candidates = sorted(dump_dir.glob(f"ocrin_{sid:04d}_f*.ppm"))
            p = candidates[0] if candidates else None
            if p and p.exists():
                out.append({"path": p, "app_text": r.get("app_text", ""), "row": r})
        return out
    return [{"path": p, "app_text": "", "row": {}} for p in sorted(dump_dir.glob("ocrin_*.ppm"))]


def prepare(img):
    if img.shape[:2] != (24, 94):
        img = cv2.resize(img, (94, 24), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x = g3.astype(np.float32)
    x -= 127.5
    x *= 0.0078125
    return torch.from_numpy(np.transpose(x, (2, 0, 1))[None, ...]), g3


def greedy(logits):
    blank = len(CHARS) - 1
    out = []
    prev = None
    for t in range(logits.shape[1]):
        c = int(np.argmax(logits[:, t]))
        if c != blank and c != prev:
            out.append(CHARS[c])
        prev = c
    return "".join(out)


def eval_dir(net, dump_dir, limit, device):
    rows = read_dump_rows(dump_dir)[:limit]
    pred_rows = []
    with torch.no_grad():
        for r in rows:
            img = read_ppm_p6_payload(str(r["path"]))
            x, g3 = prepare(img)
            raw = net(x.to(device))
            logits = _select_family_logits_from_dict(raw, sample_families=["green8"]).detach().cpu().numpy()[0]
            ids = decode_logits(logits[None, ...], "family_aware_beam", 20, 12, sample_families=["green8"])[0]
            beam = "".join(CHARS[int(c)] for c in ids)
            pred_rows.append({
                "file": r["path"].name,
                "app_text": r["app_text"],
                "beam": beam,
                "greedy": greedy(logits),
                "mean": float(g3.mean()),
                "std": float(g3.std()),
                "min": int(g3.min()),
                "max": int(g3.max()),
                "channels_equal": bool(np.all(g3[..., 0] == g3[..., 1]) and np.all(g3[..., 0] == g3[..., 2])),
            })
    return pred_rows


def summarize(rows):
    return {
        "n": len(rows),
        "beam_top": dict(Counter(r["beam"] for r in rows).most_common(12)),
        "greedy_top": dict(Counter(r["greedy"] for r in rows).most_common(12)),
        "mean_avg": sum(r["mean"] for r in rows) / max(1, len(rows)),
        "max_avg": sum(r["max"] for r in rows) / max(1, len(rows)),
        "first_rows": rows[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "experiments/curriculum_gray3_stageB_B2D_paradigm3_progress/Final_LPRNet_model.pth"))
    ap.add_argument("--dump-dir", action="append", required=True)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = load_model(Path(args.model), device)
    out = {"model": args.model, "device": str(device), "dirs": {}}
    for d in args.dump_dir:
        rows = eval_dir(net, Path(d), args.limit, device)
        out["dirs"][d] = summarize(rows)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
