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
from load_data import CHARS  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa: E402
from train_LPRNet import _select_family_logits_from_dict  # noqa: E402


def read_ppm(path):
    with open(path, "rb") as f:
        blob = f.read()
    if not blob.startswith(b"P6"):
        raise ValueError(f"not P6 ppm: {path}")
    i = 2
    toks = []
    n = len(blob)
    while len(toks) < 3:
        while i < n and blob[i] in b" \t\r\n":
            i += 1
        if i < n and blob[i] == ord("#"):
            while i < n and blob[i] not in b"\r\n":
                i += 1
            continue
        j = i
        while j < n and blob[j] not in b" \t\r\n":
            j += 1
        toks.append(blob[i:j].decode("ascii"))
        i = j
    w, h, maxv = map(int, toks)
    if maxv != 255:
        raise ValueError(f"unsupported ppm maxv={maxv}: {path}")
    if i < n and blob[i] in b" \t\r\n":
        i += 1
    arr = np.frombuffer(blob[i:], dtype=np.uint8)
    if arr.size != w * h * 3:
        raise ValueError(f"bad ppm payload {path}: got {arr.size}, expected {w*h*3}")
    return arr.reshape(h, w, 3).copy()


def read_rows(dump_dir, limit):
    rows = []
    idx = dump_dir / "index.csv"
    with idx.open("r", encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if limit and len(rows) >= limit:
                break
            sid = int(row.get("sample_id") or i)
            frame = int(row.get("frame_id") or 0)
            p = dump_dir / f"ocrin_{sid:04d}_f{frame:06d}.ppm"
            if not p.exists():
                cand = sorted(dump_dir.glob(f"ocrin_{sid:04d}_f*.ppm"))
                p = cand[0] if cand else p
            if p.exists():
                rows.append(
                    {
                        "sample_id": sid,
                        "frame_id": frame,
                        "path": str(p),
                        "app_text": (row.get("app_text") or "").strip(),
                        "app_occ_ratio": row.get("app_occ_ratio") or "",
                    }
                )
    return rows


def load_model(path, device):
    state = torch.load(str(path), map_location=device)
    net, _ = build_lprnet_multihead_from_state_dict(
        state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    return net.to(device).eval()


def gray_channel(img):
    if img.shape[:2] != (24, 94):
        img = cv2.resize(img, (94, 24), interpolation=cv2.INTER_NEAREST)
    # The board gray3 dump already has equal channels. Mean is robust if a file is not perfectly equal.
    return img.astype(np.float32).mean(axis=2)


def valid_mask(g):
    # Keep letterbox padding fixed. This also avoids stretching pure black borders.
    return g > 0


def to_gray3_uint8(g):
    u = np.clip(g, 0, 255).astype(np.uint8)
    return np.repeat(u[..., None], 3, axis=2)


def affine(g, mask, gain, bias):
    out = g.copy()
    out[mask] = out[mask] * gain + bias
    return out


def meanstd(g, mask, target_mean, target_std, max_value):
    out = g.copy()
    vals = out[mask]
    if vals.size < 8:
        return out
    mu = float(vals.mean())
    sd = float(vals.std())
    if sd < 1e-3:
        sd = 1.0
    out[mask] = (vals - mu) * (target_std / sd) + target_mean
    out[mask] = np.clip(out[mask], 0, max_value)
    return out


def pct_stretch(g, mask, lo_pct, hi_pct, out_lo, out_hi):
    out = g.copy()
    vals = out[mask]
    if vals.size < 8:
        return out
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if hi <= lo + 1e-3:
        return out
    out[mask] = (vals - lo) * ((out_hi - out_lo) / (hi - lo)) + out_lo
    out[mask] = np.clip(out[mask], 0, out_hi)
    return out


def gamma_lift(g, mask, gamma, max_value):
    out = g.copy()
    vals = np.clip(out[mask] / max_value, 0.0, 1.0)
    out[mask] = (vals ** gamma) * max_value
    return out


def clahe_lift(g, mask, clip_limit, tile):
    u = np.clip(g, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    c = clahe.apply(u).astype(np.float32)
    out = g.copy()
    out[mask] = c[mask]
    return out


def sharpen(g, mask, amount):
    blur = cv2.GaussianBlur(g, (0, 0), 0.8)
    out = g + amount * (g - blur)
    keep = g.copy()
    keep[mask] = out[mask]
    return keep


def build_configs():
    configs = [("identity", {})]
    for gain in [1.2, 1.5, 1.8, 2.0, 2.4, 2.8]:
        for bias in [0, 8, 16, 24, 32, 48]:
            configs.append((f"affine_g{gain}_b{bias}", {"type": "affine", "gain": gain, "bias": bias}))
    for tm in [96, 112, 128, 144]:
        for ts in [28, 36, 44, 52]:
            for mx in [180, 220, 255]:
                configs.append((f"meanstd_m{tm}_s{ts}_max{mx}", {"type": "meanstd", "target_mean": tm, "target_std": ts, "max_value": mx}))
    for lo, hi in [(1, 99), (2, 98), (5, 95)]:
        for out_hi in [160, 190, 220, 255]:
            configs.append((f"pct{lo}_{hi}_hi{out_hi}", {"type": "pct", "lo_pct": lo, "hi_pct": hi, "out_lo": 0, "out_hi": out_hi}))
    for gamma in [0.45, 0.6, 0.75, 0.9]:
        for mx in [160, 190, 220, 255]:
            configs.append((f"gamma{gamma}_max{mx}", {"type": "gamma", "gamma": gamma, "max_value": mx}))
    for clip in [1.5, 2.0, 3.0]:
        for tile in [4, 8]:
            configs.append((f"clahe_c{clip}_t{tile}", {"type": "clahe", "clip_limit": clip, "tile": tile}))
    # Conservative composed candidates: normalize first, then recover a little edge contrast.
    for tm, ts, mx, amount in [(112, 36, 220, 0.25), (128, 44, 220, 0.25), (128, 44, 255, 0.35)]:
        configs.append(
            (
                f"meanstd_m{tm}_s{ts}_max{mx}_sharp{amount}",
                {"type": "meanstd_sharp", "target_mean": tm, "target_std": ts, "max_value": mx, "amount": amount},
            )
        )
    return configs


def apply_config(g, cfg):
    mask = valid_mask(g)
    t = cfg.get("type", "identity")
    if t == "identity":
        return g.copy()
    if t == "affine":
        return affine(g, mask, cfg["gain"], cfg["bias"])
    if t == "meanstd":
        return meanstd(g, mask, cfg["target_mean"], cfg["target_std"], cfg["max_value"])
    if t == "pct":
        return pct_stretch(g, mask, cfg["lo_pct"], cfg["hi_pct"], cfg["out_lo"], cfg["out_hi"])
    if t == "gamma":
        return gamma_lift(g, mask, cfg["gamma"], cfg["max_value"])
    if t == "clahe":
        return clahe_lift(g, mask, cfg["clip_limit"], cfg["tile"])
    if t == "meanstd_sharp":
        out = meanstd(g, mask, cfg["target_mean"], cfg["target_std"], cfg["max_value"])
        return sharpen(out, mask, cfg["amount"])
    raise ValueError(f"unknown config type: {t}")


def make_tensor(images):
    arr = np.stack(images, axis=0).astype(np.float32)
    arr = (arr - 127.5) * 0.0078125
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr)


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


def run_config(net, device, rows, gray_images, name, cfg, batch_size, target_texts):
    proc = []
    stats = []
    for g in gray_images:
        pg = apply_config(g, cfg)
        u = to_gray3_uint8(pg)
        m = valid_mask(pg)
        vals = pg[m]
        stats.append(
            {
                "mean": float(vals.mean()) if vals.size else 0.0,
                "std": float(vals.std()) if vals.size else 0.0,
                "max": float(vals.max()) if vals.size else 0.0,
                "pos_ratio_norm": float(np.mean(((pg - 127.5) * 0.0078125) > 0.0)),
            }
        )
        proc.append(u)
    preds = []
    with torch.no_grad():
        for i in range(0, len(proc), batch_size):
            x = make_tensor(proc[i : i + batch_size]).to(device)
            raw = net(x)
            logits = _select_family_logits_from_dict(
                raw, sample_families=["green8"] * x.shape[0]
            ).detach().cpu().numpy()
            ids = decode_logits(logits, "family_aware_beam", 20, 12, sample_families=["green8"] * x.shape[0])
            for j, one in enumerate(logits):
                beam = "".join(CHARS[int(c)] for c in ids[j])
                preds.append((beam, greedy(one)))
    detail = []
    for r, (beam, gr), st in zip(rows, preds, stats):
        rec = {
            **r,
            "config": name,
            "beam": beam,
            "greedy": gr,
            "proc_mean": f"{st['mean']:.4f}",
            "proc_std": f"{st['std']:.4f}",
            "proc_max": f"{st['max']:.2f}",
            "proc_pos_ratio_norm": f"{st['pos_ratio_norm']:.6f}",
        }
        for tgt in target_texts:
            rec[f"hit_{tgt}"] = int(beam == tgt or gr == tgt)
        detail.append(rec)
    beams = [r["beam"] for r in detail]
    greedies = [r["greedy"] for r in detail]
    summary = {
        "config": name,
        "n": len(detail),
        "nonempty_beam": sum(1 for x in beams if x),
        "nonempty_greedy": sum(1 for x in greedies if x),
        "avg_beam_len": sum(len(x) for x in beams) / max(1, len(beams)),
        "avg_greedy_len": sum(len(x) for x in greedies) / max(1, len(greedies)),
        "beam_top": dict(Counter(beams).most_common(8)),
        "greedy_top": dict(Counter(greedies).most_common(8)),
        "same_as_app_text_beam": sum(1 for r in detail if r["app_text"] and r["beam"] == r["app_text"]),
        "same_as_app_text_greedy": sum(1 for r in detail if r["app_text"] and r["greedy"] == r["app_text"]),
        "proc_mean_avg": sum(float(r["proc_mean"]) for r in detail) / max(1, len(detail)),
        "proc_std_avg": sum(float(r["proc_std"]) for r in detail) / max(1, len(detail)),
        "proc_max_avg": sum(float(r["proc_max"]) for r in detail) / max(1, len(detail)),
        "proc_pos_ratio_avg": sum(float(r["proc_pos_ratio_norm"]) for r in detail) / max(1, len(detail)),
    }
    for tgt in target_texts:
        summary[f"hit_{tgt}"] = sum(int(r[f"hit_{tgt}"]) for r in detail)
    return summary, detail


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/gray3_ocr_demp")
    ap.add_argument("--model", default=str(ROOT / "experiments/curriculum_gray3_stageB_B2D_paradigm3_progress/Final_LPRNet_model.pth"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports/ocrin_preproc_sweep_gray3_ocr_demp"))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--target-text", action="append", default=[])
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(dump_dir, args.limit)
    gray_images = [gray_channel(read_ppm(r["path"])) for r in rows]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = load_model(Path(args.model), device)

    all_summary = []
    all_details = []
    for name, cfg in build_configs():
        summary, details = run_config(net, device, rows, gray_images, name, cfg, args.batch_size, args.target_text)
        all_summary.append(summary)
        all_details.extend(details)

    # Ranking without ground truth: prefer non-empty, fuller plate length, and avoid a single collapsed prediction.
    def rank_key(s):
        top_count = max(s["beam_top"].values()) if s["beam_top"] else 0
        target_hits = sum(s.get(f"hit_{t}", 0) for t in args.target_text)
        return (
            target_hits,
            s["same_as_app_text_beam"] + s["same_as_app_text_greedy"],
            s["nonempty_beam"],
            s["avg_beam_len"],
            -top_count,
        )

    ranked = sorted(all_summary, key=rank_key, reverse=True)
    write_csv(out_dir / "summary.csv", ranked)
    write_csv(out_dir / "details.csv", all_details)
    report = {
        "dump_dir": str(dump_dir),
        "model": args.model,
        "device": str(device),
        "n_samples": len(rows),
        "target_text": args.target_text,
        "top": ranked[: args.top_k],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
