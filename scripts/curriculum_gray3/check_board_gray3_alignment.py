#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


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
        raise ValueError(f"unsupported max value in {path}: {maxv}")
    if i < n and blob[i] in b" \t\r\n":
        i += 1
    arr = np.frombuffer(blob[i:], dtype=np.uint8)
    if arr.size != w * h * 3:
        raise ValueError(f"bad payload size in {path}: {arr.size} != {w*h*3}")
    return arr.reshape(h, w, 3).copy()


def board_gray3_rgb(src_rgb):
    g = ((77 * src_rgb[..., 0].astype(np.uint16)
          + 150 * src_rgb[..., 1].astype(np.uint16)
          + 29 * src_rgb[..., 2].astype(np.uint16)) >> 8).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def resize_nn_board(src, dst_w, dst_h):
    sh, sw = src.shape[:2]
    ys = (np.arange(dst_h, dtype=np.int64) * sh) // dst_h
    xs = (np.arange(dst_w, dtype=np.int64) * sw) // dst_w
    return src[ys[:, None], xs[None, :], :]


def letterbox_board(src, dst_w=94, dst_h=24, pad=0):
    sh, sw = src.shape[:2]
    out = np.full((dst_h, dst_w, 3), pad, dtype=np.uint8)
    scale = min(dst_w / float(sw), dst_h / float(sh))
    scaled_w = int(sw * scale + 0.5)
    scaled_h = int(sh * scale + 0.5)
    scaled_w = max(1, min(dst_w, scaled_w))
    scaled_h = max(1, min(dst_h, scaled_h))
    off_x = (dst_w - scaled_w) // 2
    off_y = (dst_h - scaled_h) // 2
    resized = resize_nn_board(src, scaled_w, scaled_h)
    out[off_y:off_y + scaled_h, off_x:off_x + scaled_w] = resized
    return out


def make_from_crop_board(crop_rgb):
    return letterbox_board(board_gray3_rgb(crop_rgb), 94, 24, 0)


def make_from_crop_cv2_bgr(crop_rgb):
    # Simulates the common mistake: raw PPM RGB array treated as BGR by cv2.
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return letterbox_board(g3, 94, 24, 0)


def diff_stats(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "max_abs": int(d.max()),
        "mean_abs": float(d.mean()),
        "nonzero_px": int(np.count_nonzero(np.any(d != 0, axis=2))),
        "nonzero_chan": int(np.count_nonzero(d)),
    }


def read_index(dump_dir):
    with (dump_dir / "index.csv").open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def local_path(dump_dir, prefix, row):
    sid = int(row["sample_id"])
    frame = int(row["frame_id"])
    return dump_dir / f"{prefix}_{sid:04d}_f{frame:06d}.ppm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    rows = read_index(dump_dir)[:args.limit]
    per = []
    for row in rows:
        crop_p = local_path(dump_dir, "crop", row)
        ocr_p = local_path(dump_dir, "ocrin", row)
        if not crop_p.exists() or not ocr_p.exists():
            continue
        crop = read_ppm(crop_p)
        ocr = read_ppm(ocr_p)
        board = make_from_crop_board(crop)
        cv2_bgr = make_from_crop_cv2_bgr(crop)
        per.append({
            "sample_id": int(row["sample_id"]),
            "frame_id": int(row["frame_id"]),
            "app_text": row.get("app_text", ""),
            "crop_shape": list(crop.shape),
            "ocr_shape": list(ocr.shape),
            "board_formula": diff_stats(board, ocr),
            "cv2_bgr_on_raw_ppm": diff_stats(cv2_bgr, ocr),
            "ocr_channels_equal": bool(np.all(ocr[..., 0] == ocr[..., 1]) and np.all(ocr[..., 0] == ocr[..., 2])),
            "ocr_mean": float(ocr.mean()),
            "board_mean": float(board.mean()),
        })
    summary = {
        "dump_dir": str(dump_dir),
        "n": len(per),
        "board_formula_max_abs_max": max((r["board_formula"]["max_abs"] for r in per), default=None),
        "board_formula_mean_abs_avg": sum((r["board_formula"]["mean_abs"] for r in per), 0.0) / max(1, len(per)),
        "board_formula_nonzero_px_sum": sum((r["board_formula"]["nonzero_px"] for r in per), 0),
        "cv2_bgr_mean_abs_avg": sum((r["cv2_bgr_on_raw_ppm"]["mean_abs"] for r in per), 0.0) / max(1, len(per)),
        "first_rows": per[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
