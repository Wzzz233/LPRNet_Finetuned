#!/usr/bin/env python3
"""Shared image handling for the 6-class plate type classifier."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path("/home/wzzz/LPRNet")
IMG_W = 224
IMG_H = 72
CLASS_NAMES = ["blue", "green", "yellow", "police", "embassy", "other"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_quad(row: Dict[str, str]) -> Optional[np.ndarray]:
    pts = []
    for idx in range(1, 5):
        x = row.get(f"quad_{idx}x", "")
        y = row.get(f"quad_{idx}y", "")
        if x == "" or y == "":
            return None
        try:
            pts.append([float(x), float(y)])
        except ValueError:
            return None
    return np.asarray(pts, dtype=np.float32)


def order_quad(quad: np.ndarray) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = quad[np.argmin(s)]
    ordered[2] = quad[np.argmax(s)]
    ordered[1] = quad[np.argmin(d)]
    ordered[3] = quad[np.argmax(d)]
    return ordered


def resize_pad(img: np.ndarray, out_w: int = IMG_W, out_h: int = IMG_H) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    scale = min(out_w / w, out_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def load_plate_bgr(row: Dict[str, str], out_w: int = IMG_W, out_h: int = IMG_H) -> np.ndarray:
    path = resolve_path(row["img_path"])
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8) + 128

    crop_mode = row.get("crop_mode", "")
    quad = parse_quad(row)
    if crop_mode == "perspective_warp" and quad is not None:
        src = order_quad(quad)
        dst = np.asarray([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        mat = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, mat, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if crop_mode in ("board_fc224", "already_224x72"):
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    return resize_pad(img, out_w, out_h)


def plate_color_scores(plate_bgr: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = (s > 45) & (v > 35)
    blue = ((h > 90) & (h < 135) & valid).mean()
    green = ((h > 45) & (h < 95) & valid).mean()
    yellow = ((h > 14) & (h < 45) & valid).mean()
    dark = (v < 70).mean()
    bright = (v > 180).mean()
    return {
        "blue": float(blue),
        "green": float(green),
        "yellow": float(yellow),
        "dark": float(dark),
        "bright": float(bright),
        "v_mean": float(v.mean()),
        "s_mean": float(s.mean()),
    }


def apply_real_board_aug(plate_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Make synthetic special plates closer to the saved board fc224 dumps."""
    img = plate_bgr.astype(np.float32)

    contrast = rng.uniform(0.58, 1.05)
    bias = rng.uniform(22.0, 82.0)
    img = (img - 127.5) * contrast + 127.5 + bias
    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < 0.70:
        k = rng.choice([3, 3, 5])
        sigma = rng.uniform(0.45, 1.35)
        img = cv2.GaussianBlur(img, (k, k), sigma)

    if rng.random() < 0.65:
        scale = rng.uniform(0.42, 0.78)
        small = cv2.resize(img, (max(24, int(IMG_W * scale)), max(12, int(IMG_H * scale))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    if rng.random() < 0.55:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(0.55, 1.02)
        hsv[:, :, 2] *= rng.uniform(1.02, 1.42)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.45:
        noise = rng.uniform(2.0, 8.0)
        img = np.clip(img.astype(np.float32) + rng.normalvariate(0.0, noise), 0, 255).astype(np.uint8)

    if rng.random() < 0.75:
        quality = rng.randint(32, 76)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                img = dec

    return img


def bgr_to_tensor_array(plate_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1)).copy()

