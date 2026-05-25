from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from load_data import order_quad_points, parse_ccpd_quad_from_name
from .geometry import build_patch_box_from_quad, map_quad_to_patch


SPECIAL_HINT_CHARS = set('警学挂使领港澳')


def classify_family_from_text(text: str):
    t = (text or '').strip().upper()
    if len(t) == 8:
        if len(t) > 2 and t[2] in {'D', 'F'}:
            return 'green8', 'green_small'
        if t[-1] in {'D', 'F'}:
            return 'green8', 'green_large'
        return 'green8', 'green'
    if len(t) == 7 and not any(ch in SPECIAL_HINT_CHARS for ch in t):
        return 'normal7', 'blue'
    return 'special', 'special'


def _record(sample_id, image_path, split, source_name, gt_quad, coarse_quad=None, text='', family='', sub_type=''):
    gt = order_quad_points(np.asarray(gt_quad, dtype=np.float32)).tolist()
    coarse = gt if coarse_quad is None else order_quad_points(np.asarray(coarse_quad, dtype=np.float32)).tolist()
    return {
        'sample_id': str(sample_id),
        'image_path': str(image_path),
        'split': str(split),
        'source_name': str(source_name),
        'text': str(text),
        'family': str(family),
        'sub_type': str(sub_type),
        'gt_quad': gt,
        'coarse_quad': coarse,
    }


def build_ccpd_record(image_path, split: str, source_name: str, coarse_quad=None, text: str = '', family: str = '', sub_type: str = ''):
    image_path = Path(image_path)
    quad = parse_ccpd_quad_from_name(image_path.name)
    if quad is None:
        raise ValueError(f'failed to parse CCPD quad from {image_path}')
    if not family:
        family, sub_type = classify_family_from_text(text)
    return _record(
        sample_id=f'{source_name}:{image_path.stem}',
        image_path=image_path,
        split=split,
        source_name=source_name,
        gt_quad=quad,
        coarse_quad=coarse_quad,
        text=text,
        family=family,
        sub_type=sub_type,
    )


def build_crpd_records(image_path, label_path, split: str, source_name: str, coarse_quads: dict | None = None):
    image_path = Path(image_path)
    label_path = Path(label_path)
    records = []
    with label_path.open('r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            coords = [float(v) for v in parts[:8]]
            text = parts[9]
            family, sub_type = classify_family_from_text(text)
            coarse = None
            if coarse_quads:
                coarse = coarse_quads.get(idx) or coarse_quads.get(f'{image_path.stem}_obj{idx}')
            records.append(
                _record(
                    sample_id=f'{source_name}:{image_path.stem}_obj{idx}',
                    image_path=image_path,
                    split=split,
                    source_name=source_name,
                    gt_quad=np.asarray(coords, dtype=np.float32).reshape(4, 2),
                    coarse_quad=coarse,
                    text=text,
                    family=family,
                    sub_type=sub_type,
                )
            )
    return records


def write_jsonl_records(records: Iterable[dict], output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl_records(path_or_records):
    if isinstance(path_or_records, (str, os.PathLike, Path)):
        records = []
        with Path(path_or_records).open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    return list(path_or_records)


def _scale_points(points, in_w, in_h, out_w, out_h):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2).copy()
    sx = 0.0 if in_w <= 1 or out_w <= 1 else float(out_w - 1) / float(in_w - 1)
    sy = 0.0 if in_h <= 1 or out_h <= 1 else float(out_h - 1) / float(in_h - 1)
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return pts


def _gaussian_2d(height, width, center_x, center_y, sigma):
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    return np.exp(-((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2.0 * sigma * sigma))


def generate_corner_heatmaps(quad_patch, in_w: int, in_h: int, out_w: int = 64, out_h: int = 32, sigma: float = 1.5):
    pts = _scale_points(quad_patch, in_w, in_h, out_w, out_h)
    heatmaps = np.zeros((4, out_h, out_w), dtype=np.float32)
    for i, (x, y) in enumerate(pts):
        heatmaps[i] = np.maximum(heatmaps[i], _gaussian_2d(out_h, out_w, float(x), float(y), float(sigma)))
    return np.clip(heatmaps, 0.0, 1.0)


def generate_corner_offsets(quad_patch, in_w: int, in_h: int, out_w: int = 64, out_h: int = 32):
    """Generate offset targets: for each corner, compute fractional offset relative to nearest integer grid point.
    Output shape: (8, out_h, out_w) — dx1,dy1, dx2,dy2, dx3,dy3, dx4,dy4. Only populated at GT corner locations.
    """
    pts = _scale_points(quad_patch, in_w, in_h, out_w, out_h)
    offsets = np.zeros((8, out_h, out_w), dtype=np.float32)
    offset_masks = np.zeros((4, out_h, out_w), dtype=np.float32)  # per-corner mask
    for i, (x, y) in enumerate(pts):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        xi = max(0, min(out_w - 1, xi))
        yi = max(0, min(out_h - 1, yi))
        offsets[i * 2, yi, xi] = float(x) - xi
        offsets[i * 2 + 1, yi, xi] = float(y) - yi
        offset_masks[i, yi, xi] = 1.0
    return offsets, offset_masks


def rasterize_quad_mask(quad_patch, in_w: int, in_h: int, out_w: int = 64, out_h: int = 32):
    pts = _scale_points(quad_patch, in_w, in_h, out_w, out_h)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask[None, ...].astype(np.float32)


class QuadRefinerDataset(Dataset):
    def __init__(
        self,
        records,
        input_size=(256, 128),
        output_size=(64, 32),
        pad_x: float = 0.20,
        pad_y: float = 0.25,
        normalize: bool = True,
        enable_offset: bool = False,
    ):
        self.records = load_jsonl_records(records)
        self.input_w, self.input_h = int(input_size[0]), int(input_size[1])
        self.output_w, self.output_h = int(output_size[0]), int(output_size[1])
        self.pad_x = float(pad_x)
        self.pad_y = float(pad_y)
        self.normalize = bool(normalize)
        self.enable_offset = bool(enable_offset)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image_path = row['image_path']
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f'failed to read image: {image_path}')
        img_h, img_w = img.shape[:2]
        gt_quad = order_quad_points(np.asarray(row['gt_quad'], dtype=np.float32))
        coarse_quad = order_quad_points(np.asarray(row.get('coarse_quad', row['gt_quad']), dtype=np.float32))
        patch_box = build_patch_box_from_quad(coarse_quad, img_w=img_w, img_h=img_h, pad_x=self.pad_x, pad_y=self.pad_y)
        patch = img[patch_box.y1:patch_box.y2 + 1, patch_box.x1:patch_box.x2 + 1]
        patch = cv2.resize(patch, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        gt_patch = map_quad_to_patch(gt_quad, patch_box, out_w=self.input_w, out_h=self.input_h)
        coarse_patch = map_quad_to_patch(coarse_quad, patch_box, out_w=self.input_w, out_h=self.input_h)
        heatmaps = generate_corner_heatmaps(gt_patch, self.input_w, self.input_h, self.output_w, self.output_h)
        mask = rasterize_quad_mask(gt_patch, self.input_w, self.input_h, self.output_w, self.output_h)
        gt_points_out = _scale_points(gt_patch, self.input_w, self.input_h, self.output_w, self.output_h)

        patch = patch.astype(np.float32)
        if self.normalize:
            patch /= 255.0
        patch = patch.transpose(2, 0, 1)
        out = {
            'image': torch.from_numpy(patch).float(),
            'heatmaps': torch.from_numpy(heatmaps).float(),
            'mask': torch.from_numpy(mask).float(),
            'gt_points': torch.from_numpy(gt_patch).float(),
            'gt_points_out': torch.from_numpy(gt_points_out).float(),
            'coarse_points': torch.from_numpy(coarse_patch).float(),
            'patch_box': torch.tensor([patch_box.x1, patch_box.y1, patch_box.x2, patch_box.y2], dtype=torch.float32),
            'gt_quad': torch.from_numpy(gt_quad).float(),
            'coarse_quad': torch.from_numpy(coarse_quad).float(),
            'sample_id': row['sample_id'],
            'image_path': image_path,
            'source_name': row.get('source_name', ''),
            'text': row.get('text', ''),
        }
        if self.enable_offset:
            offset_targets, offset_masks = generate_corner_offsets(gt_patch, self.input_w, self.input_h, self.output_w, self.output_h)
            out['offset_targets'] = torch.from_numpy(offset_targets).float()
            out['offset_masks'] = torch.from_numpy(offset_masks).float()
        return out
