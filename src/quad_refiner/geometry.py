from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from load_data import Box, clamp_box, order_quad_points, quad_to_box


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    metrics: dict = field(default_factory=dict)


def _as_quad(quad) -> np.ndarray:
    arr = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    return order_quad_points(arr)


def quad_center(quad) -> np.ndarray:
    q = _as_quad(quad)
    return np.mean(q, axis=0)


def quad_area(quad) -> float:
    q = _as_quad(quad)
    x = q[:, 0]
    y = q[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def edge_lengths(quad) -> np.ndarray:
    q = _as_quad(quad)
    return np.asarray([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)], dtype=np.float32)


def is_convex_quad(quad, eps: float = 1e-5) -> bool:
    q = _as_quad(quad)
    crosses = []
    for i in range(4):
        a = q[(i + 1) % 4] - q[i]
        b = q[(i + 2) % 4] - q[(i + 1) % 4]
        cross = float(a[0] * b[1] - a[1] * b[0])
        crosses.append(cross)
    pos = any(c > eps for c in crosses)
    neg = any(c < -eps for c in crosses)
    return not (pos and neg)


def build_patch_box_from_quad(quad, img_w: int, img_h: int, pad_x: float = 0.20, pad_y: float = 0.25) -> Box:
    box = quad_to_box(quad, img_w=img_w, img_h=img_h)
    ex = int(round(box.w * float(pad_x)))
    ey = int(round(box.h * float(pad_y)))
    return clamp_box(Box(box.x1 - ex, box.y1 - ey, box.x2 + ex, box.y2 + ey), img_w, img_h)


def _span_scale(src_span: int, dst_span: int) -> float:
    if src_span <= 1 or dst_span <= 1:
        return 0.0
    return float(dst_span - 1) / float(src_span - 1)


def map_quad_to_patch(quad, patch_box: Box, out_w: int, out_h: int) -> np.ndarray:
    q = _as_quad(quad)
    sx = _span_scale(patch_box.w, out_w)
    sy = _span_scale(patch_box.h, out_h)
    out = np.zeros((4, 2), dtype=np.float32)
    out[:, 0] = (q[:, 0] - float(patch_box.x1)) * sx
    out[:, 1] = (q[:, 1] - float(patch_box.y1)) * sy
    if out_w > 0:
        out[:, 0] = np.clip(out[:, 0], 0.0, float(out_w - 1))
    if out_h > 0:
        out[:, 1] = np.clip(out[:, 1], 0.0, float(out_h - 1))
    return _as_quad(out)


def map_quad_from_patch(quad_patch, patch_box: Box, in_w: int, in_h: int) -> np.ndarray:
    q = _as_quad(quad_patch)
    sx = _span_scale(patch_box.w, in_w)
    sy = _span_scale(patch_box.h, in_h)
    out = np.zeros((4, 2), dtype=np.float32)
    if sx == 0.0:
        out[:, 0] = float(patch_box.x1)
    else:
        out[:, 0] = q[:, 0] / sx + float(patch_box.x1)
    if sy == 0.0:
        out[:, 1] = float(patch_box.y1)
    else:
        out[:, 1] = q[:, 1] / sy + float(patch_box.y1)
    return _as_quad(out)


def gate_refined_quad(
    coarse_quad,
    refined_quad,
    corner_conf: Iterable[float] | None,
    patch_diag: float,
    min_corner_conf: float = 0.20,
    min_area_ratio: float = 0.65,
    max_area_ratio: float = 1.45,
    max_center_shift_ratio: float = 0.20,
    max_corner_shift_ratio: float = 0.18,
    max_edge_ratio_ratio: float = 2.2,
) -> GateDecision:
    coarse = _as_quad(coarse_quad)
    refined = _as_quad(refined_quad)
    conf = np.asarray(list(corner_conf or [1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
    if conf.size != 4:
        conf = np.ones(4, dtype=np.float32)

    coarse_area = max(quad_area(coarse), 1e-6)
    refined_area = quad_area(refined)
    area_ratio = refined_area / coarse_area
    center_shift = float(np.linalg.norm(quad_center(refined) - quad_center(coarse)))
    corner_shift = np.linalg.norm(refined - coarse, axis=1)
    refined_edges = edge_lengths(refined)
    coarse_edges = np.maximum(edge_lengths(coarse), 1e-6)
    edge_change_ratio = refined_edges / coarse_edges
    edge_ratio_sanity = float(max(edge_change_ratio.max(), 1.0 / max(edge_change_ratio.min(), 1e-6)))

    metrics = {
        'min_corner_conf': float(conf.min()),
        'area_ratio': float(area_ratio),
        'center_shift': float(center_shift),
        'max_corner_shift': float(corner_shift.max()),
        'edge_ratio_sanity': float(edge_ratio_sanity),
    }

    if float(conf.min()) < float(min_corner_conf):
        return GateDecision(False, 'low_corner_conf', metrics)
    if not is_convex_quad(refined):
        return GateDecision(False, 'non_convex_quad', metrics)
    if area_ratio < float(min_area_ratio) or area_ratio > float(max_area_ratio):
        return GateDecision(False, 'area_ratio_out_of_range', metrics)
    if center_shift > float(max_center_shift_ratio) * float(max(patch_diag, 1e-6)):
        return GateDecision(False, 'center_shift_too_large', metrics)
    if float(corner_shift.max()) > float(max_corner_shift_ratio) * float(max(patch_diag, 1e-6)):
        return GateDecision(False, 'corner_shift_too_large', metrics)
    if edge_ratio_sanity > float(max_edge_ratio_ratio):
        return GateDecision(False, 'edge_ratio_abnormal', metrics)
    if refined_area < 4.0:
        return GateDecision(False, 'degenerate_area', metrics)
    return GateDecision(True, 'accepted', metrics)
