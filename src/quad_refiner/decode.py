from __future__ import annotations

import numpy as np


def _refine_peak(channel: np.ndarray, x: int, y: int):
    h, w = channel.shape
    x0 = max(0, x - 1)
    x1 = min(w, x + 2)
    y0 = max(0, y - 1)
    y1 = min(h, y + 2)
    window = channel[y0:y1, x0:x1].astype(np.float32)
    total = float(window.sum())
    if total <= 1e-6:
        return float(x), float(y)
    xs = np.arange(x0, x1, dtype=np.float32)[None, :]
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    rx = float((window * xs).sum() / total)
    ry = float((window * ys).sum() / total)
    return rx, ry


def decode_corner_heatmaps(heatmaps, in_w: int, in_h: int):
    """Decode 4-corner heatmaps to image-space corner points using 3x3 area-weighted centroid."""
    hm = np.asarray(heatmaps, dtype=np.float32)
    if hm.ndim != 3 or hm.shape[0] != 4:
        raise ValueError(f'expected heatmaps shape (4,H,W), got {hm.shape}')
    _, out_h, out_w = hm.shape
    sx = 0.0 if out_w <= 1 or in_w <= 1 else float(in_w - 1) / float(out_w - 1)
    sy = 0.0 if out_h <= 1 or in_h <= 1 else float(in_h - 1) / float(out_h - 1)
    pts = np.zeros((4, 2), dtype=np.float32)
    confs = []
    for i in range(4):
        ch = hm[i]
        idx = int(np.argmax(ch))
        y, x = divmod(idx, out_w)
        rx, ry = _refine_peak(ch, x, y)
        pts[i, 0] = rx * sx
        pts[i, 1] = ry * sy
        confs.append(float(ch[y, x]))
    return pts, confs


def decode_corner_heatmaps_with_offset(heatmaps, offsets, in_w: int, in_h: int):
    """Decode heatmaps with local offset refinement for sub-pixel accuracy.

    Steps:
    1. Find argmax peak in each heatmap channel
    2. Sample learned offset (dx, dy) at the peak location
    3. Add offset to integer peak position for fractional refinement
    4. Scale to input image space

    Args:
        heatmaps: (4, H, W) - sigmoid heatmaps
        offsets: (8, H, W) - predicted local offsets (dx1,dy1, dx2,dy2, dx3,dy3, dx4,dy4)
        in_w: input image width (256)
        in_h: input image height (128)

    Returns:
        pts: (4, 2) corner points in input coordinate space
        confs: [4] corner confidences
    """
    hm = np.asarray(heatmaps, dtype=np.float32)
    off = np.asarray(offsets, dtype=np.float32)
    if hm.ndim != 3 or hm.shape[0] != 4:
        raise ValueError(f'expected heatmaps shape (4,H,W), got {hm.shape}')
    if off.ndim != 3 or off.shape[0] != 8:
        raise ValueError(f'expected offsets shape (8,H,W), got {off.shape}')
    _, out_h, out_w = hm.shape
    sx = 0.0 if out_w <= 1 or in_w <= 1 else float(in_w - 1) / float(out_w - 1)
    sy = 0.0 if out_h <= 1 or in_h <= 1 else float(in_h - 1) / float(out_h - 1)

    pts = np.zeros((4, 2), dtype=np.float32)
    confs = []
    for i in range(4):
        ch = hm[i]
        idx = int(np.argmax(ch))
        y, x = divmod(idx, out_w)

        # Sample offset at peak location
        dx = float(off[i * 2, y, x])
        dy = float(off[i * 2 + 1, y, x])

        # Refined position in output space: integer peak + fractional offset
        rx = float(x) + dx
        ry = float(y) + dy

        pts[i, 0] = rx * sx
        pts[i, 1] = ry * sy
        confs.append(float(ch[y, x]))

    return pts, confs
