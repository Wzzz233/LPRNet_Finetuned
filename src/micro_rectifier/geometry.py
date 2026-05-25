from __future__ import annotations

import cv2
import numpy as np


def apply_parametric_warp_bgr(image, dx, dy, sx, sy, shx):
    h, w = image.shape[:2]
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    m = np.array([
        [sx, shx, dx + (1.0 - sx) * cx - shx * cy],
        [0.0, sy, dy + (1.0 - sy) * cy],
    ], dtype=np.float32)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
