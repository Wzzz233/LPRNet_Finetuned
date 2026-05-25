from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from load_data import Box, parse_ccpd_quad_from_name
from .decode import decode_corner_heatmaps
from .geometry import build_patch_box_from_quad, gate_refined_quad, map_quad_from_patch
from .model import QuadHeatmapRefiner


def parse_quad_arg(text):
    vals = [float(v.strip()) for v in text.split(',')]
    if len(vals) != 8:
        raise ValueError('quad must be x1,y1,x2,y2,x3,y3,x4,y4')
    return np.asarray(vals, dtype=np.float32).reshape(4, 2)


def draw_quad(img, quad, color):
    pts = np.round(np.asarray(quad, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, 2)


def load_model(path, device):
    model = QuadHeatmapRefiner(pretrained=False).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt))
    model.eval()
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description='Run quad refiner on a single image.')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--quad', default='')
    ap.add_argument('--ccpd-quad-from-name', action='store_true')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--input-width', type=int, default=256)
    ap.add_argument('--input-height', type=int, default=128)
    ap.add_argument('--pad-x', type=float, default=0.20)
    ap.add_argument('--pad-y', type=float, default=0.25)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    coarse_quad = parse_quad_arg(args.quad) if args.quad else None
    if coarse_quad is None and args.ccpd_quad_from_name:
        coarse_quad = parse_ccpd_quad_from_name(Path(args.image).name)
    if coarse_quad is None:
        raise ValueError('provide --quad or --ccpd-quad-from-name')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_h, img_w = image.shape[:2]
    patch_box = build_patch_box_from_quad(coarse_quad, img_w=img_w, img_h=img_h, pad_x=args.pad_x, pad_y=args.pad_y)
    patch = image[patch_box.y1:patch_box.y2 + 1, patch_box.x1:patch_box.x2 + 1]
    patch = cv2.resize(patch, (args.input_width, args.input_height), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(patch.transpose(2, 0, 1)).float()[None] / 255.0
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    with torch.no_grad():
        outputs = model(x.to(device))
    heatmaps = torch.sigmoid(outputs['heatmaps'])[0].cpu().numpy()
    pred_patch, confs = decode_corner_heatmaps(heatmaps, in_w=args.input_width, in_h=args.input_height)
    pred_quad = map_quad_from_patch(pred_patch, patch_box, in_w=args.input_width, in_h=args.input_height)
    gate = gate_refined_quad(coarse_quad, pred_quad, confs, patch_diag=math.hypot(args.input_width, args.input_height))
    final_quad = pred_quad if gate.accepted else coarse_quad

    vis = image.copy()
    draw_quad(vis, coarse_quad, (0, 255, 255))
    draw_quad(vis, pred_quad, (255, 0, 0))
    draw_quad(vis, final_quad, (0, 255, 0))
    cv2.imwrite(str(out_dir / 'overlay.jpg'), vis)
    cv2.imwrite(str(out_dir / 'patch.jpg'), patch)
    payload = {
        'image': args.image,
        'coarse_quad': np.asarray(coarse_quad, dtype=float).round(3).tolist(),
        'pred_quad': np.asarray(pred_quad, dtype=float).round(3).tolist(),
        'final_quad': np.asarray(final_quad, dtype=float).round(3).tolist(),
        'corner_conf': [float(v) for v in confs],
        'gate': {'accepted': gate.accepted, 'reason': gate.reason, 'metrics': gate.metrics},
    }
    (out_dir / 'result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
