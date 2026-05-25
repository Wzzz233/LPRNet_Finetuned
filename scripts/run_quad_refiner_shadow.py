#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_data import Box, prepare_board_ocr_input_from_quad_bgr888
from quad_refiner.dataset import load_jsonl_records
from quad_refiner.decode import decode_corner_heatmaps
from quad_refiner.geometry import build_patch_box_from_quad, gate_refined_quad, map_quad_from_patch
from quad_refiner.model import QuadHeatmapRefiner


def draw_quad(img, quad, color):
    pts = np.round(np.asarray(quad, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, 2)


def load_model(path, device):
    model = QuadHeatmapRefiner(pretrained=False).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt))
    model.eval()
    return model


def prep_ocr(image, quad):
    ocr, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(image, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0)
    return ocr


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description='Shadow visualization for quad refiner.')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--records-jsonl', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--input-width', type=int, default=256)
    ap.add_argument('--input-height', type=int, default=128)
    ap.add_argument('--pad-x', type=float, default=0.20)
    ap.add_argument('--pad-y', type=float, default=0.25)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    records = load_jsonl_records(args.records_jsonl)[: args.limit]
    summary = []

    for idx, rec in enumerate(records, 1):
        image = cv2.imread(rec['image_path'])
        if image is None:
            continue
        gt_quad = np.asarray(rec['gt_quad'], dtype=np.float32)
        coarse_quad = np.asarray(rec.get('coarse_quad', rec['gt_quad']), dtype=np.float32)
        img_h, img_w = image.shape[:2]
        patch_box = build_patch_box_from_quad(coarse_quad, img_w=img_w, img_h=img_h, pad_x=args.pad_x, pad_y=args.pad_y)
        patch = image[patch_box.y1:patch_box.y2 + 1, patch_box.x1:patch_box.x2 + 1]
        patch_resized = cv2.resize(patch, (args.input_width, args.input_height), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(patch_resized.transpose(2, 0, 1)).float()[None] / 255.0
        with torch.no_grad():
            out = model(x.to(device))
        heatmaps = torch.sigmoid(out['heatmaps'])[0].cpu().numpy()
        pred_patch, confs = decode_corner_heatmaps(heatmaps, in_w=args.input_width, in_h=args.input_height)
        pred_quad = map_quad_from_patch(pred_patch, patch_box, in_w=args.input_width, in_h=args.input_height)
        gate = gate_refined_quad(coarse_quad, pred_quad, confs, patch_diag=math.hypot(args.input_width, args.input_height))
        final_quad = pred_quad if gate.accepted else coarse_quad

        overlay = image.copy()
        draw_quad(overlay, coarse_quad, (0, 255, 255))
        draw_quad(overlay, gt_quad, (0, 165, 255))
        draw_quad(overlay, pred_quad, (255, 0, 0))
        draw_quad(overlay, final_quad, (0, 255, 0))

        gt_ocr = prep_ocr(image, gt_quad)
        coarse_ocr = prep_ocr(image, coarse_quad)
        final_ocr = prep_ocr(image, final_quad)
        strip = np.concatenate([coarse_ocr, final_ocr, gt_ocr], axis=1)

        stem = f'{idx:04d}_{Path(rec["image_path"]).stem}'
        cv2.imwrite(str(out_dir / f'{stem}_overlay.jpg'), overlay)
        cv2.imwrite(str(out_dir / f'{stem}_patch.jpg'), patch_resized)
        cv2.imwrite(str(out_dir / f'{stem}_ocr_triptych.jpg'), strip)
        summary.append({
            'sample_id': rec['sample_id'],
            'accepted': gate.accepted,
            'reason': gate.reason,
            'coarse_corner_err': float(np.linalg.norm(coarse_quad - gt_quad, axis=1).mean()),
            'final_corner_err': float(np.linalg.norm(final_quad - gt_quad, axis=1).mean()),
        })

    (out_dir / 'shadow_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'count': len(summary), 'output_dir': str(out_dir)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
