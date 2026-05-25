from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from load_data import prepare_board_ocr_input_from_quad_bgr888
from .dataset import QuadRefinerDataset
from .decode import decode_corner_heatmaps
from .geometry import build_patch_box_from_quad, gate_refined_quad, map_quad_from_patch
from .model import QuadHeatmapRefiner
from .train import collate_batch, move_batch_to_device


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    # Detect if checkpoint has offset head
    has_offset = any('offset_head' in k for k in state_dict.keys())
    model = QuadHeatmapRefiner(pretrained=False, enable_offset=has_offset).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, has_offset


def corner_error(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32), axis=1).mean())


def _prep_ocr(image, quad, ocr_w, ocr_h):
    prepared, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        image,
        quad,
        ocr_w,
        ocr_h,
        'letterbox',
        'nn',
        'none',
        'bgr',
        quad_pad_ratio=0.0,
    )
    return prepared


def build_argparser():
    ap = argparse.ArgumentParser(description='Evaluate OBB quad post-refiner.')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--records-jsonl', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--input-width', type=int, default=256)
    ap.add_argument('--input-height', type=int, default=128)
    ap.add_argument('--output-width', type=int, default=64)
    ap.add_argument('--output-height', type=int, default=32)
    ap.add_argument('--pad-x', type=float, default=0.20)
    ap.add_argument('--pad-y', type=float, default=0.25)
    ap.add_argument('--ocr-width', type=int, default=94)
    ap.add_argument('--ocr-height', type=int, default=24)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ds = QuadRefinerDataset(args.records_jsonl, input_size=(args.input_width, args.input_height), output_size=(args.output_width, args.output_height), pad_x=args.pad_x, pad_y=args.pad_y, enable_offset=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)
    model, has_offset = load_model(args.checkpoint, device)

    rows = []
    coarse_errs = []
    refined_errs = []
    coarse_mads = []
    refined_mads = []
    accepted = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        with torch.no_grad():
            outputs = model(batch['image'])
        heatmaps = torch.sigmoid(outputs['heatmaps']).cpu().numpy()
        patch_boxes = batch['patch_box'].cpu().numpy()
        coarse_quads = batch['coarse_quad'].cpu().numpy()
        gt_quads = batch['gt_quad'].cpu().numpy()
        gt_points = batch['gt_points'].cpu().numpy()
        for i in range(heatmaps.shape[0]):
            if has_offset and 'offsets' in outputs:
                offset_np = outputs['offsets'].cpu().numpy()[i]
                from .decode import decode_corner_heatmaps_with_offset
                pred_patch, confs = decode_corner_heatmaps_with_offset(
                    heatmaps[i], offset_np, in_w=args.input_width, in_h=args.input_height)
            else:
                pred_patch, confs = decode_corner_heatmaps(heatmaps[i], in_w=args.input_width, in_h=args.input_height)
            patch_box_arr = patch_boxes[i]
            from load_data import Box
            patch_box = Box(int(round(patch_box_arr[0])), int(round(patch_box_arr[1])), int(round(patch_box_arr[2])), int(round(patch_box_arr[3])))
            pred_quad = map_quad_from_patch(pred_patch, patch_box, in_w=args.input_width, in_h=args.input_height)
            gate = gate_refined_quad(coarse_quads[i], pred_quad, confs, patch_diag=math.hypot(args.input_width, args.input_height))
            final_quad = pred_quad if gate.accepted else coarse_quads[i]
            if gate.accepted:
                accepted += 1

            coarse_err = corner_error(coarse_quads[i], gt_quads[i])
            refined_err = corner_error(final_quad, gt_quads[i])
            coarse_errs.append(coarse_err)
            refined_errs.append(refined_err)

            image = cv2.imread(raw_batch['image_path'][i])
            coarse_mad = None
            refined_mad = None
            if image is not None:
                gt_ocr = _prep_ocr(image, gt_quads[i], args.ocr_width, args.ocr_height)
                coarse_ocr = _prep_ocr(image, coarse_quads[i], args.ocr_width, args.ocr_height)
                refined_ocr = _prep_ocr(image, final_quad, args.ocr_width, args.ocr_height)
                coarse_mad = float(np.abs(coarse_ocr.astype(np.float32) - gt_ocr.astype(np.float32)).mean())
                refined_mad = float(np.abs(refined_ocr.astype(np.float32) - gt_ocr.astype(np.float32)).mean())
                coarse_mads.append(coarse_mad)
                refined_mads.append(refined_mad)

            rows.append({
                'sample_id': raw_batch['sample_id'][i],
                'source_name': raw_batch['source_name'][i],
                'text': raw_batch['text'][i],
                'accepted': int(gate.accepted),
                'gate_reason': gate.reason,
                'coarse_corner_err': coarse_err,
                'refined_corner_err': refined_err,
                'coarse_warp_mad': coarse_mad,
                'refined_warp_mad': refined_mad,
            })

    with (out_dir / 'eval_rows.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['sample_id'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        'count': len(rows),
        'accept_rate': accepted / max(len(rows), 1),
        'coarse_corner_err_mean': float(np.mean(coarse_errs)) if coarse_errs else None,
        'refined_corner_err_mean': float(np.mean(refined_errs)) if refined_errs else None,
        'coarse_warp_mad_mean': float(np.mean(coarse_mads)) if coarse_mads else None,
        'refined_warp_mad_mean': float(np.mean(refined_mads)) if refined_mads else None,
    }
    (out_dir / 'eval_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
