#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
for p in [ROOT / 'src', ROOT / 'src' / 'evaluation', ROOT / 'src' / 'training', ROOT / 'src' / 'utils']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import CHARS, prepare_board_ocr_input_bgr888
from eval_lpr_detailed import decode_logits
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from train_LPRNet import forward_family_logits


def apply_fixed_cut(image: np.ndarray, cut_px: int) -> np.ndarray:
    h, w = image.shape[:2]
    cut_px = max(0, min(int(cut_px), w - 2))
    if cut_px == 0:
        return image.copy()
    cropped = image[:, cut_px:, :]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_content_cut(image: np.ndarray, cut_px: int) -> np.ndarray:
    return apply_fixed_cut(image, cut_px)


def apply_x_remap(image: np.ndarray, left_ratio_in: float, left_ratio_out: float) -> np.ndarray:
    h, w = image.shape[:2]
    src = image.astype(np.float32)
    xs = np.arange(w, dtype=np.float32)
    xin_max = max(1.0, (w - 1) * float(left_ratio_in))
    xout_max = max(1.0, (w - 1) * float(left_ratio_out))
    xmap = np.empty_like(xs)
    for i, x in enumerate(xs):
        if x <= xout_max:
            t = x / max(xout_max, 1e-6)
            xmap[i] = t * xin_max
        else:
            t = (x - xout_max) / max((w - 1) - xout_max, 1e-6)
            xmap[i] = xin_max + t * ((w - 1) - xin_max)
    x0 = np.floor(xmap).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    x0 = np.clip(x0, 0, w - 1)
    a = (xmap - x0).reshape(1, w, 1)
    out = (1.0 - a) * src[:, x0, :] + a * src[:, x1, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def load_model(model_path: Path, device: torch.device):
    state = torch.load(str(model_path), map_location=device)
    net, _ = build_lprnet_multihead_from_state_dict(state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device)
    net.eval()
    return net


def infer_text(net, image_bgr_ocrin: np.ndarray, device: torch.device) -> str:
    image = image_bgr_ocrin.astype(np.float32)
    image -= 127.5
    image *= 0.0078125
    image = np.transpose(image, (2, 0, 1))[None, ...]
    images = torch.from_numpy(image).float().to(device)
    families = ['green8']
    with torch.no_grad():
        logits = forward_family_logits(net, images, sample_families=families).detach().cpu().numpy()
        preds = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=families)
    pred = preds[0]
    if isinstance(pred, list):
        pred = ''.join(CHARS[int(x)] for x in pred)
    return pred


def build_variants(ocrin: np.ndarray, fixed2_cut_px: int, fixed4_cut_px: int, content_cut_px: int):
    fixed2 = apply_fixed_cut(ocrin, fixed2_cut_px)
    fixed4 = apply_fixed_cut(ocrin, fixed4_cut_px)
    content = apply_content_cut(ocrin, content_cut_px)
    cvx1_a = apply_x_remap(fixed2, left_ratio_in=0.22, left_ratio_out=0.26)
    cvx1_b = apply_x_remap(fixed2, left_ratio_in=0.22, left_ratio_out=0.28)
    cvx1_c = apply_x_remap(fixed2, left_ratio_in=0.22, left_ratio_out=0.30)
    return {
        'baseline': ocrin,
        'fixed2': fixed2,
        'fixed4': fixed4,
        'content': content,
        'cvx1_a': cvx1_a,
        'cvx1_b': cvx1_b,
        'cvx1_c': cvx1_c,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster2-csv', required=True)
    ap.add_argument('--manifest-json', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    manifest = {item['id']: item for item in json.loads(Path(args.manifest_json).read_text(encoding='utf-8'))}
    rows = []
    with Path(args.cluster2_csv).open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            img_path = Path(row.get('local_ocrin_path') or row.get('ocrin_path') or '')
            sid = row.get('sample_id', '').strip()
            if sid.isdigit():
                sid = f'ocrin_{int(sid):04d}_f{int(row.get("frame_id", 0)):06d}'
            elif not sid:
                sid = img_path.stem
            if sid and img_path.exists() and sid in manifest:
                rows.append((sid, row, img_path, manifest[sid]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = load_model(Path(args.model), device)

    results = {name: {'count': 0, 'exact': 0, 'first': 0, 'pred_counter': {}, 'details': []} for name in ['baseline', 'fixed2', 'fixed4', 'content', 'cvx1_a', 'cvx1_b', 'cvx1_c']}

    thumbs = []
    meta = []
    for idx, (sid, row, img_path, cfg) in enumerate(rows):
        ocrin = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        variants = build_variants(ocrin, cfg['fixed2_cut_px'], cfg['fixed4_cut_px'], cfg['content_cut_px'])
        gt = row['gt_text']
        pred_map = {}
        for name, img in variants.items():
            pred = infer_text(net, prepare_board_ocr_input_bgr888(img, 94, 24, 'letterbox', 'nn', 'none', 'bgr')[0], device)
            pred_map[name] = pred
            res = results[name]
            res['count'] += 1
            res['exact'] += int(pred == gt)
            res['first'] += int(pred[:1] == gt[:1])
            res['pred_counter'][pred] = res['pred_counter'].get(pred, 0) + 1
            res['details'].append({'sample_id': sid, 'gt_text': gt, 'pred_text': pred})

        lines = []
        for name in ['baseline', 'fixed2', 'fixed4', 'content', 'cvx1_a', 'cvx1_b', 'cvx1_c']:
            img = prepare_board_ocr_input_bgr888(variants[name], 94, 24, 'letterbox', 'nn', 'none', 'bgr')[0]
            canvas = cv2.resize(img, (360, 96), interpolation=cv2.INTER_NEAREST)
            cv2.putText(canvas, f'{name}: {pred_map[name]}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 1, cv2.LINE_AA)
            lines.append(canvas)
        header = np.full((96, 360, 3), 255, dtype=np.uint8)
        cv2.putText(header, f'{idx} id={sid} gt={gt}', (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 1, cv2.LINE_AA)
        panel = np.vstack([header] + lines)
        panel_path = out_dir / f'{idx:02d}_{sid}.jpg'
        cv2.imwrite(str(panel_path), panel)
        thumbs.append(cv2.resize(panel, (280, 620), interpolation=cv2.INTER_AREA))
        meta.append({'idx': idx, 'sample_id': sid, 'gt_text': gt, 'preds': pred_map, 'panel_path': str(panel_path)})

    cols = 3
    cell_w, cell_h = 280, 620
    if thumbs:
        rows_n = (len(thumbs) + cols - 1) // cols
        sheet = np.full((rows_n * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
        for i, img in enumerate(thumbs):
            r = i // cols
            c = i % cols
            sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = img
    else:
        sheet = np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)
        cv2.putText(sheet, 'no matched samples', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    contact_path = out_dir / 'cluster2_cvx1_contact.jpg'
    cv2.imwrite(str(contact_path), sheet)

    summary = {'model': args.model, 'contact_sheet': str(contact_path), 'meta': meta}
    for name, res in results.items():
        count = max(1, res['count'])
        summary[name] = {
            'count': res['count'],
            'exact_acc': res['exact'] / count,
            'first_char_acc': res['first'] / count,
            'top_predictions': sorted(res['pred_counter'].items(), key=lambda kv: (-kv[1], kv[0]))[:10],
            'details': res['details'],
        }
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in {'meta'} and not isinstance(v, dict) or k in {'model', 'contact_sheet'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
