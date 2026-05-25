#!/usr/bin/env python3
"""Replay current Gray3 paradigm checkpoints on a board OCR dump directory.

Input is board-final ocrin_*.ppm.  Each image is explicitly converted to gray3
before LPRNet normalization, matching the curriculum_gray3 training/eval口径.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src/evaluation', ROOT / 'src/training', ROOT / 'src/utils']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import CHARS, read_ppm_p6_payload, ocr_preprocess_bgr888  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa: E402
from train_LPRNet import _select_family_logits_from_dict  # noqa: E402
from eval_lpr_detailed import decode_logits  # noqa: E402

PROVINCES = CHARS[:31]
BLANK = len(CHARS) - 1
MODELS = {
    'A1D_iter2000': ROOT / 'experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth',
    'B1A_iter2000': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservativeLPRNet__iteration_2000.pth',
    'E1_iter2000': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_originalLPRNet__iteration_2000.pth',
    'E2_iter2000': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E2_provanchor_lmh_ccpdboard_eval_originalLPRNet__iteration_2000.pth',
    'E3_iter2000': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E3_structural_anchor_slot_lmh_ccpdboard_eval_originalLPRNet__iteration_2000.pth',
    'B1A_extreme_v4e3_iter2000': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_extreme_ccpdboard_v4e3LPRNet__iteration_2000.pth',
    'B1A_A_v4e3_iter2000': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_A_train_v4e3_ccpdboard_eval_originalLPRNet__iteration_2000.pth',
    'B1A_C_v4e3_iter2000': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_C_train_v4e3_ccpdboard_eval_originalLPRNet__iteration_2000.pth',
    'B1A_D_extreme900_iter2000': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_originalLPRNet__iteration_2000.pth',
}


def read_index(dump_dir: Path):
    idx = dump_dir / 'index.csv'
    with idx.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        sid = int(r['sample_id'])
        frame = int(r['frame_id'])
        local_ocrin = dump_dir / f'ocrin_{sid:04d}_f{frame:06d}.ppm'
        local_crop = dump_dir / f'crop_{sid:04d}_f{frame:06d}.ppm'
        rec = dict(r)
        rec['sample_id'] = sid
        rec['frame_id'] = frame
        rec['local_ocrin_path'] = str(local_ocrin)
        rec['local_crop_path'] = str(local_crop)
        rec['ocrin_exists'] = int(local_ocrin.exists())
        rec['crop_exists'] = int(local_crop.exists())
        out.append(rec)
    return out


def load_model(path: Path, device):
    state = torch.load(str(path), map_location=device)
    net, _ = build_lprnet_multihead_from_state_dict(
        state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    return net.to(device).eval()


def preprocess_ocrin_gray3(path: str, device):
    img = read_ppm_p6_payload(path)
    if img.shape[:2] != (24, 94):
        img = cv2.resize(img, (94, 24), interpolation=cv2.INTER_NEAREST)
    gray3 = ocr_preprocess_bgr888(img, 'gray3')
    x = (gray3.astype('float32') - 127.5) * 0.0078125
    x = torch.from_numpy(np.transpose(x, (2, 0, 1))[None, ...]).to(device)
    return x, gray3


def greedy_decode(logits_np):
    labels = []
    prev = None
    for t in range(logits_np.shape[1]):
        c = int(np.argmax(logits_np[:, t]))
        if c != BLANK and c != prev:
            labels.append(c)
        prev = c
    return ''.join(CHARS[c] for c in labels)


def province_topk(logits_np):
    vec = torch.tensor(logits_np[:31, :4], dtype=torch.float32).mean(dim=1)
    prob = F.softmax(vec, dim=0).numpy()
    order = np.argsort(prob)[::-1]
    return ';'.join(f'{PROVINCES[int(i)]}:{prob[int(i)]:.3f}' for i in order[:5]), str(PROVINCES[int(order[0])]), float(prob[int(order[0])])


def write_csv(path: Path, rows):
    fields = sorted({k for r in rows for k in r.keys()}) if rows else ['empty']
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    out = {}
    for model in sorted(set(r['model'] for r in rows)):
        arr = [r for r in rows if r['model'] == model]
        n = len(arr)
        out[model] = {
            'n': n,
            'beam_top_preds': dict(Counter(r['pred_beam'] for r in arr).most_common(12)),
            'greedy_top_preds': dict(Counter(r['pred_greedy'] for r in arr).most_common(12)),
            'province_top1_counts': dict(Counter(r['province_top1'] for r in arr).most_common(12)),
            'same_as_app_text_beam': sum(int(r['pred_beam'] == r['app_text']) for r in arr),
            'same_as_app_text_greedy': sum(int(r['pred_greedy'] == r['app_text']) for r in arr),
            'empty_beam': sum(int(r['pred_beam'] == '') for r in arr),
            'empty_greedy': sum(int(r['pred_greedy'] == '') for r in arr),
            'avg_app_occ_ratio': sum(float(r['app_occ_ratio']) for r in arr if str(r.get('app_occ_ratio','')) != '') / max(1, sum(1 for r in arr if str(r.get('app_occ_ratio','')) != '')),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-dir', default='/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ocr_dump')
    ap.add_argument('--out-dir', default='/home/wzzz/LPRNet/reports/gray3_replay_current_dump_20260427')
    args = ap.parse_args()
    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_index(dump_dir)
    rows = [r for r in rows if r['ocrin_exists']]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_rows = []
    model_paths = {k: v for k, v in MODELS.items() if v.exists()}
    missing = {k: str(v) for k, v in MODELS.items() if not v.exists()}
    for name, path in model_paths.items():
        net = load_model(path, device)
        with torch.no_grad():
            for r in rows:
                x, _gray3 = preprocess_ocrin_gray3(r['local_ocrin_path'], device)
                raw = net(x)
                logits = _select_family_logits_from_dict(raw, sample_families=['green8']).detach().cpu().numpy()[0]
                beam_ids = decode_logits(logits[None, ...], 'family_aware_beam', 20, 12, sample_families=['green8'])[0]
                pred_beam = ''.join(CHARS[int(c)] for c in beam_ids)
                pred_greedy = greedy_decode(logits)
                top5, ptop1, ptop1_prob = province_topk(logits)
                all_rows.append({
                    **r,
                    'model': name,
                    'model_path': str(path),
                    'pred_beam': pred_beam,
                    'pred_greedy': pred_greedy,
                    'province_top5': top5,
                    'province_top1': ptop1,
                    'province_top1_prob': f'{ptop1_prob:.6f}',
                    'beam_len': len(pred_beam),
                    'greedy_len': len(pred_greedy),
                })
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(out_dir / 'all_rows.csv', all_rows)
    summary = {
        'dump_dir': str(dump_dir),
        'out_dir': str(out_dir),
        'device': str(device),
        'n_samples': len(rows),
        'models': list(model_paths.keys()),
        'missing_models': missing,
        'summary': summarize(all_rows),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # compact pivot for terminal review
    pivot = []
    by_sample = {}
    for r in all_rows:
        by_sample.setdefault((r['sample_id'], r['frame_id'], r['app_text'], r['app_occ_ratio']), {})[r['model']] = r
    for (sid, frame, app, occ), mm in sorted(by_sample.items())[:50]:
        rec = {'sample_id': sid, 'frame_id': frame, 'app_text': app, 'app_occ_ratio': occ}
        for m in model_paths:
            if m in mm:
                rec[m + '_beam'] = mm[m]['pred_beam']
                rec[m + '_greedy'] = mm[m]['pred_greedy']
                rec[m + '_prov_top5'] = mm[m]['province_top5']
        pivot.append(rec)
    write_csv(out_dir / 'pivot_by_sample.csv', pivot)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
