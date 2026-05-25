#!/usr/bin/env python3
"""
Front-end pre-processing ablation for LPRNet on CCPD.
Evaluates brightness-preserving BGR preproc methods before LPRNet inference.

Methods (all output BGR 3-channel):
  0. raw_bgr                          - baseline, no preprocessing
  1. ycrcb_y_clahe                    - CLAHE on Y channel of YCrCb
  2. lab_l_clahe                      - CLAHE on L channel of CIELAB
  3. y_adaptive_gamma                 - adaptive gamma correction on Y of YCrCb
  4. y_highlight_compress             - highlight compression on Y of YCrCb
  5. y_clahe_plus_highlight_compress  - combined CLAHE + highlight compress
"""

import os, sys, csv, json, time, argparse, random
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import cv2
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import CHARS, parse_ccpd_quad_from_name, prepare_board_ocr_input_from_quad_bgr888
from LPRNet import build_lprnet
from test_LPRNet import greedy_decode_logits

# ── Brightness-preserving BGR preprocessing methods ──

def raw_bgr(bgr):
    return bgr.copy()

def ycrcb_y_clahe(bgr, clip_limit=2.0, tile_grid=(8,8)):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    y_eq = clahe.apply(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def lab_l_clahe(bgr, clip_limit=2.0, tile_grid=(8,8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def y_adaptive_gamma(bgr, target_mean=128):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_f = y.astype(np.float32)
    current_mean = y_f.mean()
    gamma = np.clip(np.log(current_mean.clip(1e-6) / 255.0) / np.log(target_mean / 255.0 + 1e-8), 0.3, 3.0)
    y_gamma = ((y_f / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
    merged = cv2.merge([y_gamma, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def y_highlight_compress(bgr, threshold=200, slope=0.3):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_f = y.astype(np.float32)
    mask = y_f > threshold
    y_f[mask] = threshold + (y_f[mask] - threshold) * slope
    y_out = np.clip(y_f, 0, 255).astype(np.uint8)
    merged = cv2.merge([y_out, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def y_clahe_plus_highlight_compress(bgr, clip_limit=1.5, tile_grid=(8,8), threshold=200, slope=0.3):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    y_eq = clahe.apply(y)
    y_f = y_eq.astype(np.float32)
    mask = y_f > threshold
    y_f[mask] = threshold + (y_f[mask] - threshold) * slope
    y_out = np.clip(y_f, 0, 255).astype(np.uint8)
    merged = cv2.merge([y_out, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

METHODS = OrderedDict([
    ('raw_bgr', raw_bgr),
    ('ycrcb_y_clahe', ycrcb_y_clahe),
    ('lab_l_clahe', lab_l_clahe),
    ('y_adaptive_gamma', y_adaptive_gamma),
    ('y_highlight_compress', y_highlight_compress),
    ('y_clahe_plus_highlight_compress', y_clahe_plus_highlight_compress),
])

def compute_cer(pred, gt):
    if not pred and not gt: return 0.0
    if not pred or not gt: return 1.0
    if len(pred) == len(gt):
        return sum(1 for a,b in zip(pred,gt) if a!=b) / max(len(gt),1)
    dp = np.zeros((len(pred)+1,len(gt)+1))
    for i in range(len(pred)+1): dp[i][0]=i
    for j in range(len(gt)+1): dp[0][j]=j
    for i in range(1,len(pred)+1):
        for j in range(1,len(gt)+1):
            cost=0 if pred[i-1]==gt[j-1] else 1
            dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+cost)
    return dp[len(pred)][len(gt)]/max(len(gt),1)

def load_model(model_path, device, class_num=68, dropout_rate=0.0):
    net = build_lprnet(lpr_max_len=8, phase=False, class_num=class_num, dropout_rate=dropout_rate)
    state = torch.load(model_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_csv', default='eval_reports/front_preprocess_ablation/eval_subset_all.csv')
    parser.add_argument('--model',
        default='experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth')
    parser.add_argument('--output_dir', default='eval_reports/front_preprocess_ablation')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--class_num', type=int, default=68)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--methods', nargs='+', default=list(METHODS.keys()))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    print(f'Model: {args.model}', flush=True)
    print(f'Output: {out_dir.resolve()}', flush=True)

    # ── Load records ──
    records = []
    with open(args.subset_csv) as f:
        for row in csv.DictReader(f):
            records.append(row)
    print(f'Loaded {len(records)} records', flush=True)

    if args.max_samples > 0:
        random.seed(20260507)
        records = random.sample(records, min(args.max_samples, len(records)))
        print(f'Limited to {len(records)} samples', flush=True)

    # ── Parse quads ──
    print('Parsing quads...', flush=True)
    valid = []
    for r in records:
        quad = parse_ccpd_quad_from_name(r['filename'])
        if quad is None:
            continue
        r['quad'] = quad
        valid.append(r)
    records = valid
    print(f'Valid records: {len(records)}', flush=True)

    # ── Pre-warp all images once ──
    # This is the bottleneck, do it once
    print('Pre-warping all images (this may take a while)...', flush=True)
    warped_list = []
    t0 = time.time()
    for idx, r in enumerate(records):
        try:
            img_bgr = cv2.imread(r['image_path'])
            if img_bgr is None:
                warped_list.append(None)
                continue
            # Use prepare_board_ocr_input_from_quad_bgr888 with preproc_mode='none'
            prepared, occ, warped, ordered_quad, _ = \
                prepare_board_ocr_input_from_quad_bgr888(
                    img_bgr, r['quad'], 94, 24,
                    'letterbox', 'nn', 'none', 'bgr',
                    quad_pad_ratio=0.0)
            warped_list.append(prepared)  # [24,94,3] BGR uint8
        except Exception as e:
            print(f'  Error warping {r["filename"]}: {e}', flush=True)
            warped_list.append(None)
        if (idx+1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f'  Warped [{idx+1}/{len(records)}] {elapsed:.1f}s', flush=True)
    elapsed = time.time() - t0
    print(f'Warp done: {elapsed:.1f}s for {len(records)} images', flush=True)

    # Filter out failed warps
    valid_indices = [i for i, w in enumerate(warped_list) if w is not None]
    valid_records = [records[i] for i in valid_indices]
    warped_imgs = [warped_list[i] for i in valid_indices]
    print(f'Successfully warped: {len(valid_records)}/{len(records)}', flush=True)

    # ── Load model ──
    model = load_model(args.model, device, class_num=args.class_num)
    print('Model loaded.', flush=True)

    # Pre-allocate normalization buffer (HWC -> CHW)
    def normalize_batch(imgs_bgr):
        """List of [H,W,3] BGR uint8 -> [B,68,18] logits"""
        batch = np.stack(imgs_bgr, axis=0).astype(np.float32)  # [B,24,94,3]
        batch = np.transpose(batch, (0, 3, 1, 2))              # [B,3,24,94]
        batch = (batch - 127.5) / 128.0
        batch = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            logits = model(batch)
        return logits.detach().cpu().numpy()

    # ── Eval each method ──
    methods_to_run = [m for m in METHODS if m in args.methods]
    results = OrderedDict()

    for method_name in methods_to_run:
        print(f'\n{"="*50}', flush=True)
        print(f'Evaluating: {method_name}', flush=True)
        print(f'{"="*50}', flush=True)

        preproc_fn = METHODS[method_name]
        t0 = time.time()

        all_preds = [None] * len(valid_records)
        all_gts = [r['label'] for r in valid_records]
        all_subsets = [r['subset_alias'] for r in valid_records]
        all_fnames = [r['filename'] for r in valid_records]

        # Process in batches
        for batch_start in range(0, len(valid_records), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(valid_records))
            batch_slice = slice(batch_start, batch_end)
            batch_warps = warped_imgs[batch_slice]
            batch_warps = [preproc_fn(w) for w in batch_warps]
            logits = normalize_batch(batch_warps)
            decoded = greedy_decode_logits(logits)
            for i, seq in enumerate(decoded):
                idx = batch_start + i
                all_preds[idx] = ''.join(CHARS[int(c)] for c in seq)

            if (batch_end) % 1000 == 0 or batch_end == len(valid_records):
                rate = batch_end / (time.time() - t0) if time.time()-t0 > 0 else 0
                print(f'  [{batch_end}/{len(valid_records)}] {rate:.1f} img/s', flush=True)

        elapsed = time.time() - t0
        correct = [p == g for p, g in zip(all_preds, all_gts)]
        em = sum(correct) / len(correct)
        cers = [compute_cer(p, g) for p, g in zip(all_preds, all_gts)]
        mean_cer = np.mean(cers)

        results[method_name] = {
            'total': len(correct),
            'exact_match': float(em),
            'mean_cer': float(mean_cer),
            'correct_count': int(sum(correct)),
            'time_sec': elapsed,
            'time_per_img_ms': (elapsed / len(correct) * 1000),
            'per_sample': list(zip(all_fnames, all_subsets, all_gts, all_preds, correct)),
        }
        print(f'  EM={em:.4f} CER={mean_cer:.4f} corr={sum(correct)}/{len(correct)} ({elapsed:.1f}s)', flush=True)

    # ── Subset breakdown ──
    subset_breakdown = OrderedDict()
    for mn in methods_to_run:
        subset_breakdown[mn] = bd = {}
        by_ss = defaultdict(list)
        for fn, ss, gt, pred, corr in results[mn]['per_sample']:
            by_ss[ss].append((fn, ss, gt, pred, corr))
        for ss in ['Base','Tilt','Weather','DB','Challenge']:
            items = by_ss.get(ss, [])
            if not items:
                continue
            n = len(items)
            em = sum(c for *_, c in items) / n
            cers = [compute_cer(p, g) for *_, g, p, _ in items]
            bd[ss] = {'total': n, 'exact_match': float(em), 'mean_cer': float(np.mean(cers))}

    # ── Save reports ──
    # JSON
    json_out = {}
    for mn in methods_to_run:
        r = results[mn]
        json_out[mn] = {k: r[k] for k in ['total','exact_match','mean_cer','correct_count','time_sec','time_per_img_ms']}
    json_out['subset_breakdown'] = {mn: bd for mn, bd in subset_breakdown.items()}
    with open(out_dir / 'ablation_results.json', 'w') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f'\nJSON saved: {out_dir/"ablation_results.json"}', flush=True)

    # Per-method predictions CSV
    for mn in methods_to_run:
        p = out_dir / f'predictions_{mn}.csv'
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['filename','subset','gt','pred','correct'])
            for fn, ss, gt, pred, corr in results[mn]['per_sample']:
                w.writerow([fn, ss, gt, pred, int(corr)])
        print(f'Predictions: {p}', flush=True)

    # Summary CSV
    p = out_dir / 'ablation_summary.csv'
    baseln = results.get('raw_bgr')
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method','total','exact_match','mean_cer','correct_count','vs_raw_em_diff','vs_raw_cer_diff','time_per_img_ms'])
        for mn in methods_to_run:
            r = results[mn]
            ed = r['exact_match'] - baseln['exact_match'] if baseln else 0
            cd = r['mean_cer'] - baseln['mean_cer'] if baseln else 0
            w.writerow([mn, r['total'], f'{r["exact_match"]:.4f}', f'{r["mean_cer"]:.4f}',
                        r['correct_count'], f'{ed:+.4f}', f'{cd:+.4f}', f'{r["time_per_img_ms"]:.2f}'])
    print(f'Summary: {p}', flush=True)

    # Subset summary CSV
    p = out_dir / 'ablation_subset_summary.csv'
    bsl_subs = subset_breakdown.get('raw_bgr', {})
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method','subset','total','exact_match','mean_cer','vs_raw_em_diff','vs_raw_cer_diff'])
        for mn in methods_to_run:
            bd = subset_breakdown[mn]
            for ss in ['Base','Tilt','Weather','DB','Challenge']:
                if ss not in bd: continue
                r = bd[ss]
                br = bsl_subs.get(ss, {})
                ed = r['exact_match'] - br.get('exact_match',0) if br else 0
                cd = r['mean_cer'] - br.get('mean_cer',0) if br else 0
                w.writerow([mn, ss, r['total'], f'{r["exact_match"]:.4f}', f'{r["mean_cer"]:.4f}',
                            f'{ed:+.4f}', f'{cd:+.4f}'])
    print(f'Subset summary: {p}', flush=True)

    # Print table
    print(f'\n{"="*80}', flush=True)
    print('Method'.ljust(35), end='')
    for ss in ['Base','Tilt','Weather','DB','Challenge']:
        print(f'  {ss:>8s}', end='')
    print(f'  {"Overall":>8s}  {"CER":>8s}')
    print('-'*80, flush=True)
    for mn in methods_to_run:
        print(f'{mn:35s}', end='')
        for ss in ['Base','Tilt','Weather','DB','Challenge']:
            if ss in subset_breakdown[mn]:
                print(f'  {subset_breakdown[mn][ss]["exact_match"]:>8.4f}', end='')
            else:
                print(f'  {"N/A":>8s}', end='')
        print(f'  {results[mn]["exact_match"]:>8.4f}  {results[mn]["mean_cer"]:>8.4f}', flush=True)

    # ── Failure case analysis ──
    print(f'\n{"="*60}', flush=True)
    print('Failure Case Analysis', flush=True)
    print(f'{"="*60}', flush=True)

    sample_map = {}
    for mn in methods_to_run:
        for fn, ss, gt, pred, corr in results[mn]['per_sample']:
            if fn not in sample_map:
                sample_map[fn] = {'subset':ss, 'gt':gt, 'preds':{}}
            sample_map[fn]['preds'][mn] = (pred, corr)

    all_wrong = []
    raw_right_other_wrong = []
    raw_wrong_other_right = []

    for fn, info in sample_map.items():
        raw_c = info['preds'].get('raw_bgr', ('',False))[1]
        others_c = any(info['preds'][m][1] for m in methods_to_run if m!='raw_bgr' and m in info['preds'])
        all_c = all(info['preds'][m][1] for m in methods_to_run if m in info['preds'])
        if not all_c:
            all_wrong.append((fn, info['subset'], info['gt'], info['preds']))
        if raw_c and not others_c:
            raw_right_other_wrong.append((fn, info['subset'], info['gt'], info['preds']))
        if not raw_c and others_c:
            raw_wrong_other_right.append((fn, info['subset'], info['gt'], info['preds']))

    print(f'  All methods wrong: {len(all_wrong)}', flush=True)
    print(f'  Raw correct but others wrong: {len(raw_right_other_wrong)}', flush=True)
    print(f'  Raw wrong but >=1 other correct: {len(raw_wrong_other_right)}', flush=True)

    def write_fail_csv(name, cases):
        p = out_dir / f'fail_{name}.csv'
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            headers = ['filename','subset','gt']
            for m in methods_to_run:
                headers.extend([f'{m}_pred', f'{m}_correct'])
            w.writerow(headers)
            for fn, ss, gt, preds in cases[:200]:
                row = [fn, ss, gt]
                for m in methods_to_run:
                    pred, corr = preds.get(m, ('',False))
                    row.extend([pred, int(corr)])
                w.writerow(row)
        print(f'  Saved: {p}', flush=True)

    for cat, cases in [('all_methods_wrong', all_wrong), ('raw_correct_others_wrong', raw_right_other_wrong), ('raw_wrong_others_correct', raw_wrong_other_right)]:
        if cases:
            write_fail_csv(cat, cases)

    print(f'\nDone. All outputs in {out_dir.resolve()}', flush=True)

if __name__ == '__main__':
    main()
