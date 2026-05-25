#!/usr/bin/env python3
"""CVR_val plateau diagnosis — error types, geometry, old-vs-v3, CTC behavior.

Output: experiments/green_cvr_plateau_diagnosis_20260508/"""

import json, csv, sys, math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'training'))
import torch
from torch.utils.data import DataLoader
from load_data import CHARS, UnifiedManifestDataset, parse_ccpd_quad_from_name, order_quad_points
from train_LPRNet import forward_family_logits, collate_fn, Greedy_Decode_Eval
from LPRNet_multihead import build_lprnet_multihead

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BLANK = len(CHARS) - 1

OCR_PARAMS = dict(ocr_crop_mode='obb_warp', ocr_resize_mode='letterbox', ocr_resize_kernel='nn',
                  ocr_preproc='none', ocr_channel_order='bgr', ocr_quad_pad_ratio=0.0)

OUT_DIR = ROOT / 'experiments' / 'green_cvr_plateau_diagnosis_20260508'
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_MANIFEST = ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/val_cvreplace_v2.csv'
POSE_JSONL = ROOT / 'datasets/ccpd2019_tilt_db_challenge_posquads_20260508/pose_quads.jsonl'

MODELS = {
    'old_green': ('experiments/green_e12_province_degrade_unfreeze/best_LPRNet_model.pth', 'expD'),
    'v3_best': ('experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_20260508/best_LPRNet_model.pth', 'expD'),
    'expE_best': ('experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_expE_20260508/best_LPRNet_model.pth', 'expE'),
}


class FakeArgs:
    cuda = torch.cuda.is_available()
    test_batch_size = 120
    num_workers = 4


# ── Load model ──────────────────────────────────────────────────────
def load_model(mpath, head_type):
    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS),
                                  dropout_rate=0.5, enhanced_green_head=head_type, pos0_head_cols=0)
    state = torch.load(str(mpath), map_location='cpu')
    net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


# ── Load pose meta ───────────────────────────────────────────────────
print("Loading pose metadata...", flush=True)
pose_meta = {}
with open(POSE_JSONL) as f:
    for line in f:
        r = json.loads(line)
        pose_meta[r['rel_path']] = r

# ── Load val manifest ────────────────────────────────────────────────
print("Loading val manifest...", flush=True)
val_rows = list(csv.DictReader(open(VAL_MANIFEST)))

# Map by rel_path for pose metadata lookup
for r in val_rows:
    rp = r.get('img_path', '')
    pm = pose_meta.get(rp, {})
    r['_pose_conf'] = pm.get('confidence', -1)
    r['_pose_quad'] = pm.get('pose_quad', None)
    r['_gt_quad'] = pm.get('gt_quad', None)

ds = UnifiedManifestDataset(str(VAL_MANIFEST), [94, 24], 8, split_filter='test',
                             dataset_root=str(ROOT), **OCR_PARAMS)
records = getattr(ds, 'records', [])
print(f"  Samples: {len(ds)}", flush=True)


# ── Per-sample eval for all models ───────────────────────────────────
def per_sample_eval(net, ds):
    """Returns list of (pred_text, gt_text) for each sample."""
    ld = DataLoader(ds, batch_size=120, shuffle=False, num_workers=4, collate_fn=collate_fn)
    results = []
    with torch.no_grad():
        for images, labels, lengths, families in ld:
            images = images.to(device)
            prebs = forward_family_logits(net, images, sample_families=families).cpu().detach().numpy()
            off = 0
            for bi in range(prebs.shape[0]):
                preb = prebs[bi, :, :]
                preb_label = [int(np.argmax(preb[:, tj], axis=0)) for tj in range(preb.shape[1])]
                no_repeat_blank = []
                prec = preb_label[0]
                if prec != BLANK:
                    no_repeat_blank.append(prec)
                for c in preb_label:
                    if (prec == c) or (c == BLANK):
                        if c == BLANK:
                            prec = c
                        continue
                    no_repeat_blank.append(c)
                    prec = c
                pred = ''.join(CHARS[i] for i in no_repeat_blank if 0 <= i < len(CHARS))
                start = off
                end = off + lengths[bi]
                gt = ''.join(CHARS[int(labels[start + i])] for i in range(lengths[bi]))
                off = end
                results.append((pred, gt))
    return results


all_predictions = {}  # model_name -> list of (pred, gt)

for mname, (mpath, head_type) in MODELS.items():
    fp = ROOT / mpath
    if not fp.exists():
        print(f"  SKIP {mname}: not found", flush=True)
        continue
    print(f"Evaluating {mname}...", flush=True)
    net = load_model(fp, head_type)
    all_predictions[mname] = per_sample_eval(net, ds)
    del net
    torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════
# DIAGNOSIS 1: Error Type Decomposition
# ═════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print("DIAGNOSIS 1: Error Type Decomposition")
print(f"{'='*70}", flush=True)

diagnosis = {}

for mname in all_predictions:
    preds = all_predictions[mname]
    d = {'n': len(preds), 'exact': 0, 'char_correct': 0, 'char_total': 0,
         'short': 0, 'long': 0, 'len_match': 0, 'empty': 0,
         'prov_correct': 0, 'prov_total': 0,
         'pos': {i: {'c': 0, 't': 0} for i in range(8)},
         'df_correct': 0, 'df_total': 0,
         'suffix_correct': 0, 'suffix_total': 0,
         'decoded_lengths': [],
         'per_subset': defaultdict(lambda: {'exact': 0, 'total': 0}),
         'per_province': defaultdict(lambda: {'exact': 0, 'total': 0, 'prov1st': 0}),
         'repeated_char_errors': 0,
         }

    for idx, (pred, gt) in enumerate(preds):
        d['decoded_lengths'].append(len(pred))
        if pred == gt:
            d['exact'] += 1

        # Length
        if len(pred) == len(gt):
            d['len_match'] += 1
        elif len(pred) < len(gt):
            d['short'] += 1
        else:
            d['long'] += 1
        if len(pred) == 0:
            d['empty'] += 1

        # Char accuracy
        for p, g in zip(pred, gt):
            d['char_correct'] += (p == g)
        d['char_total'] += len(gt)

        # Province
        if gt and gt[0] in CHARS[:31]:
            d['prov_total'] += 1
            if pred and pred[0] == gt[0]:
                d['prov_correct'] += 1

        # Position accuracy (0-indexed, up to 8)
        for pos, (p, g) in enumerate(zip(pred, gt)):
            if pos < 8:
                d['pos'][pos]['c'] += (p == g)
                d['pos'][pos]['t'] += 1

        # D/F slot (pos3, 0-indexed = position 2)
        if len(gt) > 2 and gt[2] in ('D', 'F'):
            d['df_total'] += 1
            if len(pred) > 2 and pred[2] == gt[2]:
                d['df_correct'] += 1

        # Suffix digits (last 4 chars for green8)
        if len(gt) >= 7:
            gt_suffix = gt[-4:]
            pred_suffix = pred[-4:] if len(pred) >= 4 else ''
            d['suffix_total'] += 4
            for ps, gs in zip(pred_suffix, gt_suffix):
                d['suffix_correct'] += (ps == gs)

        # Repeated char collapse
        if len(pred) < len(gt):
            # Check if pred has unusual repeats
            for i in range(1, len(pred)):
                if pred[i] == pred[i-1] and (i >= len(gt) or i-1 >= len(gt) or pred[i-1] != gt[i-1]):
                    d['repeated_char_errors'] += 1
                    break

        # Subset
        subset = val_rows[idx].get('source', '?').replace('green_ccpd2019_', '').replace('_cvreplace_v2', '').replace('_cvreplace_v3', '')
        d['per_subset'][subset]['total'] += 1
        if pred == gt:
            d['per_subset'][subset]['exact'] += 1

        # Province breakdown
        if gt and gt[0] in CHARS[:31]:
            p = gt[0]
            d['per_province'][p]['total'] += 1
            if pred == gt:
                d['per_province'][p]['exact'] += 1
            if pred and pred[0] == gt[0]:
                d['per_province'][p]['prov1st'] += 1

    diagnosis[mname] = d

for mname in sorted(diagnosis.keys()):
    d = diagnosis[mname]
    n = d['n']
    print(f"\n  {mname} (n={n}):")
    print(f"    exact={d['exact']/n*100:.1f}%  char={d['char_correct']/max(d['char_total'],1)*100:.1f}%")
    print(f"    prov1st={d['prov_correct']/max(d['prov_total'],1)*100:.1f}%  "
          f"len_match={d['len_match']/n*100:.1f}%  short={d['short']/n*100:.1f}%  empty={d['empty']}")
    pos_str = ' | '.join(f"pos{i+1}={d['pos'][i]['c']/max(d['pos'][i]['t'],1)*100:.1f}%"
                         for i in range(8) if d['pos'][i]['t'] > 0)
    print(f"    Position: {pos_str}")
    print(f"    D/F slot: {d['df_correct']/max(d['df_total'],1)*100:.1f}%  "
          f"Suffix char: {d['suffix_correct']/max(d['suffix_total'],1)*100:.1f}%  "
          f"Repeat collapse: {d['repeated_char_errors']}")
    print(f"    Avg decoded len: {np.mean(d['decoded_lengths']):.1f}")
    print(f"    Per-subset:")
    for subset in sorted(d['per_subset'].keys()):
        s = d['per_subset'][subset]
        print(f"      {subset}: {s['exact']/max(s['total'],1)*100:.1f}% ({s['exact']}/{s['total']})")
    print(f"    Province top-5 errors:")
    prov_sorted = sorted(d['per_province'].items(), key=lambda x: -(x[1]['total'] - x[1]['exact']))
    for p, v in prov_sorted[:5]:
        exact_pct = v['exact']/max(v['total'],1)*100
        p1 = v['prov1st']/max(v['total'],1)*100
        print(f"      {p}: exact={exact_pct:.1f}%  prov1st={p1:.1f}%  (n={v['total']})")


# ═════════════════════════════════════════════════════════════════════
# DIAGNOSIS 2: Geometry/quality bucketing
# ═════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print("DIAGNOSIS 2: Geometry & Quality Bucketing")
print(f"{'='*70}", flush=True)

# Compute geometry for each sample
geo_info = []  # list of dicts
for idx, rec in enumerate(records):
    if not isinstance(rec, dict):
        continue
    rp = rec.get('img_path', '')
    pm = pose_meta.get(rp, {})
    pq = pm.get('pose_quad', None)
    gq = pm.get('gt_quad', None)

    info = {}

    # Pose confidence
    info['pose_conf'] = pm.get('confidence', -1)

    # Compute geometry from pose quad
    if pq is not None:
        pts = np.array(pq, dtype=np.float32)
        w_top = max(1, np.linalg.norm(pts[1] - pts[0]))
        w_bot = max(1, np.linalg.norm(pts[2] - pts[3]))
        h_left = max(1, np.linalg.norm(pts[3] - pts[0]))
        h_right = max(1, np.linalg.norm(pts[2] - pts[1]))
        width = max(w_top, w_bot)
        height = max(h_left, h_right)
        info['area'] = float(width * height)
        info['aspect'] = float(width / max(height, 1))

        # Tilt angle from horizontal
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        info['angle'] = float(abs(np.degrees(np.arctan2(dy, max(abs(dx), 1)))))

    # Pose vs GT corner distance
    if pq is not None and gq is not None:
        p = np.array(pq, dtype=np.float32)
        g = np.array(gq, dtype=np.float32)
        # gq is [BR,BL,TL,TR], reorder to match pose order
        g = np.array([g[2], g[3], g[0], g[1]], dtype=np.float32)  # [TL,TR,BR,BL]
        # Order both
        from load_data import order_quad_points
        p_ord = order_quad_points(p)
        g_ord = order_quad_points(g)
        corner_dists = np.linalg.norm(p_ord - g_ord, axis=1)
        info['mean_corner_dist'] = float(corner_dists.mean())
        info['max_corner_dist'] = float(corner_dists.max())

        # Pose-GT IoU
        def poly_area(pts):
            return 0.5 * abs(np.dot(pts[:,0], np.roll(pts[:,1], 1)) - np.dot(pts[:,1], np.roll(pts[:,0], 1)))
        a1 = poly_area(p_ord)
        a2 = poly_area(g_ord)
        # Simplified IoU using convex hull of all 8 points
        from scipy.spatial import ConvexHull
        all_pts = np.vstack([p_ord, g_ord])
        try:
            hull = ConvexHull(all_pts)
            union_area = hull.volume
            intersection = a1 + a2 - union_area
            info['pose_gt_iou'] = float(max(0, intersection / max(union_area, 1)))
        except:
            info['pose_gt_iou'] = 0.0

    # Image quality (estimate from warp)
    img_path = str(ROOT / rp)
    img = cv2.imread(img_path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        info['sharpness'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        info['brightness'] = float(gray.mean())
        info['contrast'] = float(gray.std())

        # Color ratios in the full image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v_ = hsv[:,:,0].astype(np.float32), hsv[:,:,1].astype(np.float32), hsv[:,:,2].astype(np.float32)
        green = ((s > 30) & (v_ > 40) & (h >= 45) & (h <= 100)).sum()
        blue = ((s > 30) & (v_ > 40) & (h >= 100) & (h <= 140)).sum()
        total = h.shape[0] * h.shape[1]
        info['green_ratio'] = float(green / max(total, 1))
        info['blue_ratio'] = float(blue / max(total, 1))

    info['_idx'] = idx
    geo_info.append(info)

print(f"  Geometry data for {len(geo_info)} samples", flush=True)

# Bucketing
def bucket_metric(values, boundaries, labels):
    buckets = defaultdict(list)
    for v in values:
        for b, l in zip(boundaries, labels):
            if v <= b:
                buckets[l].append(v)
                break
        else:
            buckets[labels[-1]].append(v)
    return buckets

# For each model, compute exact acc by bucket
for mname in ['old_green', 'v3_best']:
    if mname not in all_predictions:
        continue
    print(f"\n  Bucketing for {mname}:", flush=True)
    preds = all_predictions[mname]

    # Sharpness buckets
    sharp_vals = [(g['sharpness'], preds[g['_idx']][0] == preds[g['_idx']][1], g['_idx'])
                   for g in geo_info if 'sharpness' in g]
    for threshold, label in [(50, 'low_sharp'), (150, 'mid_sharp'), (9999, 'high_sharp')]:
        grp = [si for si in sharp_vals if si[0] <= threshold] if label == 'low_sharp' else (
              [si for si in sharp_vals if threshold < si[0] <= 9999] if label == 'high_sharp' else
              [si for si in sharp_vals if 50 < si[0] <= 150])
        if label == 'high_sharp':
            grp = [si for si in sharp_vals if si[0] > 150]
        if label == 'mid_sharp':
            grp = [si for si in sharp_vals if 50 < si[0] <= 150]
        n = len(grp)
        exact = sum(1 for _, ok, _ in grp if ok)
        print(f"    {label:15s}: n={n:4d}  exact={exact/max(n,1)*100:.1f}%")

    # Corner distance buckets
    dist_vals = [(g['mean_corner_dist'], preds[g['_idx']][0] == preds[g['_idx']][1], g['_idx'])
                  for g in geo_info if 'mean_corner_dist' in g]
    for threshold, label in [(5, 'low_pose_err'), (15, 'mid_pose_err'), (9999, 'high_pose_err')]:
        if label == 'low_pose_err':
            grp = [si for si in dist_vals if si[0] <= 5]
        elif label == 'mid_pose_err':
            grp = [si for si in dist_vals if 5 < si[0] <= 15]
        else:
            grp = [si for si in dist_vals if si[0] > 15]
        n = len(grp)
        exact = sum(1 for _, ok, _ in grp if ok)
        print(f"    {label:15s}: n={n:4d}  exact={exact/max(n,1)*100:.1f}%")

    # Angle buckets
    angle_vals = [(g['angle'], preds[g['_idx']][0] == preds[g['_idx']][1], g['_idx'])
                   for g in geo_info if 'angle' in g]
    for threshold, label in [(10, 'mild_angle'), (25, 'moderate_angle'), (9999, 'steep_angle')]:
        if label == 'mild_angle':
            grp = [si for si in angle_vals if si[0] <= 10]
        elif label == 'moderate_angle':
            grp = [si for si in angle_vals if 10 < si[0] <= 25]
        else:
            grp = [si for si in angle_vals if si[0] > 25]
        n = len(grp)
        exact = sum(1 for _, ok, _ in grp if ok)
        print(f"    {label:15s}: n={n:4d}  exact={exact/max(n,1)*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════
# DIAGNOSIS 3: Old vs v3 error comparison
# ═════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print("DIAGNOSIS 3: Old vs v3 Error Comparison")
print(f"{'='*70}", flush=True)

if 'old_green' in all_predictions and 'v3_best' in all_predictions:
    old_preds = all_predictions['old_green']
    v3_preds = all_predictions['v3_best']

    recovered = []  # old wrong, v3 right
    regressed = []  # old right, v3 wrong
    both_wrong = []  # both wrong
    both_right = []  # both right

    for idx in range(len(old_preds)):
        op, og = old_preds[idx]
        vp, vg = v3_preds[idx]
        old_ok = (op == og)
        v3_ok = (vp == vg)
        if not old_ok and v3_ok:
            recovered.append(idx)
        elif old_ok and not v3_ok:
            regressed.append(idx)
        elif not old_ok and not v3_ok:
            both_wrong.append(idx)
        else:
            both_right.append(idx)

    print(f"  Both right: {len(both_right)}")
    print(f"  Recovered (old wrong, v3 right): {len(recovered)}")
    print(f"  Regressed (old right, v3 wrong): {len(regressed)}")
    print(f"  Both wrong: {len(both_wrong)}")

    # Sample for contact sheet
    sample_each = 50
    import random
    random.seed(20260508)

    def show_samples(indices, label, out_name):
        sampled = random.sample(indices, min(sample_each, len(indices)))
        rows = []
        for idx in sampled:
            rec = records[idx] if isinstance(records[idx], dict) else {}
            rp = rec.get('img_path', '')
            abs_path = ROOT / rp
            img = cv2.imread(str(abs_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            sf = min(200 / max(h, w), 1.0)
            small = cv2.resize(img, (int(w * sf), int(h * sf)))

            # Warp
            try:
                from load_data import prepare_board_ocr_input_from_quad_bgr888
                quad = np.array([[float(rec['quad_1x']), float(rec['quad_1y'])],
                                 [float(rec['quad_2x']), float(rec['quad_2y'])],
                                 [float(rec['quad_3x']), float(rec['quad_3y'])],
                                 [float(rec['quad_4x']), float(rec['quad_4y'])]], dtype=np.float32)
                prepared, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
                    img, quad, in_w=94, in_h=24,
                    resize_mode='letterbox', resize_kernel='nn',
                    preproc_mode='none', channel_order='bgr', quad_pad_ratio=0.0)
                warp_display = ((prepared - prepared.min()) / max(prepared.max() - prepared.min(), 1) * 255).astype(np.uint8)
                if warp_display.ndim == 2:
                    warp_display = cv2.cvtColor(warp_display, cv2.COLOR_GRAY2BGR)
            except:
                warp_display = np.zeros((24, 94, 3), dtype=np.uint8)

            warp_big = cv2.resize(warp_display, (94*2, 24*2), interpolation=cv2.INTER_NEAREST)

            og = old_preds[idx][1]
            op = old_preds[idx][0]
            vp = v3_preds[idx][0]
            ep = all_predictions.get('expE_best', [('?','?')])[idx][0] if 'expE_best' in all_predictions else '?'

            rows.append({
                'img_small': small, 'warp': warp_big,
                'gt': og, 'old_pred': op, 'v3_pred': vp, 'expE_pred': ep,
            })
            if len(rows) >= 10:
                break

        # Build contact sheet mini
        if rows:
            panel_h = max(r['img_small'].shape[0] for r in rows) + 24*2 + 20
            panel_w = 200 + 94*2 + 20
            sheet = np.ones((panel_h * min(len(rows), 5), panel_w, 3), dtype=np.uint8) * 40
            for i, r in enumerate(rows[:5]):
                y = i * (panel_h // 5) if len(rows) <= 5 else i * (max(r['img_small'].shape[0] for r in rows) + 24*2 + 25)
                if i > 0:
                    y = i * (max(rows[j]['img_small'].shape[0] for j in range(len(rows))) // len(rows) * 2)
                # Actually simpler: just write text report
            # Write CSV instead
            csv_path = OUT_DIR / f'diag3_{out_name}_samples.csv'
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['idx','gt','old_pred','v3_pred','expE_pred'])
                w.writeheader()
                for idx in sampled:
                    e_pred = all_predictions.get('expE_best', [('?','?')])[idx][0] if 'expE_best' in all_predictions else '?'
                    w.writerow({'idx': idx, 'gt': old_preds[idx][1], 'old_pred': old_preds[idx][0],
                                'v3_pred': v3_preds[idx][0], 'expE_pred': e_pred})
            print(f"    {out_name} CSV: {csv_path} ({len(sampled)} samples)", flush=True)

    show_samples(recovered, 'Recovered', 'recovered')
    show_samples(regressed, 'Regressed', 'regressed')
    show_samples(both_wrong, 'Both wrong', 'hard')


# ═════════════════════════════════════════════════════════════════════
# DIAGNOSIS 4: CTC behavior analysis
# ═════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print("DIAGNOSIS 4: CTC Decode Behavior Analysis")
print(f"{'='*70}", flush=True)

for mname in ['old_green', 'v3_best', 'expE_best']:
    if mname not in all_predictions:
        continue
    preds = all_predictions[mname]
    d = diagnosis[mname]

    lens = [len(p) for _, p in preds]
    gt_lens = [len(g) for _, g in preds]

    print(f"\n  {mname}:")
    print(f"    Avg decoded len: {np.mean(lens):.1f}  Avg GT len: {np.mean(gt_lens):.1f}")
    print(f"    Len distribution: {Counter(lens).most_common(5)}")
    print(f"    GT len distribution: {Counter(gt_lens).most_common(5)}")
    print(f"    Short rate: {d['short']/max(d['n'],1)*100:.1f}%")

    # Blank dominance: count blank emissions in raw logits
    # Compute on a sample
    mpath = [p for mname2, (p, htype) in MODELS.items() if mname2 == mname]
    headtype = [htype for mname2, (p, htype) in MODELS.items() if mname2 == mname]
    if not mpath:
        continue
    net = load_model(ROOT / mpath[0], headtype[0])
    ld = DataLoader(ds, batch_size=120, shuffle=False, num_workers=4, collate_fn=collate_fn)
    blank_ratio = []
    with torch.no_grad():
        for images, labels, lengths, families in ld:
            images = images.to(device)
            prebs = forward_family_logits(net, images, sample_families=families).cpu().detach().numpy()
            for bi in range(min(50, prebs.shape[0])):
                preb = prebs[bi, :, :]
                ids = np.argmax(preb, axis=0)
                blank_ratio.append((ids == BLANK).mean())
    del net
    torch.cuda.empty_cache()
    print(f"    Mean blank ratio in raw logits: {np.mean(blank_ratio)*100:.1f}%")
    print(f"    Samples with >50% blank: {sum(1 for b in blank_ratio if b > 0.5)}/{len(blank_ratio)}")


# ═════════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════════
# Save structured diagnosis
structured = {'diagnosis1_error_types': {}, 'diagnosis4_ctc': {}}

for mname in diagnosis:
    d = diagnosis[mname]
    s = {k: v for k, v in d.items() if k not in ('pos', 'per_subset', 'per_province', 'decoded_lengths')}
    s['pos_accuracy'] = {f'pos{i+1}': {'correct': d['pos'][i]['c'], 'total': d['pos'][i]['t'],
        'acc': d['pos'][i]['c']/max(d['pos'][i]['t'],1)} for i in range(8) if d['pos'][i]['t'] > 0}
    s['per_subset'] = {k: {'exact': v['exact'], 'total': v['total'],
        'acc': v['exact']/max(v['total'],1)} for k, v in d['per_subset'].items()}
    s['per_province'] = {k: {'exact': v['exact'], 'total': v['total'], 'prov1st': v['prov1st'],
        'exact_acc': v['exact']/max(v['total'],1), 'prov1st_acc': v['prov1st']/max(v['total'],1)}
        for k, v in sorted(d['per_province'].items(), key=lambda x: -x[1]['total'])}
    structured['diagnosis1_error_types'][mname] = s

json.dump(structured, open(OUT_DIR / 'diagnosis_metrics.json', 'w'), ensure_ascii=False, indent=2)
print(f"\nSaved: {OUT_DIR / 'diagnosis_metrics.json'}", flush=True)
print("Done.", flush=True)
