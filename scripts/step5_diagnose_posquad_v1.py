#!/usr/bin/env python3
"""Step 1: Diagnose bottlenecks of current best posquad model.
Error analysis: per-subset, per-province, per-position, confidence/geometry buckets."""

import csv, json, sys, os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import CHARS, UnifiedManifestDataset
from LPRNet import build_lprnet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BLANK = len(CHARS) - 1

OCR_PARAMS = dict(
    ocr_crop_mode='obb_warp', ocr_resize_mode='letterbox',
    ocr_resize_kernel='nn', ocr_preproc='none',
    ocr_channel_order='bgr', ocr_quad_pad_ratio=0.0,
)

MODEL_PATH = ROOT / 'experiments/blue_ccpd2019_tilt_db_challenge_posquad_20260508/best_LPRNet_model.pth'
TEST_MANIFEST = ROOT / 'manifests_rebased/blue_ccpd2019_tilt_db_challenge_posquad_20260508/test_posquad.csv'
POSE_QUADS_JSONL = ROOT / 'datasets/ccpd2019_tilt_db_challenge_posquads_20260508/pose_quads.jsonl'
OUTPUT_JSON = ROOT / 'experiments/blue_ccpd2019_tilt_db_challenge_posquad_20260508/diagnosis.json'


def greedy_decode(logits):
    preds = logits.permute(2, 0, 1).cpu().numpy()
    ret = []
    for b in range(logits.shape[0]):
        ids = np.argmax(preds[:, b, :], axis=1)
        dec, prev = [], ids[0]
        if prev != BLANK:
            dec.append(prev)
        for c in ids:
            if c == prev or c == BLANK:
                if c == BLANK: prev = c
                continue
            dec.append(c); prev = c
        ret.append(''.join(CHARS[i] for i in dec if 0 <= i < len(CHARS)))
    return ret


# Load model
print("Loading model...", flush=True)
net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS))
state = torch.load(str(MODEL_PATH), map_location='cpu')
net.load_state_dict(state, strict=False)
net.to(device)
net.eval()

# Load pose quad metadata for confidence/geometry
print("Loading pose metadata...", flush=True)
pose_meta = {}
with open(POSE_QUADS_JSONL) as f:
    for line in f:
        r = json.loads(line)
        pose_meta[r['img_path']] = r  # has confidence, num_detections

# Load test manifest
print("Loading test manifest...", flush=True)
all_rows = []
with open(TEST_MANIFEST) as f:
    for row in csv.DictReader(f):
        all_rows.append(row)

# Group by subset
by_subset = defaultdict(list)
for row in all_rows:
    subset = row.get('source', '').replace('ccpd2019_', '')
    by_subset[subset].append(row)

subset_manifests = {}
for subset, rows in by_subset.items():
    path = Path(f'/tmp/diag_{subset}.csv')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        for r in rows:
            r['split'] = 'test'
            w.writerow(r)
    subset_manifests[subset] = (path, len(rows))

# Combined
combined_path = Path('/tmp/diag_all.csv')
with open(combined_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    w.writeheader()
    for r in all_rows:
        r['split'] = 'test'
        w.writerow(r)
subset_manifests['all'] = (combined_path, len(all_rows))


def evaluate_detail(net, manifest_path, subset_name, pose_meta):
    """Evaluate and return detailed per-sample results."""
    dataset = UnifiedManifestDataset(
        str(manifest_path), [94, 24], 8, split_filter='test',
        dataset_root=str(ROOT), **OCR_PARAMS,
    )
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=120, shuffle=False, num_workers=4,
        collate_fn=lambda b: (
            torch.from_numpy(np.stack([x[0] for x in b])),
            torch.from_numpy(np.concatenate([x[1] for x in b])),
            [x[2] for x in b],
            [x[3] if len(x) > 3 else 'normal7' for x in b],
        ))

    # Statistics
    total = 0
    exact_correct = 0
    empty_preds = 0
    short_preds = 0
    long_preds = 0
    len_mismatch = 0
    char_correct = 0
    char_total = 0

    # Per-position accuracy (for 7-char plates)
    pos_correct = Counter()
    pos_total = Counter()

    # Per-province
    prov_correct = defaultdict(int)
    prov_total = defaultdict(int)

    # Confusion: wrong predictions per GT char
    confusion = defaultdict(Counter)

    # Per-sample detail
    samples_detail = []

    # Access dataset records for img_path matching
    records = getattr(dataset, 'records', [])

    with torch.no_grad():
        for batch_idx, (imgs, labels, lengths, fams) in enumerate(loader):
            imgs = imgs.to(device)
            logits = net(imgs)
            preds = greedy_decode(logits)

            off = 0
            for b in range(len(preds)):
                gt_len = int(lengths[b])
                gt_ids = [int(labels[off + i]) for i in range(gt_len)]
                gt_text = ''.join(CHARS[i] for i in gt_ids if 0 <= i < len(CHARS))
                off += gt_len

                pred_text = preds[b]
                total += 1

                # Get record index
                rec_idx = batch_idx * loader.batch_size + b
                img_path = ''
                if rec_idx < len(records):
                    img_path = records[rec_idx].get('img_path', '') if isinstance(records[rec_idx], dict) else ''

                # Pose metadata
                abs_path = str(ROOT / img_path) if img_path else ''
                pm = pose_meta.get(abs_path, {})
                pose_conf = pm.get('confidence', -1)
                num_det = pm.get('num_detections', 0)

                # Exact match
                is_exact = (pred_text == gt_text)
                if is_exact:
                    exact_correct += 1

                # Empty / short / long
                if len(pred_text) == 0:
                    empty_preds += 1
                if len(pred_text) < len(gt_text):
                    short_preds += 1
                if len(pred_text) > len(gt_text):
                    long_preds += 1
                if len(pred_text) != len(gt_text):
                    len_mismatch += 1

                # Per-char accuracy and position accuracy
                for pos, (p, g) in enumerate(zip(pred_text, gt_text)):
                    if pos < 7:
                        pos_total[pos] += 1
                        if p == g:
                            pos_correct[pos] += 1
                    char_total += 1
                    if p == g:
                        char_correct += 1
                    elif g in CHARS:
                        confusion[g][p] = confusion[g].get(p, 0) + 1

                # Province
                if gt_text and gt_text[0] in CHARS and CHARS.index(gt_text[0]) < 31:
                    prov_total[gt_text[0]] += 1
                    if pred_text and pred_text[0] == gt_text[0]:
                        prov_correct[gt_text[0]] += 1

                # Store sample detail
                samples_detail.append({
                    'img_path': img_path,
                    'subset': subset_name,
                    'gt_text': gt_text,
                    'pred_text': pred_text,
                    'is_exact': is_exact,
                    'pred_len': len(pred_text),
                    'gt_len': len(gt_text),
                    'pose_confidence': pose_conf,
                    'num_detections': num_det,
                })

    # Compute stats
    prov_breakdown = {}
    for p in sorted(prov_total.keys()):
        prov_breakdown[p] = {
            'count': prov_total[p],
            'correct': prov_correct[p],
            'acc': prov_correct[p] / max(prov_total[p], 1) * 100,
        }

    pos_acc = {}
    for p in range(7):
        pos_acc[f'pos{p+1}'] = {
            'total': pos_total.get(p, 0),
            'correct': pos_correct.get(p, 0),
            'acc': pos_correct.get(p, 0) / max(pos_total.get(p, 1), 1) * 100,
        }

    # Top confusions: most frequent wrong predictions
    top_confusions = {}
    for gt_char, wrongs in confusion.items():
        top = wrongs.most_common(5)
        top_confusions[gt_char] = [{'predicted': p, 'count': c} for p, c in top]

    return {
        'sample_count': total,
        'exact_plate_acc': exact_correct / max(total, 1),
        'exact_correct': exact_correct,
        'char_acc': char_correct / max(char_total, 1),
        'empty_pred_rate': empty_preds / max(total, 1),
        'empty_pred_count': empty_preds,
        'short_pred_rate': short_preds / max(total, 1),
        'short_pred_count': short_preds,
        'long_pred_rate': long_preds / max(total, 1),
        'len_mismatch_rate': len_mismatch / max(total, 1),
        'province_first_char_acc': (
            sum(prov_correct.values()) / max(sum(prov_total.values()), 1)
        ),
        'province_breakdown': prov_breakdown,
        'position_accuracy': pos_acc,
        'top_confusions': top_confusions,
    }


results = {}
for subset_name, (manifest_path, n_expected) in subset_manifests.items():
    print(f"\nEvaluating {subset_name} ({n_expected} samples)...", flush=True)
    r = evaluate_detail(net, manifest_path, subset_name, pose_meta)
    if r:
        results[subset_name] = r
        print(f"  exact={r['exact_plate_acc']*100:.1f}%  "
              f"char={r['char_acc']*100:.1f}%  "
              f"prov1st={r['province_first_char_acc']*100:.1f}%  "
              f"empty={r['empty_pred_rate']*100:.1f}%  "
              f"short={r['short_pred_rate']*100:.1f}%", flush=True)
        print(f"  Position acc: " + " ".join(
            f"pos{p}={v['acc']:.1f}%" for p, v in sorted(r['position_accuracy'].items())
        ), flush=True)


# ── Per-province summary across all ──────────────────────────────
if 'all' in results:
    r = results['all']
    prov = r['province_breakdown']
    print(f"\n  --- Province Breakdown (All, n={r['sample_count']}) ---", flush=True)
    print(f"  {'Prov':6s} {'Count':>7s} {'%Total':>7s} {'Acc':>7s} {'ErrShare':>9s}", flush=True)
    total = r['sample_count']
    total_errors = total - r['exact_correct']
    for p in sorted(prov.keys(), key=lambda x: -prov[x]['count']):
        err_share = (prov[p]['count'] - prov[p]['correct']) / max(total_errors, 1) * 100
        mark = ' ***' if prov[p]['count'] / max(total, 1) > 0.30 else ''
        print(f"  {p:6s} {prov[p]['count']:7d} {prov[p]['count']/total*100:6.1f}%{mark} "
              f"{prov[p]['acc']:6.1f}% {err_share:7.1f}%", flush=True)

# ── Confidence buckets ──────────────────────────────────────────
# This needs per-sample detail with pose confidence. Let me compute it.
print(f"\n  --- Pose Confidence Buckets ---", flush=True)
if 'all' in results:
    # Need to reload with per-sample detail for confidence binning
    dataset = UnifiedManifestDataset(
        str(combined_path), [94, 24], 8, split_filter='test',
        dataset_root=str(ROOT), **OCR_PARAMS,
    )
    records = getattr(dataset, 'records', [])
    conf_buckets = {'high': {'total': 0, 'ok': 0, 'samples': []},
                    'mid': {'total': 0, 'ok': 0, 'samples': []},
                    'low': {'total': 0, 'ok': 0, 'samples': []}}

    for rec in records:
        if not isinstance(rec, dict):
            continue
        img_path = rec.get('img_path', '')
        abs_path = str(ROOT / img_path)
        pm = pose_meta.get(abs_path, {})
        conf = pm.get('confidence', -1)

    # Re-evaluate with confidence tracking
    loader = DataLoader(dataset, batch_size=120, shuffle=False, num_workers=4,
        collate_fn=lambda b: (
            torch.from_numpy(np.stack([x[0] for x in b])),
            torch.from_numpy(np.concatenate([x[1] for x in b])),
            [x[2] for x in b],
            [x[3] if len(x) > 3 else 'normal7' for x in b],
        ))
    conf_buckets = {'high': {'total': 0, 'ok': 0},
                    'mid': {'total': 0, 'ok': 0},
                    'low': {'total': 0, 'ok': 0}}
    
    with torch.no_grad():
        rec_idx = 0
        for imgs, labels, lengths, fams in loader:
            imgs = imgs.to(device)
            logits = net(imgs)
            preds = greedy_decode(logits)
            off = 0
            for b in range(len(preds)):
                if rec_idx >= len(records):
                    break
                rec = records[rec_idx]
                img_path = rec.get('img_path', '') if isinstance(rec, dict) else ''
                abs_path = str(ROOT / img_path)
                pm = pose_meta.get(abs_path, {})
                conf = pm.get('confidence', -1)
                rec_idx += 1

                gt_len = int(lengths[b])
                gt_ids = [int(labels[off + i]) for i in range(gt_len)]
                gt_text = ''.join(CHARS[i] for i in gt_ids if 0 <= i < len(CHARS))
                off += gt_len

                if conf >= 0.9:
                    bucket = 'high'
                elif conf >= 0.7:
                    bucket = 'mid'
                else:
                    bucket = 'low'

                conf_buckets[bucket]['total'] += 1
                if preds[b] == gt_text:
                    conf_buckets[bucket]['ok'] += 1

    for bucket in ['high', 'mid', 'low']:
        b = conf_buckets[bucket]
        acc = b['ok'] / max(b['total'], 1) * 100
        print(f"  {bucket:6s}: {b['total']:6d} samples  acc={acc:5.1f}%", flush=True)
    results['confidence_buckets'] = {k: {'count': v['total'], 'exact_count': v['ok'],
        'acc': v['ok']/max(v['total'],1)} for k, v in conf_buckets.items()}

# ── Quad geometry buckets ────────────────────────────────────────
# Compute quad area, aspect ratio, tilt angle from pose metadata + GT quad
print(f"\n  --- Quad Geometry Analysis ---", flush=True)
if 'all' in results:
    geo_buckets = defaultdict(lambda: {'total': 0, 'ok': 0})
    for rec in records:
        if not isinstance(rec, dict):
            continue
        img_path = rec.get('img_path', '')
        abs_path = str(ROOT / img_path)
        pm = pose_meta.get(abs_path, {})
        pq = pm.get('pose_quad', None)
        if pq is None:
            continue
        pts = np.array(pq, dtype=np.float32)
        # Compute width and height from quad
        w_top = np.linalg.norm(pts[1] - pts[0])
        w_bot = np.linalg.norm(pts[2] - pts[3])
        h_left = np.linalg.norm(pts[3] - pts[0])
        h_right = np.linalg.norm(pts[2] - pts[1])
        width = max(w_top, w_bot)
        height = max(h_left, h_right)
        aspect = width / max(height, 1)
        area = width * height
        
        # Tilt angle (from horizontal of top edge)
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        angle = abs(np.degrees(np.arctan2(dy, max(dx, 1))))
        
        # Bucket by area
        if area > 20000:
            area_bucket = 'large'
        elif area > 8000:
            area_bucket = 'medium'
        else:
            area_bucket = 'small'
        geo_buckets[f'area_{area_bucket}']['total'] += 1
        
        # Bucket by aspect ratio
        if aspect > 4.0:
            aspect_bucket = 'wide'
        elif aspect > 2.5:
            aspect_bucket = 'normal'
        else:
            aspect_bucket = 'narrow'
        geo_buckets[f'aspect_{aspect_bucket}']['total'] += 1
        
        # Bucket by angle
        if angle > 20:
            angle_bucket = 'steep'
        elif angle > 10:
            angle_bucket = 'moderate'
        else:
            angle_bucket = 'mild'
        geo_buckets[f'angle_{angle_bucket}']['total'] += 1

    for key in sorted(geo_buckets.keys()):
        b = geo_buckets[key]
        print(f"  {key:20s}: {b['total']:6d} samples", flush=True)
    results['geometry_distribution'] = {k: dict(v) for k, v in geo_buckets.items()}

# ── Save results ────────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUTPUT_JSON, 'w'), ensure_ascii=False, indent=2)
print(f"\nDiagnosis saved: {OUTPUT_JSON}", flush=True)
