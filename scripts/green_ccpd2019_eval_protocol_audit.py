#!/usr/bin/env python3
"""Eval protocol audit. Checks all models, manifests, and overlaps."""

import csv, json, torch, sys
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))

OUT_DIR = ROOT / 'experiments/green_ccpd2019_eval_protocol_audit_20260508'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model configs ──────────────────────────────────────────────────
MODEL_CONFIGS = {
    'old_green': {
        'path': ROOT / 'experiments/green_e12_province_degrade_unfreeze/best_LPRNet_model.pth',
        'summary': ROOT / 'experiments/green_e12_province_degrade_unfreeze/train_summary.json',
    },
    'v2_cvreplace': {
        'path': ROOT / 'experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/best_LPRNet_model.pth',
        'summary': ROOT / 'experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/train_summary.json',
    },
    'v3_best': {
        'path': ROOT / 'experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_20260508/best_LPRNet_model.pth',
        'summary': ROOT / 'experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_20260508/train_summary.json',
    },
}

EVAL_MANIFESTS = {
    'cvreplace_val': ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/val_cvreplace_v2.csv',
    'green_simple': ROOT / 'manifests_rebased/curriculum_gray3/test_green_simple.csv',
    'green_val': ROOT / 'manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv',
    'green_hard': ROOT / 'manifests_rebased/curriculum_gray3/test_green_hard.csv',
}

audit = {
    'eval_protocol': {
        'method': 'Greedy_Decode_Eval from train_LPRNet.py',
        'decode_mode': 'greedy CTC (argmax per time-step + collapse repeats+blanks)',
        'collate_fn': 'train_LPRNet.collate_fn (same as training)',
        'family_selection': 'forward_family_logits (selects correct multihead per sample)',
        'no_grad': True,
        'model_eval': True,
        'optimizer_backward': False,
        'batch_size': 120,
        'num_workers': 4,
    },
    'models': {},
    'eval_sets': {},
    'train_eval_overlap': {},
}

# ── Check each model ──────────────────────────────────────────────
for mname, mc in MODEL_CONFIGS.items():
    m = {'checkpoint': str(mc['path']), 'exists': mc['path'].exists()}
    
    if mc['summary'].exists():
        s = json.load(open(mc['summary']))
        a = s.get('args', {})
        m['training_config'] = {
            k: a.get(k, 'N/A') for k in [
                'head_mode', 'enhanced_green_head', 'pos0_head_cols',
                'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
                'ocr_preproc', 'ocr_channel_order', 'ocr_quad_pad_ratio',
                'dataset_root', 'cuda', 'learning_rate', 'max_epoch',
                'province_balance_mode', 'first_char_aux_weight',
                'freeze_backbone', 'trainable_backbone_prefixes',
                'train_brightness_aug_max',
            ]
        }
        m['train_manifest_path'] = a.get('train_manifest', 'N/A')
        m['test_manifest_path'] = a.get('test_manifest', 'N/A')
        m['best_proxy_acc'] = s.get('best_proxy_acc', 'N/A')
        m['best_epoch'] = s.get('best_epoch', 'N/A')
    else:
        m['training_config'] = {'error': 'summary not found'}
    
    # Check checkpoint structure
    if mc['path'].exists():
        state = torch.load(str(mc['path']), map_location='cpu')
        m['checkpoint_num_keys'] = len(state)
        m['has_green8_container'] = any('green8' in k for k in state)
        m['has_normal7'] = any('normal7' in k for k in state)
        # Check if model is multihead
        has_container = any(k.startswith('containers.') for k in state)
        m['is_multihead'] = has_container
    
    audit['models'][mname] = m

# ── Check each eval manifest ──────────────────────────────────────
for ename, epath in EVAL_MANIFESTS.items():
    if not epath.exists():
        audit['eval_sets'][ename] = {'error': 'NOT FOUND', 'path': str(epath)}
        continue
    
    rows = list(csv.DictReader(open(epath)))
    r0 = rows[0] if rows else {}
    
    splits = Counter(r.get('split', '?') for r in rows)
    families = Counter(r.get('family', '?') for r in rows)
    sources = Counter(r.get('source', '?') for r in rows)
    provs = Counter(r.get('text', '?')[:1] for r in rows if r.get('text'))
    has_quad = Counter(r.get('has_quad', '0') for r in rows)
    quad_source = Counter(r.get('quad_source', '?') for r in rows)
    
    audit['eval_sets'][ename] = {
        'path': str(epath),
        'total_rows': len(rows),
        'unique_img_paths': len(set(r.get('img_path', '') for r in rows)),
        'splits': dict(splits),
        'families': dict(families),
        'sources_top10': dict(sources.most_common(10)),
        'provinces_top10': dict(provs.most_common(10)),
        'n_provinces': len(provs),
        'has_quad': dict(has_quad),
        'quad_source': dict(quad_source),
        'manifest_ocr_params': {
            'ocr_crop_mode': r0.get('ocr_crop_mode', ''),
            'ocr_resize_mode': r0.get('ocr_resize_mode', ''),
            'ocr_resize_kernel': r0.get('ocr_resize_kernel', ''),
            'ocr_preproc': r0.get('ocr_preproc', ''),
            'ocr_channel_order': r0.get('ocr_channel_order', ''),
            'ocr_quad_pad_ratio': r0.get('ocr_quad_pad_ratio', ''),
        },
    }

# ── Train/eval overlap check ─────────────────────────────────────
for mname in ['v2_cvreplace', 'v3_best']:
    train_manifest = audit['models'][mname].get('train_manifest_path', '')
    overlap_info = {}
    
    train_path = ROOT / train_manifest if train_manifest and train_manifest != 'N/A' else None
    if train_path and train_path.exists():
        train_paths = set()
        for r in csv.DictReader(open(train_path)):
            fp = r.get('img_path', '').strip()
            if fp: train_paths.add(fp)
        
        for ename, epath in EVAL_MANIFESTS.items():
            if not epath.exists(): continue
            eval_paths = set()
            for r in csv.DictReader(open(epath)):
                fp = r.get('img_path', '').strip()
                if fp: eval_paths.add(fp)
            overlap = train_paths & eval_paths
            overlap_info[ename] = {
                'train_count': len(train_paths),
                'eval_count': len(eval_paths),
                'overlap_count': len(overlap),
                'overlap_pct_of_eval': len(overlap) / max(len(eval_paths), 1) * 100,
                'sample_overlaps': list(overlap)[:5],
            }
    else:
        overlap_info = {'error': f'train manifest not found or not set: {train_manifest}'}
    
    audit['train_eval_overlap'][mname] = overlap_info

# ── Summary consistency check ────────────────────────────────────
ocr_params_set = {
    'ocr_crop_mode': 'obb_warp',
    'ocr_resize_mode': 'letterbox',
    'ocr_resize_kernel': 'nn',
    'ocr_preproc': 'none',
    'ocr_channel_order': 'bgr',
    'ocr_quad_pad_ratio': 0.0,
}
consistency = {'eval_param_consistency': {}, 'issues': []}

# Check all models have same training param
for mname, m in audit['models'].items():
    cfg = m.get('training_config', {})
    for k, expected in ocr_params_set.items():
        val = cfg.get(k, 'N/A')
        if val != expected and val != 'N/A':
            consistency['issues'].append(f'{mname}.training.{k}={val}, expected={expected}')

# Check all eval manifests
for ename, e in audit['eval_sets'].items():
    if 'error' in e: continue
    for k, expected in ocr_params_set.items():
        val = e.get('manifest_ocr_params', {}).get(k, '')
        if val != '' and val != str(expected) and val != expected:
            consistency['issues'].append(f'{ename}.manifest.{k}={val}, expected={expected}')
    
    if e.get('splits', {}).get('train', 0) > 0:
        consistency['issues'].append(f'{ename} has train split samples!')
    
    if e.get('families', {}).get('green8', 0) == 0 and ename.startswith('green'):
        consistency['issues'].append(f'{ename} has no green8 samples!')
    
    if e.get('has_quad', {}).get('0', 0) > 0:
        consistency['issues'].append(f'{ename} has samples without quad!')

# Check overlaps
for mname, oinfo in audit['train_eval_overlap'].items():
    for ename, o in oinfo.items():
        if isinstance(o, dict) and o.get('overlap_count', 0) > 0:
            consistency['issues'].append(f'TRAIN-EVAL OVERLAP: {mname}/{ename}: {o["overlap_count"]} samples ({o["overlap_pct_of_eval"]:.1f}%)')

consistency['all_checks_passed'] = len(consistency['issues']) == 0
audit['consistency'] = consistency

# ── Save ─────────────────────────────────────────────────────────
json.dump(audit, open(OUT_DIR / 'audit_report.json', 'w'), ensure_ascii=False, indent=2)
print(f'Audit saved: {OUT_DIR / "audit_report.json"}', flush=True)
print(f'\nConsistency issues: {len(consistency["issues"])}')
for iss in consistency['issues']:
    print(f'  ISSUE: {iss}')
if consistency['all_checks_passed']:
    print('  ALL CHECKS PASSED')
