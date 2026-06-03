#!/usr/bin/env python3
"""
Proper fusion regression eval using obb_warp preprocessing pipeline.
Uses UnifiedManifestDataset for correct OCR preprocessing,
and warp_quad_to_rect for sidecar input.
"""
import sys, os, csv, json
from collections import Counter
from pathlib import Path
import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/evaluation'))

from load_data import (
    UnifiedManifestDataset, CHARS, CHARS_DICT,
    warp_quad_to_rect, prepare_board_ocr_input_bgr888,
    parse_ccpd_quad_from_name, clip_quad_to_image,
)
from LPRNet import build_lprnet
from evaluation.test_LPRNet import greedy_decode_logits, collate_fn

OUT = ROOT / 'experiments/police_sidecar_synth_fusion_regression_20260603'
OUT.mkdir(parents=True, exist_ok=True)

# Keys
with open(str(ROOT / 'keys/police_keys.txt')) as f:
    all_keys = [l.strip() for l in f if l.strip()]
province_chars = all_keys[:31]
province_map = {c: i for i, c in enumerate(province_chars)}
idx2prov = {i: c for i, c in enumerate(province_chars)}
blank_idx = len(all_keys)
MENG_IDX = province_map['蒙']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── Models ──
ocr_net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(all_keys)+1, dropout_rate=0)
state = torch.load(str(ROOT / 'experiments/police_v2_fullft_officialwarm_20260601/best_LPRNet_model.pth'), map_location='cpu')
if any(k.startswith('module.') for k in state.keys()):
    state = {k.replace('module.', ''): v for k, v in state.items()}
ocr_net.load_state_dict(state)
ocr_net.to(DEVICE)
ocr_net.eval()
print('OCR model loaded.')

import torchvision.models as tvmodels
sidecar = tvmodels.resnet18(weights=None)
sidecar.fc = nn.Linear(sidecar.fc.in_features, 31)
state_s = torch.load(str(ROOT / 'experiments/police_sidecar_realdomain_v1_20260603/runs/lr1e-4_cvw2_realw8/best.pt'), map_location='cpu')
if any(k.startswith('module.') for k in state_s.keys()):
    state_s = {k.replace('module.', ''): v for k, v in state_s.items()}
sidecar.load_state_dict(state_s)
sidecar.to(DEVICE)
sidecar.eval()
print('Sidecar loaded.')

# ── Helper: sidecar inference from warped plate ──
def sidecar_from_warped(warped_bgr):
    """Take a warped plate (BGR), resize to 224x72, gray3, infer province."""
    img = cv2.resize(warped_bgr, (224, 72), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tensor = torch.from_numpy(gray3.astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = sidecar(tensor)
    return int(logits.argmax(dim=1).item()), idx2prov.get(int(logits.argmax(dim=1).item()), '?')

# ── Eval function ──
def evaluate_manifest(name, manifest_path, split_filter=None):
    print(f'\n=== {name} ===')
    
    # Use UnifiedManifestDataset for OCR (94x24 with obb_warp)
    # Override CHARS_DICT with police keys before dataset creation
    import load_data as _ld
    with open(str(ROOT / 'keys/police_keys.txt')) as f:
        police_chars = [l.strip() for l in f if l.strip()]
    _ld.CHARS.clear()
    _ld.CHARS.extend(police_chars)
    _ld.CHARS.append('-')  # CTC blank
    _ld.CHARS_DICT.clear()
    _ld.CHARS_DICT.update({c: i for i, c in enumerate(_ld.CHARS)})
    
    ds_ocr = UnifiedManifestDataset(
        str(manifest_path),
        img_size=[94, 24],
        lpr_max_len=8,
        split_filter=split_filter,
        ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_channel_order='bgr',
        dataset_root=str(ROOT),
    )
    
    loader_ocr = DataLoader(ds_ocr, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    results = []
    for idx, (images, labels, lengths, families) in enumerate(loader_ocr):
        if idx >= len(ds_ocr):
            break
        
        row = ds_ocr.records[idx]
        image = images[0].numpy()  # C, H, W, already normalized
        
        # OCR inference
        with torch.no_grad():
            logits = ocr_net(torch.from_numpy(image).unsqueeze(0).to(DEVICE))
        pred_arr = greedy_decode_logits(logits.cpu().numpy())[0]
        pred_ocr = ''.join(all_keys[i] for i in pred_arr if i < blank_idx)
        
        # Sidecar inference: need to read original image and warp
        img_path = ds_ocr._resolve_img_path(row['img_path'])
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Parse quad
        quad = None
        q_keys = ['quad_1x', 'quad_1y', 'quad_2x', 'quad_2y', 'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y']
        if all(k in row and row[k] for k in q_keys):
            try:
                quad = np.asarray([
                    (float(row['quad_1x']), float(row['quad_1y'])),
                    (float(row['quad_2x']), float(row['quad_2y'])),
                    (float(row['quad_3x']), float(row['quad_3y'])),
                    (float(row['quad_4x']), float(row['quad_4y'])),
                ], dtype=np.float32)
            except (ValueError, TypeError):
                pass
        if quad is None:
            quad = parse_ccpd_quad_from_name(str(img_path))
        if quad is not None:
            img_h, img_w = img.shape[:2]
            quad = clip_quad_to_image(quad, img_w, img_h)
        
        if quad is None:
            # Fallback: use full image for sidecar
            sc_idx, sc_char = sidecar_from_warped(img)
        else:
            # Warp then sidecar
            warped, ordered_quad, matrix = warp_quad_to_rect(img, quad, pad_ratio=0.0, dynamic_pad=True)
            sc_idx, sc_char = sidecar_from_warped(warped)
        
        # Fusion
        gt = row['text']
        if len(pred_ocr) >= 2:
            fused = sc_char + pred_ocr[1:]
        else:
            fused = sc_char + pred_ocr
        
        results.append({
            'gt': gt,
            'ocr': pred_ocr,
            'sc_prov': sc_char,
            'sc_idx': sc_idx,
            'fused': fused,
        })
    
    # Metrics
    n = len(results)
    ocr_exact = sum(1 for r in results if r['ocr'] == r['gt'])
    fusion_exact = sum(1 for r in results if r['fused'] == r['gt'])
    ocr_prov_ok = sum(1 for r in results if len(r['gt']) > 0 and len(r['ocr']) > 0 and r['ocr'][0] == r['gt'][0])
    sc_prov_ok = sum(1 for r in results if len(r['gt']) > 0 and r['sc_prov'] == r['gt'][0])
    fusion_prov_ok = sum(1 for r in results if len(r['gt']) > 0 and len(r['fused']) > 0 and r['fused'][0] == r['gt'][0])
    
    changed = sum(1 for r in results if r['fused'] != r['ocr'])
    changed_right = sum(1 for r in results if r['fused'] != r['ocr'] and r['fused'] == r['gt'])
    changed_wrong = sum(1 for r in results if r['fused'] != r['ocr'] and r['fused'] != r['gt'])
    
    # ocr correct but fusion broke it
    ocr_correct_broken = sum(1 for r in results if r['ocr'] == r['gt'] and r['fused'] != r['gt'])
    
    # non-meng → meng
    nonmeng_to_meng = sum(1 for r in results if r['gt'] and len(r['gt']) > 0 and r['gt'][0] != '蒙' and r['sc_prov'] == '蒙')
    nonmeng_total = sum(1 for r in results if r['gt'] and r['gt'][0] != '蒙')
    
    # Per-province
    per_prov = {}
    for r in results:
        p = r['gt'][0] if r['gt'] else '?'
        if p == '?':
            continue
        if p not in per_prov:
            per_prov[p] = {'total': 0, 'ocr_ok': 0, 'sc_ok': 0, 'fusion_ok': 0}
        per_prov[p]['total'] += 1
        if r['ocr'] == r['gt']:
            per_prov[p]['ocr_ok'] += 1
        if r['gt'] and r['sc_prov'] == r['gt'][0]:
            per_prov[p]['sc_ok'] += 1
        if r['fused'] == r['gt']:
            per_prov[p]['fusion_ok'] += 1
    
    # Confusion matrix for sidecar
    confusion = Counter()
    for r in results:
        if r['gt'] and r['sc_prov'] != r['gt'][0]:
            confusion[f'{r["gt"][0]}->{r["sc_prov"]}'] += 1
    
    # Province-specific acc for target provinces
    targets = ['蒙', '青', '琼', '黑', '赣']
    target_acc = {}
    for t in targets:
        o = sum(1 for r in results if r['gt'] and r['gt'][0] == t and r['ocr'] == r['gt'])
        s = sum(1 for r in results if r['gt'] and r['gt'][0] == t and r['sc_prov'] == t)
        f = sum(1 for r in results if r['gt'] and r['gt'][0] == t and r['fused'] == r['gt'])
        tot = sum(1 for r in results if r['gt'] and r['gt'][0] == t)
        target_acc[t] = {'total': tot, 'ocr_exact': o, 'sc_prov': s, 'fusion_exact': f}
    
    metrics = {
        'total': n,
        'ocr_exact': ocr_exact, 'ocr_exact_pct': ocr_exact/max(1,n)*100,
        'fusion_exact': fusion_exact, 'fusion_exact_pct': fusion_exact/max(1,n)*100,
        'ocr_prov_ok': ocr_prov_ok, 'ocr_prov_pct': ocr_prov_ok/max(1,n)*100,
        'sidecar_prov_ok': sc_prov_ok, 'sidecar_prov_pct': sc_prov_ok/max(1,n)*100,
        'fusion_prov_ok': fusion_prov_ok, 'fusion_prov_pct': fusion_prov_ok/max(1,n)*100,
        'changed': changed, 'changed_right': changed_right, 'changed_wrong': changed_wrong,
        'ocr_correct_broken_by_fusion': ocr_correct_broken,
        'nonmeng_to_meng': nonmeng_to_meng, 'nonmeng_total': nonmeng_total,
        'nonmeng_to_meng_pct': nonmeng_to_meng/max(1,nonmeng_total)*100,
        'per_province': {k: {'total': v['total'], 'ocr_acc': v['ocr_ok']/v['total']*100 if v['total'] else 0,
                             'sc_prov_acc': v['sc_ok']/v['total']*100 if v['total'] else 0,
                             'fusion_acc': v['fusion_ok']/v['total']*100 if v['total'] else 0}
                         for k, v in sorted(per_prov.items())},
        'target_provinces': target_acc,
        'sidecar_confusion': dict(confusion.most_common(20)),
    }
    
    print(f'  Total: {n}')
    print(f'  OCR exact:        {ocr_exact}/{n} = {ocr_exact/max(1,n)*100:.2f}%')
    print(f'  Fusion exact:     {fusion_exact}/{n} = {fusion_exact/max(1,n)*100:.2f}%')
    print(f'  Delta (fusion-ocr): {fusion_exact-ocr_exact}')
    print(f'  OCR prov:         {ocr_prov_ok}/{n} = {ocr_prov_ok/max(1,n)*100:.2f}%')
    print(f'  Sidecar prov:     {sc_prov_ok}/{n} = {sc_prov_ok/max(1,n)*100:.2f}%')
    print(f'  Fusion prov:      {fusion_prov_ok}/{n} = {fusion_prov_ok/max(1,n)*100:.2f}%')
    print(f'  Changed right:    {changed_right}')
    print(f'  Changed wrong:    {changed_wrong}')
    print(f'  OCR correct→broken: {ocr_correct_broken}')
    print(f'  Non-meng→meng:    {nonmeng_to_meng}/{nonmeng_total}')
    if confusion:
        print(f'  Top confusion:   {confusion.most_common(5)}')
    for t, v in target_acc.items():
        print(f'  {t}: total={v["total"]} ocr_exact={v["ocr_exact"]}/{v["total"]} sc_prov={v["sc_prov"]}/{v["total"]} fusion={v["fusion_exact"]}/{v["total"]}')
    
    # Write CSV
    csv_path = OUT / f'predictions_{name.replace("/", "_")}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['gt', 'ocr', 'sc_prov', 'fused', 'ocr_exact', 'fusion_exact', 'sc_prov_ok'])
        for r in results:
            w.writerow([r['gt'], r['ocr'], r['sc_prov'], r['fused'],
                        'Y' if r['ocr'] == r['gt'] else 'N',
                        'Y' if r['fused'] == r['gt'] else 'N',
                        'Y' if r['gt'] and r['sc_prov'] == r['gt'][0] else 'N'])
    
    # Write JSON
    json_path = OUT / f'metrics_{name.replace("/", "_")}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f'  Metrics saved to {json_path}')
    return metrics

# ═══ 1. Real same-plate (mgC0001J + police_dump) ═══
# Do this separately using the direct PPM approach (already validated)
print('\n=== Real same-plate (using direct PPM eval) ===')
# Re-run using the previous fusion validation results
import csv as csv_mod
real_preds = []
for dump_path, label in [
    ('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/polic_dump_mgC0001J', 'mgC0001J'),
    ('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/police_dump', 'police_dump'),
]:
    with open(os.path.join(dump_path, 'index.csv')) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            sid, fid = int(row['sample_id']), int(row['frame_id'])
            ocrin_p = os.path.join(dump_path, f'ocrin_{sid:04d}_f{fid:06d}.ppm')
            fc224_p = os.path.join(dump_path, f'fc224_{sid:04d}_f{fid:06d}.ppm')
            
            # OCR on ocrin (BGR-corrected PPM)
            img_o = cv2.imread(ocrin_p)
            if img_o is None: continue
            img_o = img_o[:,:,::-1].copy()
            h, w = img_o.shape[:2]
            if h != 24 or w != 94:
                img_o = cv2.resize(img_o, (94, 24), interpolation=cv2.INTER_LINEAR)
            x = img_o.astype('float32'); x = (x - 127.5) * 0.0078125
            x = np.transpose(x, (2, 0, 1))[None, :]
            with torch.no_grad():
                logits = ocr_net(torch.from_numpy(x).to(DEVICE))
            pred_arr = greedy_decode_logits(logits.cpu().numpy())[0]
            pred_ocr = ''.join(all_keys[i] for i in pred_arr if i < blank_idx)
            
            # Sidecar on fc224 (BGR-corrected PPM)
            img_fc = cv2.imread(fc224_p)
            img_fc = img_fc[:,:,::-1].copy()
            sc_idx, sc_char = '', '?'
            if img_fc is not None:
                SC_IMG = cv2.resize(img_fc, (224, 72), interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(SC_IMG, cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                tensor = torch.from_numpy(gray3.astype('float32') / 255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = sidecar(tensor)
                sc_idx = int(logits.argmax(dim=1).item())
                sc_char = idx2prov.get(sc_idx, '?')
            
            fused = sc_char + pred_ocr[1:] if len(pred_ocr) >= 2 else sc_char + pred_ocr
            
            # GT = 蒙C0001警 for all
            gt = '蒙C0001警'
            real_preds.append({
                'source': label, 'frame': fid,
                'gt': gt, 'ocr': pred_ocr, 'sc_prov': sc_char, 'fused': fused,
            })

n = len(real_preds)
ocr_exact = sum(1 for r in real_preds if r['ocr'] == r['gt'])
fusion_exact = sum(1 for r in real_preds if r['fused'] == r['gt'])
ocr_prov = sum(1 for r in real_preds if r['ocr'] and r['ocr'][0] == '蒙')
sc_prov = sum(1 for r in real_preds if r['sc_prov'] == '蒙')
fusion_prov = sum(1 for r in real_preds if r['fused'] and r['fused'][0] == '蒙')

print(f'  Real same-plate total: {n}')
print(f'  OCR exact:   {ocr_exact}/{n} = {ocr_exact/max(1,n)*100:.2f}%')
print(f'  Fusion exact:{fusion_exact}/{n} = {fusion_exact/max(1,n)*100:.2f}%')
print(f'  OCR prov=蒙:  {ocr_prov}/{n} = {ocr_prov/max(1,n)*100:.2f}%')
print(f'  SC prov=蒙:   {sc_prov}/{n} = {sc_prov/max(1,n)*100:.2f}%')
print(f'  Fusion prov=蒙:{fusion_prov}/{n} = {fusion_prov/max(1,n)*100:.2f}%')

# Body error analysis for fusion-failed cases
body_errors = 0
for r in real_preds:
    if r['fused'] != r['gt'] and len(r['fused']) > 1 and len(r['gt']) > 1:
        body_fused = r['fused'][1:]
        body_gt = r['gt'][1:]
        if body_fused != body_gt:
            body_errors += 1
print(f'  Fusion failed due to body error: {body_errors}')

real_metrics = {
    'total': n, 'ocr_exact': ocr_exact, 'fusion_exact': fusion_exact,
    'ocr_prov_meng': ocr_prov, 'sc_prov_meng': sc_prov, 'fusion_prov_meng': fusion_prov,
    'body_errors_in_fusion_failures': body_errors,
}
with open(OUT / 'metrics_real_same_plate.json', 'w', encoding='utf-8') as f:
    json.dump(real_metrics, f, ensure_ascii=False, indent=2)

csv_path = OUT / 'predictions_real_same_plate.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['source', 'frame', 'gt', 'ocr', 'sc_prov', 'fused'])
    for r in real_preds:
        w.writerow([r['source'], r['frame'], r['gt'], r['ocr'], r['sc_prov'], r['fused']])
print(f'  CSV: {csv_path}')

# ═══ 2 & 3. Synthetic val_clean and val_hard ═══
for split in ['val_clean', 'val_hard']:
    mani = ROOT / 'manifests_rebased/special_split_v2_20260601' / f'{split}_police.csv'
    evaluate_manifest(split, mani)

# Combine all metrics
all_metrics = {'real_same_plate': real_metrics}
for split in ['val_clean', 'val_hard']:
    mj = OUT / f'metrics_{split}.json'
    if mj.exists():
        all_metrics[split] = json.load(open(mj))

with open(OUT / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump(all_metrics, f, ensure_ascii=False, indent=2)
print(f'\nAll metrics saved to {OUT / "metrics.json"}')
print('Done.')
