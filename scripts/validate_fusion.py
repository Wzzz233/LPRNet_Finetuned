#!/usr/bin/env python3
"""Fusion validation: main OCR + RealDomain V1 sidecar."""
import sys, os, csv, json
from collections import Counter
from pathlib import Path
import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tvmodels

ROOT = '/home/wzzz/LPRNet'
OUT = os.path.join(ROOT, 'experiments/police_sidecar_fusion_validation_20260603')
os.makedirs(OUT, exist_ok=True)

# Keys / province mapping
with open(os.path.join(ROOT, 'keys/police_keys.txt')) as f:
    all_keys = [l.strip() for l in f if l.strip()]
province_chars = all_keys[:31]
province_map = {c: i for i, c in enumerate(province_chars)}
idx2prov = {i: c for i, c in enumerate(province_chars)}
MENG_IDX = province_map['蒙']
BLANK_IDX = len(all_keys)  # 66

# ── Models ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# Main OCR
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src/evaluation'))
from LPRNet import build_lprnet
from evaluation.test_LPRNet import greedy_decode_logits

ocr_net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(all_keys)+1, dropout_rate=0)
state = torch.load(os.path.join(ROOT, 'experiments/police_v2_fullft_officialwarm_20260601/best_LPRNet_model.pth'), map_location='cpu')
if any(k.startswith('module.') for k in state.keys()):
    state = {k.replace('module.', ''): v for k, v in state.items()}
ocr_net.load_state_dict(state)
ocr_net.to(DEVICE)
ocr_net.eval()

# Sidecar
sidecar = tvmodels.resnet18(weights=None)
sidecar.fc = nn.Linear(sidecar.fc.in_features, 31)
state_s = torch.load(os.path.join(ROOT, 'experiments/police_sidecar_realdomain_v1_20260603/runs/lr1e-4_cvw2_realw8/best.pt'), map_location='cpu')
if any(k.startswith('module.') for k in state_s.keys()):
    state_s = {k.replace('module.', ''): v for k, v in state_s.items()}
sidecar.load_state_dict(state_s)
sidecar.to(DEVICE)
sidecar.eval()
print('Models loaded.')

# ── Helpers ──
def read_img_bgr(path):
    p = Path(path)
    if p.suffix.lower() == '.ppm':
        img = cv2.imread(str(path))
        return img[:,:,::-1].copy() if img is not None else None
    return cv2.imread(str(path))

def ocr_infer(img_bgr):
    h, w = img_bgr.shape[:2]
    if h != 24 or w != 94:
        img_bgr = cv2.resize(img_bgr, (94, 24), interpolation=cv2.INTER_LINEAR)
    x = img_bgr.astype('float32')
    x = (x - 127.5) * 0.0078125
    x = np.transpose(x, (2, 0, 1))[None, :]
    with torch.no_grad():
        logits = ocr_net(torch.from_numpy(x).to(DEVICE))
    pred = greedy_decode_logits(logits.cpu().numpy())
    return ''.join(all_keys[i] for i in pred[0] if i < BLANK_IDX)

def sidecar_infer(img_bgr):
    img = cv2.resize(img_bgr, (224, 72), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tensor = torch.from_numpy(gray3.astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = sidecar(tensor)
    idx = int(logits.argmax(dim=1).item())
    return idx, idx2prov.get(idx, '?')

def fusion(ocr_pred, sidecar_prov_char):
    if len(ocr_pred) < 2:
        return sidecar_prov_char + ocr_pred  # fallback
    return sidecar_prov_char + ocr_pred[1:]

def fmt_rate(c, t):
    return f'{c}/{t} = {c/t*100:.2f}%' if t > 0 else 'N/A'

# ── Evaluate one set ──
def evaluate_set(name, items):
    """items: list of dict with keys: ocrin_path, fc224_path (optional), ref (optional)"""
    results = []
    for item in items:
        # Main OCR
        img_ocr = read_img_bgr(item['ocrin_path'])
        if img_ocr is None:
            continue
        p_ocr = ocr_infer(img_ocr)
        
        # Sidecar (if fc224 available)
        p_sc_char = '?'
        p_sc_idx = -1
        if item.get('fc224_path') and os.path.exists(item['fc224_path']):
            img_sc = read_img_bgr(item['fc224_path'])
            if img_sc is not None:
                p_sc_idx, p_sc_char = sidecar_infer(img_sc)
        
        # Fusion
        p_fused = fusion(p_ocr, p_sc_char)
        
        ref = item.get('ref', '')
        
        results.append({
            'file_ocrin': os.path.basename(item['ocrin_path']),
            'file_fc224': os.path.basename(item.get('fc224_path', '')),
            'ref': ref,
            'ocr_pred': p_ocr,
            'ocr_prov': p_ocr[0] if p_ocr else '?',
            'ocr_body': p_ocr[1:] if len(p_ocr) > 1 else '',
            'sidecar_prov': p_sc_char,
            'sidecar_idx': p_sc_idx,
            'fused': p_fused,
            'has_sidecar': p_sc_char != '?',
        })
    
    if not results:
        return
    
    n = len(results)
    has_sc = [r for r in results if r['has_sidecar']]
    
    # Main OCR metrics
    ocr_exact = sum(1 for r in results if r['ref'] and r['ocr_pred'] == r['ref'])
    ocr_prov_ok = sum(1 for r in results if r['ref'] and len(r['ref']) > 0 and r['ocr_prov'] == r['ref'][0])
    
    # Body accuracy (excl province)
    body_correct = 0
    body_total = 0
    for r in results:
        if not r['ref'] or len(r['ref']) < 2 or len(r['ocr_pred']) < 2:
            continue
        ref_body = r['ref'][1:]
        pred_body = r['ocr_pred'][1:]
        for i in range(min(len(ref_body), len(pred_body))):
            body_correct += 1 if ref_body[i] == pred_body[i] else 0
            body_total += 1
    
    # Sidecar metrics
    sc_prov_ok = sum(1 for r in has_sc if r['ref'] and len(r['ref']) > 0 and r['sidecar_prov'] == r['ref'][0])
    
    # Fusion metrics
    fused_exact = sum(1 for r in results if r['ref'] and r['fused'] == r['ref'])
    fused_changed = sum(1 for r in results if r['fused'] != r['ocr_pred'])
    fused_changed_correct = sum(1 for r in results if r['ref'] and r['fused'] == r['ref'] and r['fused'] != r['ocr_pred'])
    fused_changed_wrong = sum(1 for r in results if r['ref'] and r['fused'] != r['ref'] and r['fused'] != r['ocr_pred'])
    
    # Fusion: non-meng → meng
    nonmeng_to_meng = sum(1 for r in has_sc if r['ref'] and len(r['ref']) > 0 and r['ref'][0] != '蒙' and r['sidecar_prov'] == '蒙')
    
    # Print
    has_ref = [r for r in results if r['ref']]
    print(f'\n=== {name} ===')
    print(f'  Total: {n} (ref: {len(has_ref)}, sidecar: {len(has_sc)})')
    print(f'  Main OCR exact:      {fmt_rate(ocr_exact, len(has_ref))}')
    print(f'  Main OCR prov ok:    {fmt_rate(ocr_prov_ok, len(has_ref))}')
    print(f'  Main OCR body acc:   {fmt_rate(body_correct, body_total)}')
    if has_sc:
        print(f'  Sidecar prov ok:     {fmt_rate(sc_prov_ok, len(has_sc))}')
        print(f'  Fusion exact:        {fmt_rate(fused_exact, len(has_ref))}')
        print(f'  Changed total:       {fused_changed}')
        print(f'  Changed correct:     {fused_changed_correct}')
        print(f'  Changed wrong:       {fused_changed_wrong}')
        print(f'  Non-meng->meng fp:   {nonmeng_to_meng}')
    
    # Sidecar-only: prov distribution
    if has_sc:
        sc_dist = Counter(r['sidecar_prov'] for r in has_sc)
        print(f'  Sidecar prov dist:   {dict(sc_dist.most_common())}')
    
    # Write CSV
    fieldnames = ['file_ocrin', 'file_fc224', 'ref', 'ocr_pred', 'ocr_prov', 'ocr_body', 'sidecar_prov', 'fused']
    csv_path = os.path.join(OUT, f'predictions_{name.replace("/", "_")}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f'  CSV: {csv_path}')
    
    return results

print('Starting fusion validation...')

# ═══ A. mgC0001J ═══
dump_mg = '/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/polic_dump_mgC0001J'
mg_items = []
with open(os.path.join(dump_mg, 'index.csv')) as f:
    for row in csv.DictReader(f):
        sid, fid = int(row['sample_id']), int(row['frame_id'])
        ocrin = os.path.join(dump_mg, f'ocrin_{sid:04d}_f{fid:06d}.ppm')
        fc224 = os.path.join(dump_mg, f'fc224_{sid:04d}_f{fid:06d}.ppm')
        mg_items.append({'ocrin_path': ocrin, 'fc224_path': fc224, 'ref': '蒙C0001警'})
evaluate_set('mgC0001J', mg_items)

# ═══ B. police_dump ═══
dump_pd = '/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/police_dump'
pd_items = []
with open(os.path.join(dump_pd, 'index.csv')) as f:
    for row in csv.DictReader(f):
        sid, fid = int(row['sample_id']), int(row['frame_id'])
        ocrin = os.path.join(dump_pd, f'ocrin_{sid:04d}_f{fid:06d}.ppm')
        fc224 = os.path.join(dump_pd, f'fc224_{sid:04d}_f{fid:06d}.ppm')
        ref = row['app_text'] if row.get('app_text', '') else ''
        pd_items.append({'ocrin_path': ocrin, 'fc224_path': fc224, 'ref': ref})
evaluate_set('police_dump', pd_items)

print('\nFusion validation complete.')
