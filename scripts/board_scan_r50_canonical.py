#!/usr/bin/env python3
"""R50 board scan using canonical protocol (matches board_scan_ablation_reference.py).
   GT hardcoded: 苏BF01111 / 京AD06088
   pp_exact/pp_char: post-province
   seg_B: indices 11-40
   Image: cv2.imread (BGR)
   Model: forward_family_logits
"""
import sys, json, os, torch, cv2, numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/training'))
from load_data import CHARS
from train_LPRNet import forward_family_logits
from LPRNet_multihead import build_lprnet_multihead

device = torch.device('cuda:0')
BLANK = len(CHARS) - 1
GT1 = '苏BF01111'
GT2 = '京AD06088'

POS_OCR_DUMP = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump')
POS_OCR_DUMP_2 = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2')
SEG_B_START, SEG_B_END = 11, 40  # indices 11-40 inclusive = 30 frames

CKPTS = [
    'best_LPRNet_model.pth', 'Final_LPRNet_model.pth', 'last_LPRNet_model.pth',
    'LPRNet__iteration_2000.pth', 'LPRNet__iteration_4000.pth', 'LPRNet__iteration_6000.pth',
]

def decode_ctc(prebs):
    results = []
    for bi in range(prebs.shape[0]):
        preb = prebs[bi, :, :]
        preb_label = [int(np.argmax(preb[:, t], axis=0)) for t in range(preb.shape[1])]
        decoded = []
        prev = preb_label[0]
        if prev != BLANK: decoded.append(prev)
        for c in preb_label[1:]:
            if c == prev or c == BLANK:
                if c == BLANK: prev = c
                continue
            decoded.append(c); prev = c
        results.append(''.join(CHARS[i] for i in decoded if 0 <= i < len(CHARS)))
    return results

def infer_one(net, ocrin_path):
    img = cv2.imread(str(ocrin_path))
    if img is None: return ''
    h, w = img.shape[:2]
    if w != 94 or h != 24:
        img = cv2.resize(img, (94, 24))
    img = img.astype('float32')
    img -= 127.5; img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    batch = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prebs = forward_family_logits(net, batch, sample_families=['green8'])
    return decode_ctc(prebs.cpu().numpy())[0]

def load_model(ckpt_path):
    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS),
                                 dropout_rate=0.5, enhanced_green_head='expD', pos0_head_cols=0)
    net.load_state_dict(torch.load(str(ckpt_path), map_location='cpu'), strict=False)
    net.to(device); net.eval()
    return net

def compute_metrics(preds, gt):
    n = len(preds)
    if n == 0: return {}
    pp_ref = gt[1:]
    exact_ok = sum(1 for p in preds if p == gt)
    fc_ok = sum(1 for p in preds if p and len(p) > 0 and p[0] == gt[0])
    char_accs = [sum(1 for p_, g_ in zip(p, gt) if p_ == g_) / max(len(gt), 1) for p in preds]
    pp_exact_ok = sum(1 for p in preds if len(p) > 1 and p[1:] == pp_ref)
    pp_char_accs = []
    for p in preds:
        if len(p) > 1:
            ppc = sum(1 for p_, g_ in zip(p[1:], pp_ref) if p_ == g_) / max(len(pp_ref), 1)
        else:
            ppc = 0.0
        pp_char_accs.append(ppc)
    lengths = [len(p) for p in preds]
    prov_confusion = Counter()
    for p in preds:
        if len(p) > 0 and p[0] != gt[0]:
            prov_confusion[p[0]] += 1
    return {
        'n': n, 'exact': round(exact_ok / n * 100, 1),
        'fc': round(fc_ok / n * 100, 1),
        'char': round(sum(char_accs) / n * 100, 1),
        'pp_exact': round(pp_exact_ok / n * 100, 1),
        'pp_char': round(sum(pp_char_accs) / n * 100, 1),
        'mean_len': round(sum(lengths) / n, 2),
        'len_err_rate': round(sum(1 for l in lengths if l != len(gt)) / n * 100, 1),
        'prov_confusion': dict(prov_confusion.most_common(10)),
    }

def evaluate_dump(net, dump_path, gt, seg_bounds=None):
    ocrin_files = sorted([f for f in os.listdir(dump_path)
                          if f.startswith('ocrin_') and f.endswith('.ppm')],
                         key=lambda x: int(x.split('_')[1]))
    preds = []
    for fname in ocrin_files:
        pred = infer_one(net, dump_path / fname)
        preds.append(pred)
    full = compute_metrics(preds, gt)
    seg = {}
    if seg_bounds:
        start, end = seg_bounds
        seg_preds = preds[start:end+1]
        seg = compute_metrics(seg_preds, gt)
    return {'full': full, 'seg_B': seg, 'predictions': preds}

OUT_DIR = ROOT / 'experiments/mix_source_audit_20260510'
EXP_DIR = ROOT / 'experiments/a_ratio_r50_20260510'

print(f'{"="*60}')
print(f'R50 BOARD SCAN (canonical protocol)')
print(f'{"="*60}')

r50_data = {}
for ckpt_name in CKPTS:
    ckpt_path = EXP_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f'  SKIP {ckpt_name}')
        continue
    print(f'  {ckpt_name}...', end=' ', flush=True)
    net = load_model(ckpt_path)
    d1 = evaluate_dump(net, POS_OCR_DUMP, GT1, seg_bounds=(SEG_B_START, SEG_B_END))
    d2 = evaluate_dump(net, POS_OCR_DUMP_2, GT2)
    r50_data[ckpt_name] = {'pos_ocr_dump': d1, 'static_control': d2}
    s = d1['seg_B']; t = d2['full']
    print(f'seg: pp_char={s["pp_char"]:.1f} len_err={s["len_err_rate"]:.1f} '
          f'static: pp_exact={t["pp_exact"]:.1f} bias={list(t["prov_confusion"].keys())[:3]}')
    del net; torch.cuda.empty_cache()

# Save
out_path = OUT_DIR / 'a_ratio_board_scan_r50.json'
json.dump(r50_data, open(out_path, 'w'), ensure_ascii=False, indent=2)
print(f'\nSaved: {out_path}')

# Board-optimal
cands = [(ckpt, d['pos_ocr_dump']['seg_B']['pp_char'],
          d['pos_ocr_dump']['seg_B']['len_err_rate'],
          d['pos_ocr_dump']['seg_B']['pp_exact'],
          d['static_control']['full']['pp_exact'],
          d['static_control']['full'].get('prov_confusion', {}))
         for ckpt, d in r50_data.items()]
cands.sort(key=lambda x: (-x[1], x[2]))
best = cands[0]
ckpt, pp_char, len_err, pp_ex, st_pp_ex, prov = best
bias = ' '.join(f'{k}={v}' for k,v in list(prov.items())[:3]) if prov else 'none'

print(f'\n{"="*50}')
print(f'R50 Board-Optimal: {ckpt}')
print(f'  seg_B pp_char={pp_char:.1f}%  len_err={len_err:.1f}%  pp_exact={pp_ex:.1f}%')
print(f'  static pp_exact={st_pp_ex:.1f}%  bias={bias}')
print(f'{"="*50}')

# Full leaderboard
print(f'\nR50 Leaderboard (sorted by seg_B pp_char):')
print(f'{"Checkpoint":30s} {"pp_char":>8s} {"len_err":>8s} {"pp_exact":>8s} {"st_pp_ex":>8s} {"bias":>20s}')
print('-' * 80)
for ckpt, pp_char, len_err, pp_ex, st_pp_ex, prov in cands:
    bias = ' '.join(f'{k}={v}' for k,v in list(prov.items())[:3]) if prov else 'none'
    print(f'{ckpt:30s} {pp_char:>7.1f}% {len_err:>7.1f}% {pp_ex:>7.1f}% {st_pp_ex:>7.1f}% {bias:>20s}')
