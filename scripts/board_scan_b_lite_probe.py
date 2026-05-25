#!/usr/bin/env python3
"""Canonical board scan for B-lite balanced probe best checkpoint.
Protocol matches board_scan_ablation_reference.py exactly.
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
SEG_B_START, SEG_B_END = 11, 40

def decode_ctc(prebs):
    results = []
    for bi in range(prebs.shape[0]):
        preb = prebs[bi, :, :]
        preb_label = [int(np.argmax(preb[:, t], axis=0)) for t in range(preb.shape[1])]
        decoded = []; prev = preb_label[0]
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
    h, w = img.shape[:2]
    if w != 94 or h != 24: img = cv2.resize(img, (94, 24))
    img = img.astype('float32'); img -= 127.5; img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    with torch.no_grad():
        prebs = forward_family_logits(net, torch.from_numpy(img).unsqueeze(0).to(device), sample_families=['green8'])
    return decode_ctc(prebs.cpu().numpy())[0]

def compute_metrics(preds, gt):
    n = len(preds)
    pp_ref = gt[1:]
    exact_ok = sum(1 for p in preds if p == gt)
    fc_ok = sum(1 for p in preds if p and len(p) > 0 and p[0] == gt[0])
    char_accs = [sum(1 for p_, g_ in zip(p, gt) if p_ == g_) / max(len(gt), 1) for p in preds]
    pp_exact_ok = sum(1 for p in preds if len(p) > 1 and p[1:] == pp_ref)
    pp_char_accs = [sum(1 for p_, g_ in zip(p[1:], pp_ref) if p_ == g_) / max(len(pp_ref), 1) if len(p) > 1 else 0.0 for p in preds]
    prov_confusion = Counter()
    for p in preds:
        if len(p) > 0 and p[0] != gt[0]:
            prov_confusion[p[0]] += 1
    return {
        'n': n, 'exact': round(exact_ok/n*100, 1), 'fc': round(fc_ok/n*100, 1),
        'char': round(sum(char_accs)/n*100, 1), 'pp_exact': round(pp_exact_ok/n*100, 1),
        'pp_char': round(sum(pp_char_accs)/n*100, 1),
        'mean_len': round(np.mean([len(p) for p in preds]), 2),
        'len_err_rate': round(sum(1 for l in [len(p) for p in preds] if l != len(gt))/n*100, 1),
        'prov_confusion': dict(prov_confusion.most_common(10)),
    }

def evaluate_dump(net, dump_path, gt, seg_bounds=None):
    ocrin_files = sorted([f for f in os.listdir(dump_path) if f.startswith('ocrin_') and f.endswith('.ppm')],
                         key=lambda x: int(x.split('_')[1]))
    preds = [infer_one(net, dump_path / f) for f in ocrin_files]
    full = compute_metrics(preds, gt)
    seg = {}
    if seg_bounds:
        seg = compute_metrics(preds[seg_bounds[0]:seg_bounds[1]+1], gt)
    return {'full': full, 'seg_B': seg, 'predictions': preds}

ckpt_path = ROOT / 'experiments/b_lite_balanced_probe_20260510/best_LPRNet_model.pth'
net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS),
                             dropout_rate=0.5, enhanced_green_head='expD', pos0_head_cols=0)
net.load_state_dict(torch.load(str(ckpt_path), map_location='cpu'), strict=False)
net.to(device); net.eval()

d1 = evaluate_dump(net, POS_OCR_DUMP, GT1, seg_bounds=(SEG_B_START, SEG_B_END))
d2 = evaluate_dump(net, POS_OCR_DUMP_2, GT2)

# Compare with R50 baseline
r50_baseline = {
    'seg_pp_char': 51.9, 'seg_len_err': 70.0, 'seg_pp_exact': 20.0,
    'st_fc': 0.0, 'st_wan': 33, 'st_pp_exact': 67.5, 'st_full_exact': 0.0,
}

s = d1['seg_B']; t = d2['full']
prov = t.get('prov_confusion', {})
wan = prov.get('皖', 0)
total_conf = sum(prov.values()) if prov else 1
pc_top1 = max(prov.values()) / total_conf * 100 if prov else 0
n_distinct = len(prov)

print("=" * 70)
print("B-LITE BALANCED PROBE vs R50 (canonical protocol)")
print("=" * 70)

print(f"\n--- Tilt ---")
print(f"  Metric           R50       B-lite     Change")
print(f"  seg_B pp_char    {r50_baseline['seg_pp_char']:.1f}%     {s['pp_char']:.1f}%      {s['pp_char']-r50_baseline['seg_pp_char']:+.1f}pp")
print(f"  seg_B len_err    {r50_baseline['seg_len_err']:.1f}%     {s['len_err_rate']:.1f}%      {r50_baseline['seg_len_err']-s['len_err_rate']:+.1f}pp")
print(f"  seg_B pp_exact   {r50_baseline['seg_pp_exact']:.1f}%     {s['pp_exact']:.1f}%      {s['pp_exact']-r50_baseline['seg_pp_exact']:+.1f}pp")

print(f"\n--- Province ---")
print(f"  province_top1_acc (fc):  {t['fc']:.1f}% (R50: {r50_baseline['st_fc']:.1f}%)")
print(f"  皖 confusion count:     {wan}/{t['n']} (R50: {r50_baseline['st_wan']}/{t['n']})")
print(f"  PC_top1_concentration:  {pc_top1:.1f}% (R50: {r50_baseline['st_wan']/t['n']*100:.1f}%)")
print(f"  Distinct confused:      {n_distinct} (R50: 4)")
print(f"  Province confusion:     {prov}")

print(f"\n--- Full Plate ---")
print(f"  dump2 full exact:       {t['exact']:.1f}% (R50: {r50_baseline['st_full_exact']:.1f}%)")
print(f"  dump2 pp_exact:         {t['pp_exact']:.1f}% (R50: {r50_baseline['st_pp_exact']:.1f}%)")

# Judgment
tilt_ok = s['pp_char'] >= 49.0 and s['len_err_rate'] <= 75.0
prov_signal = t['fc'] > 0 or wan < 25  # any fc improvement or wan reduction

if not tilt_ok:
    label = "B_LITE_COLLAPSES_GREEN8"
elif prov_signal:
    label = "B_LITE_SHOWS_PROVINCE_SIGNAL"
else:
    label = "B_LITE_NO_PROVINCE_GAIN"

print(f"\n--- Final Judgment ---")
print(f"  Tilt guard (pp_char>=49%, len_err<=75%): {'PASS' if tilt_ok else 'FAIL'}")
print(f"  Province signal (fc>0 or 皖<25): {'YES' if prov_signal else 'NO'}")
print(f"  Label: {label}")

# Save
out_path = ROOT / 'experiments/mix_source_audit_20260510/b_lite_probe_board_scan.json'
with open(out_path, 'w') as f:
    json.dump({'best_LPRNet_model.pth': {'pos_ocr_dump': d1, 'static_control': d2}}, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_path}")
