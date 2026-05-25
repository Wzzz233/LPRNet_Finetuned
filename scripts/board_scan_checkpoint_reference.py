#!/usr/bin/env python3
# ============================================================
# REPO-CANONICAL REFERENCE: Board-Centric Checkpoint Scan
# ============================================================
# This is the authoritative reference implementation for
# board-centric scanning of all 6 checkpoints per experiment.
#
# Protocol (DO NOT DEVIATE):
#   - Same as board_scan_ablation_reference.py
#   - Loads all 6 checkpoints: best, Final, last, iter_2000/4000/6000
#   - Saves per-checkpoint JSON with full & seg_B metrics
#
# This file supersedes /tmp/board_checkpoint_scan.py.
# ============================================================
#!/usr/bin/env python3
"""Board-centric checkpoint scan for A and B stage1.
Loads each checkpoint, evaluates on board dumps, saves structured results."""
import sys, json, os, torch, cv2, numpy as np
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'training'))
from load_data import CHARS
from train_LPRNet import forward_family_logits
from LPRNet_multihead import build_lprnet_multihead

device = torch.device('cuda:0')
BLANK = len(CHARS) - 1
GT1 = '苏BF01111'
GT2 = '京AD06088'

# Dump configs
POS_OCR_DUMP = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump')
POS_OCR_DUMP_2 = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2')

SEG_B_START, SEG_B_END = 11, 40  # tilt regime

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

def evaluate_dump(net, dump_path, gt, seg_bounds=None):
    ocrin_files = sorted([f for f in os.listdir(dump_path)
                          if f.startswith('ocrin_') and f.endswith('.ppm')],
                         key=lambda x: int(x.split('_')[1]))
    preds = []
    for fname in ocrin_files:
        pred = infer_one(net, dump_path / fname)
        preds.append(pred)

    full_metrics = compute_metrics(preds, gt)
    seg_metrics = {}
    if seg_bounds:
        start, end = seg_bounds
        seg_preds = preds[start:end+1]
        seg_metrics = compute_metrics(seg_preds, gt)

    return {
        'n': len(preds),
        'full': compute_metrics(preds, gt),
        'seg_B': seg_metrics,
        'predictions': preds if len(preds) <= 50 else preds[:50],
    }

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
            pp = p[1:]
            ppc = sum(1 for p_, g_ in zip(pp, pp_ref) if p_ == g_) / max(len(pp_ref), 1)
        else:
            ppc = 0.0
        pp_char_accs.append(ppc)
    lengths = [len(p) for p in preds]
    len_err = sum(1 for l in lengths if l != len(gt))
    prov_confusion = Counter()
    for p in preds:
        if len(p) > 0 and p[0] != gt[0]:
            prov_confusion[p[0]] += 1

    return {
        'n': n,
        'exact': exact_ok / n * 100,
        'fc': fc_ok / n * 100,
        'char': sum(char_accs) / n * 100,
        'pp_exact': pp_exact_ok / n * 100,
        'pp_char': sum(pp_char_accs) / n * 100,
        'mean_len': sum(lengths) / n,
        'len_err_rate': len_err / n * 100,
        'prov_confusion': dict(prov_confusion.most_common(10)),
    }

# ─── Scan A ───
A_DIR = ROOT / 'experiments/green_backbone_mix_branchA_refine_only_20260510'
A_CKPTS = [
    'best_LPRNet_model.pth', 'Final_LPRNet_model.pth', 'last_LPRNet_model.pth',
    'LPRNet__iteration_2000.pth', 'LPRNet__iteration_4000.pth', 'LPRNet__iteration_6000.pth',
]
print('='*70)
print('SCANNING BRANCH A')
print('='*70)
scan_A = {}
for ckpt_name in A_CKPTS:
    ckpt_path = A_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f'  SKIP {ckpt_name}: file not found')
        continue
    print(f'  Loading {ckpt_name}...', end=' ', flush=True)
    net = load_model(ckpt_path)
    d1 = evaluate_dump(net, POS_OCR_DUMP, GT1, seg_bounds=(SEG_B_START, SEG_B_END))
    d2 = evaluate_dump(net, POS_OCR_DUMP_2, GT2)
    scan_A[ckpt_name] = {'pos_ocr_dump': d1, 'static_control': d2}
    s = d1['seg_B']
    t = d2['full']
    print(f'seg_B: pp_char={s["pp_char"]:.1f}% len_err={s["len_err_rate"]:.1f}% '
          f'static: pp_exact={t["pp_exact"]:.1f}% prov_bias={t["prov_confusion"]}')
    del net; torch.cuda.empty_cache()

out_A = ROOT / 'experiments/mix_source_audit_20260510/checkpoint_scan_A_20260510.json'
json.dump(scan_A, open(out_A, 'w'), ensure_ascii=False, indent=2)
print(f'\nSaved A scan: {out_A}')

# ─── Scan B stage1 ───
B_DIR = ROOT / 'experiments/green_backbone_mix_branchB_stage1_backbone_mix_20260510'
B_CKPTS = [
    'best_LPRNet_model.pth', 'Final_LPRNet_model.pth', 'last_LPRNet_model.pth',
    'LPRNet__iteration_2000.pth', 'LPRNet__iteration_4000.pth',
]
print('\n' + '='*70)
print('SCANNING BRANCH B stage1')
print('='*70)
scan_B = {}
for ckpt_name in B_CKPTS:
    ckpt_path = B_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f'  SKIP {ckpt_name}: file not found')
        continue
    print(f'  Loading {ckpt_name}...', end=' ', flush=True)
    net = load_model(ckpt_path)
    d1 = evaluate_dump(net, POS_OCR_DUMP, GT1, seg_bounds=(SEG_B_START, SEG_B_END))
    d2 = evaluate_dump(net, POS_OCR_DUMP_2, GT2)
    scan_B[ckpt_name] = {'pos_ocr_dump': d1, 'static_control': d2}
    s = d1['seg_B']
    t = d2['full']
    print(f'seg_B: pp_char={s["pp_char"]:.1f}% len_err={s["len_err_rate"]:.1f}% '
          f'static: pp_exact={t["pp_exact"]:.1f}% prov_bias={t["prov_confusion"]}')
    del net; torch.cuda.empty_cache()

out_B = ROOT / 'experiments/mix_source_audit_20260510/checkpoint_scan_B_stage1_20260510.json'
json.dump(scan_B, open(out_B, 'w'), ensure_ascii=False, indent=2)
print(f'\nSaved B stage1 scan: {out_B}')

# ─── Print summary tables ───
def print_table(scan_data, label):
    print(f'\n{"="*90}')
    print(f'{label}')
    print(f'{"="*90}')
    header = f'{"checkpoint":30s} {"seg_pp_char":>10s} {"seg_len_err":>10s} {"seg_pp_ex":>9s} {"st_pp_ex":>9s} {"st_pp_char":>9s} {"st_prov_bias":>20s}'
    print(header)
    print('-'*90)
    for ckpt in sorted(scan_data.keys(), key=lambda k: k if 'iteration' not in k else f'z{k}'):
        d = scan_data[ckpt]
        s = d['pos_ocr_dump']['seg_B']
        t = d['static_control']['full']
        bias_items = list(t['prov_confusion'].items())[:3]
        bias_str = ' '.join(f'{k}={v}' for k,v in bias_items) if bias_items else 'none'
        print(f'{ckpt:30s} {s["pp_char"]:>9.1f}% {s["len_err_rate"]:>9.1f}% {s["pp_exact"]:>8.1f}% '
              f'{t["pp_exact"]:>8.1f}% {t["pp_char"]:>8.1f}% {bias_str:>20s}')

print_table(scan_A, 'BRANCH A — BOARD SCAN')
print_table(scan_B, 'BRANCH B stage1 — BOARD SCAN')
print('\nBoard scan complete')
