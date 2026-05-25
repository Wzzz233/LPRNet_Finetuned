#!/usr/bin/env python3
"""Re-check old anchors using EXACT original protocol:
   - cv2.imread (BGR)
   - forward_family_logits
   - hardcoded GT1='苏BF01111', GT2='京AD06088'
   - post-province pp_exact/pp_char
   - seg_B = frames 11-40 (indices 11:41)
   - static_control = full dump2
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

BRANCHES = {
    'replace_only': ROOT / 'experiments/a_ablation_replace_only_20260510',
    'mix_rebuild': ROOT / 'experiments/a_ablation_mix_rebuild_20260510',
    'real_only': ROOT / 'experiments/a_ablation_real_only_20260510',
}

CKPTS = [
    'best_LPRNet_model.pth', 'Final_LPRNet_model.pth', 'last_LPRNet_model.pth',
    'LPRNet__iteration_2000.pth', 'LPRNet__iteration_4000.pth', 'LPRNet__iteration_6000.pth',
]

OUT_DIR = ROOT / 'experiments/mix_source_audit_20260510'

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
    return {'full': full, 'seg_B': seg, 'predictions': preds[:50]}

# ── Scan all 3 anchors ──
for branch_name, branch_dir in BRANCHES.items():
    print(f'\n{"="*60}')
    print(f'RECHECK: {branch_name}')
    print(f'{"="*60}')
    branch_data = {}
    for ckpt_name in CKPTS:
        ckpt_path = branch_dir / ckpt_name
        if not ckpt_path.exists():
            print(f'  SKIP {ckpt_name}')
            continue
        print(f'  {ckpt_name}...', end=' ', flush=True)
        net = load_model(ckpt_path)
        d1 = evaluate_dump(net, POS_OCR_DUMP, GT1, seg_bounds=(SEG_B_START, SEG_B_END))
        d2 = evaluate_dump(net, POS_OCR_DUMP_2, GT2)
        branch_data[ckpt_name] = {'pos_ocr_dump': d1, 'static_control': d2}
        s = d1['seg_B']; t = d2['full']
        print(f'seg: pp_char={s["pp_char"]:.1f} len_err={s["len_err_rate"]:.1f} '
              f'static: pp_exact={t["pp_exact"]:.1f} bias={list(t["prov_confusion"].keys())[:3]}')
        del net; torch.cuda.empty_cache()
    
    # Save
    out_path = OUT_DIR / f'recheck_anchor_{branch_name}.json'
    json.dump(branch_data, open(out_path, 'w'), ensure_ascii=False, indent=2)
    print(f'Saved: {out_path}')

# ── Print comparison table ──
print(f'\n{"="*70}')
print('COMPARISON: OLD SCAN vs RECHECK')
print(f'{"="*70}')
for branch_name in BRANCHES:
    old_path = OUT_DIR / f'a_ablation_board_scan_{branch_name}.json'
    new_path = OUT_DIR / f'recheck_anchor_{branch_name}.json'
    if not old_path.exists() or not new_path.exists():
        continue
    old_data = json.load(open(old_path))
    new_data = json.load(open(new_path))
    
    print(f'\n--- {branch_name} ---')
    for ckpt_name in CKPTS:
        if ckpt_name not in old_data or ckpt_name not in new_data:
            continue
        o = old_data[ckpt_name]['pos_ocr_dump']['seg_B']
        n = new_data[ckpt_name]['pos_ocr_dump']['seg_B']
        os_ = old_data[ckpt_name]['static_control']['full']
        ns = new_data[ckpt_name]['static_control']['full']
        
        match_pp = abs(o['pp_char'] - n['pp_char']) < 0.1
        match_len = abs(o['len_err_rate'] - n['len_err_rate']) < 0.1
        match_st = abs(os_['pp_exact'] - ns['pp_exact']) < 0.1
        
        status = '✅' if (match_pp and match_len and match_st) else '❌'
        print(f'  {ckpt_name:30s} {status} '
              f'seg_B: old={o["pp_char"]:.1f}/{o["len_err_rate"]:.1f} vs new={n["pp_char"]:.1f}/{n["len_err_rate"]:.1f} '
              f'static: old={os_["pp_exact"]:.1f} vs new={ns["pp_exact"]:.1f}')
    
    # Compare board-optimal
    def pick_optimal(bd):
        cands = [(ckpt, d['pos_ocr_dump']['seg_B']['pp_char'],
                  d['pos_ocr_dump']['seg_B']['len_err_rate'],
                  d['static_control']['full']['pp_exact'])
                 for ckpt, d in bd.items()]
        cands.sort(key=lambda x: (-x[1], x[2]))
        return cands[0] if cands else None
    
    old_best = pick_optimal(old_data)
    new_best = pick_optimal(new_data)
    if old_best and new_best:
        same = old_best[0] == new_best[0]
        print(f'  OPTIMAL: old={old_best[0]} new={new_best[0]} {"✅" if same else "❌ DIFFERENT"}')
        if not same:
            # Print full leaderboard
            print(f'    Old leaderboard:')
            cands_old = [(ckpt, d['pos_ocr_dump']['seg_B']['pp_char']) for ckpt, d in old_data.items()]
            cands_old.sort(key=lambda x: -x[1])
            for c, p in cands_old:
                print(f'      {c:30s} pp_char={p:.1f}%')
            print(f'    New leaderboard:')
            cands_new = [(ckpt, d['pos_ocr_dump']['seg_B']['pp_char']) for ckpt, d in new_data.items()]
            cands_new.sort(key=lambda x: -x[1])
            for c, p in cands_new:
                print(f'      {c:30s} pp_char={p:.1f}%')
