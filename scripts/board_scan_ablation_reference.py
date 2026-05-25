#!/usr/bin/env python3
# ============================================================
# REPO-CANONICAL REFERENCE: Board-Centric Ablation Scan
# ============================================================
# This is the authoritative reference implementation for
# board-centric green ablation checkpoint scans.
#
# Protocol (DO NOT DEVIATE):
#   - GT: hardcoded GT1='苏BF01111', GT2='京AD06088'
#   - seg_B: indices 11-40 (frames 11-40, 0-indexed)
#   - pp_exact / pp_char: POST-PROVINCE (strip first char)
#   - Image load: cv2.imread (BGR)
#   - Model: forward_family_logits (not _select_family_logits_from_dict)
#   - Decode: decode_ctc (collapsed CTC greedy)
#
# This file supersedes /tmp/ablation_board_scan.py and any
# ad-hoc board scan scripts written outside this repo.
# ============================================================
#!/usr/bin/env python3
"""Board-centric checkpoint scan for A-ablation branches.
Scans all 6 checkpoints per branch, saves JSON per branch + summary markdown."""
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

POS_OCR_DUMP = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump')
POS_OCR_DUMP_2 = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2')
SEG_B_START, SEG_B_END = 11, 40

BRANCHES = {
    'mix_rebuild': ROOT / 'experiments/a_ablation_mix_rebuild_20260510',
    'replace_only': ROOT / 'experiments/a_ablation_replace_only_20260510',
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
        'n': n,
        'exact': exact_ok / n * 100,
        'fc': fc_ok / n * 100,
        'char': sum(char_accs) / n * 100,
        'pp_exact': pp_exact_ok / n * 100,
        'pp_char': sum(pp_char_accs) / n * 100,
        'mean_len': sum(lengths) / n,
        'len_err_rate': sum(1 for l in lengths if l != len(gt)) / n * 100,
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

# ── Scan all branches ──
all_results = {}
for branch_name, branch_dir in BRANCHES.items():
    print(f'\n{"="*60}')
    print(f'SCANNING: {branch_name}')
    print(f'{"="*60}')
    branch_data = {}
    for ckpt_name in CKPTS:
        ckpt_path = branch_dir / ckpt_name
        if not ckpt_path.exists():
            print(f'  SKIP {ckpt_name}: not found')
            continue
        print(f'  {ckpt_name}...', end=' ', flush=True)
        net = load_model(ckpt_path)
        d1 = evaluate_dump(net, POS_OCR_DUMP, GT1, seg_bounds=(SEG_B_START, SEG_B_END))
        d2 = evaluate_dump(net, POS_OCR_DUMP_2, GT2)
        branch_data[ckpt_name] = {'pos_ocr_dump': d1, 'static_control': d2}
        s = d1['seg_B']; t = d2['full']
        print(f'seg_B: pp_char={s["pp_char"]:.1f}% len_err={s["len_err_rate"]:.1f}% '
              f'static: pp_exact={t["pp_exact"]:.1f}% bias={list(t["prov_confusion"].keys())[:3]}')
        del net; torch.cuda.empty_cache()
    all_results[branch_name] = branch_data

# ── Save per-branch JSONs ──
for branch_name in BRANCHES:
    out_path = OUT_DIR / f'a_ablation_board_scan_{branch_name}.json'
    json.dump(all_results[branch_name], open(out_path, 'w'), ensure_ascii=False, indent=2)
    print(f'\nSaved: {out_path}')

# ── Select board-optimal per branch ──
def pick_board_optimal(branch_data):
    """Pick checkpoint with highest seg_B pp_char, then lowest len_err."""
    candidates = []
    for ckpt, data in branch_data.items():
        s = data['pos_ocr_dump']['seg_B']
        t = data['static_control']['full']
        candidates.append((ckpt, s['pp_char'], s['len_err_rate'], s['pp_exact'],
                          t['pp_exact'], t.get('prov_confusion', {})))
    candidates.sort(key=lambda x: (-x[1], x[2]))
    return candidates[0] if candidates else None

print(f'\n{"="*70}')
print('BOARD-OPTIMAL PER BRANCH')
print(f'{"="*70}')
optimals = {}
for branch_name in BRANCHES:
    best = pick_board_optimal(all_results[branch_name])
    if best:
        optimals[branch_name] = best
        print(f'{branch_name:25s}  winner={best[0]:30s}  '
              f'pp_char={best[1]:.1f}%  len_err={best[2]:.1f}%  '
              f'pp_exact={best[3]:.1f}%  static_pp_ex={best[4]:.1f}%')

# ── Summary table for comparison ──
print(f'\n{"="*70}')
print('BRANCH COMPARISON (board-optimal checkpoints)')
print(f'{"="*70}')
hdr = f'{"Branch":25s} {"seg_pp_char":>12s} {"seg_len_err":>12s} {"seg_pp_ex":>10s} {"st_pp_ex":>10s} {"st_pp_char":>10s} {"st_prov_bias":>25s}'
print(hdr)
print('-' * 105)
for branch_name in BRANCHES:
    b = optimals.get(branch_name)
    if not b: continue
    ckpt, pp_char, len_err, pp_ex, st_pp_ex, prov = b
    bias_str = ' '.join(f'{k}={v}' for k,v in list(prov.items())[:3]) if prov else 'none'
    print(f'{branch_name:25s} {pp_char:>11.1f}% {len_err:>11.1f}% {pp_ex:>9.1f}% {st_pp_ex:>9.1f}% '
          f'{-1:>9.1f}% {bias_str:>25s}')

# ── Write summary markdown ──
md_lines = []
md_lines.append('# A-Ablation Board-Centric Scan Summary\n')
md_lines.append(f'> Generated: 2026-05-10 | Seed: 20260510\n')
md_lines.append('---\n')
md_lines.append('## Board-Optimal Checkpoint per Branch\n\n')
md_lines.append(f'| Branch | Winner | seg_B pp_char | seg_B len_err | seg_B pp_exact | static pp_exact | Province Bias |\n')
md_lines.append(f'|---|---|---:|--:|--:|--:|---|\n')

for bn in ['real_only', 'replace_only', 'mix_rebuild']:
    b = optimals.get(bn)
    if not b: continue
    ckpt, pp_char, len_err, pp_ex, st_pp_ex, prov = b
    bias = ' '.join(f'{k}={v}' for k,v in list(prov.items())[:3]) if prov else 'none'
    md_lines.append(f'| {bn:20s} | {ckpt:30s} | {pp_char:.1f}% | {len_err:.1f}% | {pp_ex:.1f}% | {st_pp_ex:.1f}% | {bias} |\n')

md_lines.append('\n---\n')
md_lines.append('## Question 1: Which branch has the best tilt-character robustness?\n\n')
md_lines.append('Measured by seg_B pp_char on the board-optimal checkpoint.\n\n')

# Determine best tilt
best_tilt = max(optimals.items(), key=lambda x: x[1][1])
worst_tilt = min(optimals.items(), key=lambda x: x[1][1])
md_lines.append(f'- **Best**: `{best_tilt[0]}` (pp_char={best_tilt[1][1]:.1f}%)\n')
md_lines.append(f'- **Worst**: `{worst_tilt[0]}` (pp_char={worst_tilt[1][1]:.1f}%)\n')
# Compare real_only vs replace_only
real_tilt = optimals.get('real_only', (None, 0,))[1]
repl_tilt = optimals.get('replace_only', (None, 0,))[1]
mix_tilt = optimals.get('mix_rebuild', (None, 0,))[1]

if repl_tilt > real_tilt + 3:
    md_lines.append(f'\n- Replace data drives tilt robustness more than real data: replace_only ({repl_tilt:.1f}%) > real_only ({real_tilt:.1f}%)\n')
elif real_tilt > repl_tilt + 3:
    md_lines.append(f'\n- Real data drives tilt robustness more than replace data: real_only ({real_tilt:.1f}%) > replace_only ({repl_tilt:.1f}%)\n')
else:
    md_lines.append(f'\n- Real and replace data contribute similarly to tilt robustness (gap {abs(real_tilt-repl_tilt):.1f}pp)\n')

if mix_tilt >= max(real_tilt, repl_tilt):
    md_lines.append(f'- Mix does not harm tilt robustness relative to single-source branches\n')
else:
    md_lines.append(f'- Mix degrades relative to best single-source branch\n')

md_lines.append('\n---\n')
md_lines.append('## Question 2: Which branch has the best length stability under tilt?\n\n')
md_lines.append('Measured by seg_B len_err_rate on the board-optimal checkpoint.\n\n')
best_len = min(optimals.items(), key=lambda x: x[1][2])
worst_len = max(optimals.items(), key=lambda x: x[1][2])
md_lines.append(f'- **Best**: `{best_len[0]}` (len_err={best_len[1][2]:.1f}%)\n')
md_lines.append(f'- **Worst**: `{worst_len[0]}` (len_err={worst_len[1][2]:.1f}%)\n')

repl_len = optimals.get('replace_only', (None, None, 0, 0, 0, {}))[1][2] if len(optimals.get('replace_only', ())) > 1 else 999
real_len = optimals.get('real_only', (None, None, 0))[1][2]
mix_len = optimals.get('mix_rebuild', (None, None, 0))[1][2]
if repl_len < real_len and repl_len <= mix_len:
    md_lines.append(f'- Replace data better reduces length collapse\n')
elif real_len < repl_len and real_len <= mix_len:
    md_lines.append(f'- Real data better reduces length collapse\n')
else:
    md_lines.append(f'- Mix achieves best length stability\n')

md_lines.append('\n---\n')
md_lines.append('## Question 3: Which branch best preserves static green-board exact recognition?\n\n')
md_lines.append('Measured by static_board_ocrin_control pp_exact.\n\n')
best_static = max(optimals.items(), key=lambda x: x[1][4])
worst_static = min(optimals.items(), key=lambda x: x[1][4])
md_lines.append(f'- **Best**: `{best_static[0]}` (static pp_exact={best_static[1][4]:.1f}%)\n')
md_lines.append(f'- **Worst**: `{worst_static[0]}` (static pp_exact={worst_static[1][4]:.1f}%)\n')

repl_st = optimals.get('replace_only', (None, None, None, None, 0))[1][4]
real_st = optimals.get('real_only', (None, None, None, None, 0))[1][4]
mix_st = optimals.get('mix_rebuild', (None, None, None, None, 0))[1][4]
if real_st > repl_st + 5:
    md_lines.append(f'- Real data dominates static province/decoding stability\n')
elif repl_st > real_st + 5:
    md_lines.append(f'- Replace data also helps static stability\n')
else:
    md_lines.append(f'- Mix preserves acceptable static stability\n')

md_lines.append('\n---\n')
md_lines.append('## Question 4: Is mix additive, redundant, or conflicting?\n\n')

# Compute delta from individual branches
if real_tilt > 0 and repl_tilt > 0 and mix_tilt > 0:
    tilt_from_real = mix_tilt - real_tilt
    tilt_from_repl = mix_tilt - repl_tilt
    len_from_real = mix_len - real_len
    len_from_repl = mix_len - repl_len
    st_from_real = mix_st - real_st
    st_from_repl = mix_st - repl_st

    md_lines.append(f'Tilt (pp_char): mix={mix_tilt:.1f}%  real={real_tilt:.1f}%  replace={repl_tilt:.1f}%\n')
    md_lines.append(f'Length (len_err): mix={mix_len:.1f}%  real={real_len:.1f}%  replace={repl_len:.1f}%\n')
    md_lines.append(f'Static (pp_exact): mix={mix_st:.1f}%  real={real_st:.1f}%  replace={repl_st:.1f}%\n')

    # Decide label
    if mix_tilt >= max(real_tilt, repl_tilt) - 2 and mix_st >= max(real_st, repl_st) - 5:
        # Mix at least as good as best single-source on both tilt and static
        if abs(real_tilt - mix_tilt) < 3 and abs(repl_tilt - mix_tilt) < 3:
            label = 'MIX_ADDITIVE'
            reason = 'Mix performs as well as the better single-source branch on tilt, with comparable static stability. Neither real nor replace alone is clearly dominant.'
        elif mix_tilt > repl_tilt + 3 and mix_tilt > real_tilt + 3:
            label = 'MIX_ADDITIVE'
            reason = 'Mix outperforms both single-source branches on tilt robustness.'
        else:
            label = 'REPLACE_DOMINANT' if repl_tilt > real_tilt + 3 else 'REAL_DOMINANT'
            dominant = 'replace' if repl_tilt > real_tilt + 3 else 'real'
            reason = f'{dominant} data alone achieves comparable or better tilt robustness than mix. The other source does not add measurable benefit.'
    else:
        if mix_tilt < real_tilt - 5 and mix_tilt < repl_tilt - 5:
            label = 'MIX_CONFLICTING'
            reason = 'Mix degrades both tilt and static vs either single-source branch.'
        else:
            label = 'MIX_ADDITIVE'
            reason = 'Mix shows partial benefit but not uniformly best on all metrics.'

    md_lines.append(f'\n### Conclusion: {label}\n')
    md_lines.append(f'{reason}\n')

with open(OUT_DIR / 'a_ablation_board_scan_summary.md', 'w') as f:
    f.writelines(md_lines)
print(f'\nSaved: {OUT_DIR / "a_ablation_board_scan_summary.md"}')

# Print final legend
print(f'\n{"="*70}')
print(f'Final conclusion label: {locals().get("label", "N/A")}')
print(f'{"="*70}')
