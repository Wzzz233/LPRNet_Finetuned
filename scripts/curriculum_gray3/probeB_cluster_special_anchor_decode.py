#!/usr/bin/env python3
"""Probe B: zero-training common-anchor decoding on cluster_special_validation_v1.

No training. No GT oracle in decoding decisions. Strategies use only existing CTC logits,
greedy/family-aware decode outputs, and province top-k derived from logits.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT/'src', ROOT/'src/evaluation', ROOT/'src/training', ROOT/'src/utils']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import CHARS, read_ppm_p6_payload, ocr_preprocess_bgr888  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa: E402
from train_LPRNet import _select_family_logits_from_dict  # noqa: E402
from eval_lpr_detailed import decode_logits  # noqa: E402

DATE = '20260426'
MANIFEST = ROOT / 'manifests/cluster_special_validation_v1/cluster_special_validation_v1.csv'
OUT = ROOT / f'reports/cluster_special_validation_probeB_anchor_decode_{DATE}'
WIN = Path(f'/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/cluster_special_validation_probeB_anchor_decode_{DATE}')
REPORT = ROOT / 'reports/GREEN_CLUSTER_SPECIAL_VALIDATION_PROBEB_ANCHOR_DECODE_REPORT.md'
for d in [OUT, WIN]:
    d.mkdir(parents=True, exist_ok=True)

MODELS = {
    'A1D_iter2000': ROOT/'experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth',
    'B1A_final': ROOT/'experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservative/Final_LPRNet_model.pth',
    'E1_final': ROOT/'experiments/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth',
    'E2_final': ROOT/'experiments/curriculum_gray3_stageB_v1_B1A_E2_provanchor_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth',
}
PROVINCES = CHARS[:31]
BLANK = len(CHARS) - 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
VALID_LETTERS = set('ABCDEFGHJKLMNPQRSTUVWXYZ')  # no I/O
GREEN_ALNUM = set('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')


def read_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = sorted({k for r in rows for k in r.keys()}) if rows else ['empty']
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def load_model(path):
    state = torch.load(str(path), map_location=DEVICE)
    net, _ = build_lprnet_multihead_from_state_dict(state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    load_multihead_state_dict_compat(net, state, strict=False)
    return net.to(DEVICE).eval()


def preprocess(path):
    img = read_ppm_p6_payload(str(path))
    if img.shape[:2] != (24, 94):
        img = cv2.resize(img, (94, 24), interpolation=cv2.INTER_NEAREST)
    gray3 = ocr_preprocess_bgr888(img, 'gray3')
    x = (gray3.astype('float32') - 127.5) * 0.0078125
    return torch.from_numpy(np.transpose(x, (2, 0, 1))[None, ...]).to(DEVICE)


def greedy_decode(logits):
    labels = []
    prev = None
    for t in range(logits.shape[1]):
        c = int(np.argmax(logits[:, t]))
        if c != BLANK and c != prev:
            labels.append(c)
        prev = c
    return ''.join(CHARS[c] for c in labels)


def province_prob(logits):
    vec = torch.tensor(logits[:31, :4], dtype=torch.float32).mean(dim=1)
    prob = F.softmax(vec, dim=0).numpy()
    order = np.argsort(prob)[::-1]
    return prob, order, ';'.join(f'{PROVINCES[int(i)]}:{prob[int(i)]:.3f}' for i in order[:5])


def ctc_char_margins(logits):
    # Diagnostic only: average top1-top2 probability margin over time, lower means unstable.
    prob = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).numpy()
    margins = []
    blank_probs = []
    for t in range(prob.shape[1]):
        order = np.argsort(prob[:, t])[::-1]
        margins.append(float(prob[order[0], t] - prob[order[1], t]))
        blank_probs.append(float(prob[BLANK, t]))
    return float(np.mean(margins)), float(np.mean(blank_probs))


def rank_of(prob, ch):
    if ch not in PROVINCES:
        return 999, 0.0
    idx = PROVINCES.index(ch)
    order = np.argsort(prob)[::-1]
    return int(np.where(order == idx)[0][0]) + 1, float(prob[idx])


def green8_valid_score(s):
    # Heuristic decoder-side score; does not use GT.
    score = 0
    if len(s) == 8:
        score += 6
    else:
        score -= abs(len(s) - 8)
    if len(s) >= 1 and s[0] in PROVINCES:
        score += 3
    if len(s) >= 2 and s[1] in VALID_LETTERS:
        score += 2
    for ch in s[2:8]:
        if ch in GREEN_ALNUM:
            score += 1
        else:
            score -= 1
    if not s:
        score -= 8
    return score


def choose_length_slot(beam, greedy):
    # Choose candidate with better green8 structure score. Tie keeps beam.
    sb = green8_valid_score(beam)
    sg = green8_valid_score(greedy)
    return (greedy, 'greedy_better_structure') if sg > sb else (beam, 'keep_beam')


def province_gate(text, prob, threshold):
    top_idx = int(np.argmax(prob))
    top_char = PROVINCES[top_idx]
    top_prob = float(prob[top_idx])
    if top_prob < threshold:
        return text, f'prov_below_{threshold}'
    if not text:
        return top_char, 'prov_fill_empty'
    if text[0] == top_char:
        return text, 'prov_same'
    # Only replace if current first is absent/non-province or province evidence is very strong.
    if text[0] not in PROVINCES or top_prob >= threshold:
        return top_char + text[1:], 'prov_replace'
    return text, 'prov_noop'


def common_anchor_v1(beam, greedy, prob):
    # Conservative combined strategy: first choose structurally better CTC output, then high-conf province gate.
    chosen, reason1 = choose_length_slot(beam, greedy)
    chosen2, reason2 = province_gate(chosen, prob, 0.85)
    return chosen2, reason1 + '+' + reason2


def strategies(beam, greedy, prob):
    length_slot, lreason = choose_length_slot(beam, greedy)
    p55, p55r = province_gate(beam, prob, 0.55)
    p85, p85r = province_gate(beam, prob, 0.85)
    anchor, ar = common_anchor_v1(beam, greedy, prob)
    return {
        'baseline_beam': (beam, 'baseline'),
        'board_greedy': (greedy, 'board_greedy'),
        'length_slot_select': (length_slot, lreason),
        'province_gate_055': (p55, p55r),
        'province_gate_085': (p85, p85r),
        'common_anchor_v1': (anchor, ar),
    }


def safe_char(s, i):
    return s[i] if i < len(s) else ''


def classify_error(gt, pred):
    if pred == gt:
        return 'exact'
    if not pred:
        return 'empty'
    first_ok = bool(pred) and pred[0] == gt[0]
    rear_ok = (len(pred) >= 4 and len(gt) >= 4 and pred[3:8] == gt[3:8]) or (len(pred) > 1 and pred[1:] == gt[1:])
    if not first_ok and rear_ok:
        return 'first_only'
    if first_ok:
        if len(pred) != len(gt):
            return 'length_collapse'
        return 'rear_or_slot'
    if len(pred) != len(gt):
        return 'mixed_length'
    return 'mixed'


def metric_row(meta, model_name, strategy, pred, reason, prob, top5, margin_mean, blank_mean, beam, greedy):
    gt = meta['gt_text']
    rr, rp = rank_of(prob, gt[:1])
    return {
        **meta,
        'model': model_name,
        'strategy': strategy,
        'decode_reason': reason,
        'pred': pred,
        'baseline_beam': beam,
        'board_greedy': greedy,
        'exact': int(pred == gt),
        'first': int(bool(pred) and pred[0] == gt[0]),
        'edit': edit_distance(gt, pred),
        'error_type': classify_error(gt, pred),
        'province_rank': rr,
        'province_prob': rp,
        'province_top5': top5,
        'pos2_acc': int(safe_char(pred, 1) == safe_char(gt, 1)),
        'pos3_acc': int(safe_char(pred, 2) == safe_char(gt, 2)),
        'rear_4_8_acc': int((pred[3:8] if len(pred) >= 4 else '') == (gt[3:8] if len(gt) >= 4 else '')),
        'empty': int(pred == ''),
        'length_collapse': int(len(pred) != len(gt)),
        'ctc_margin_mean': margin_mean,
        'blank_prob_mean': blank_mean,
        'province_top1_ctc_first_mismatch': int(rr == 1 and (not pred or pred[0] != gt[0])),
        'ctc_first_ok_rear_bad': int(bool(pred) and pred[0] == gt[0] and pred != gt),
    }


def eval_model(model_name, path, rows):
    net = load_model(path)
    out = []
    with torch.no_grad():
        for meta in rows:
            x = preprocess(meta['local_ocrin_path'])
            raw = net(x)
            logits = _select_family_logits_from_dict(raw, sample_families=['green8']).detach().cpu().numpy()[0]
            beam_ids = decode_logits(logits[None, ...], 'family_aware_beam', 20, 12, sample_families=['green8'])[0]
            beam = ''.join(CHARS[int(c)] for c in beam_ids)
            greedy = greedy_decode(logits)
            prob, order, top5 = province_prob(logits)
            margin_mean, blank_mean = ctc_char_margins(logits)
            for name, (pred, reason) in strategies(beam, greedy, prob).items():
                out.append(metric_row(meta, model_name, name, pred, reason, prob, top5, margin_mean, blank_mean, beam, greedy))
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarize(rows):
    def div(a, b): return a / b if b else 0.0
    summary = {}
    for model in sorted(set(r['model'] for r in rows)):
        model_rows = [r for r in rows if r['model'] == model]
        summary[model] = {}
        for strategy in sorted(set(r['strategy'] for r in model_rows)):
            strat_rows = [r for r in model_rows if r['strategy'] == strategy]
            summary[model][strategy] = {}
            for bucket in ['ALL'] + sorted(set(r['cluster_id'] for r in strat_rows)):
                arr = strat_rows if bucket == 'ALL' else [r for r in strat_rows if r['cluster_id'] == bucket]
                n = len(arr)
                hi = [r for r in arr if r.get('app_occ_ratio') not in ('', None) and float(r['app_occ_ratio']) >= 0.87]
                summary[model][strategy][bucket] = {
                    'n': n,
                    'exact': div(sum(int(r['exact']) for r in arr), n),
                    'first': div(sum(int(r['first']) for r in arr), n),
                    'mean_edit': div(sum(int(r['edit']) for r in arr), n),
                    'pos2_acc': div(sum(int(r['pos2_acc']) for r in arr), n),
                    'pos3_acc': div(sum(int(r['pos3_acc']) for r in arr), n),
                    'rear_4_8_acc': div(sum(int(r['rear_4_8_acc']) for r in arr), n),
                    'empty_count': sum(int(r['empty']) for r in arr),
                    'length_collapse_count': sum(int(r['length_collapse']) for r in arr),
                    'province_top1_ctc_first_mismatch': sum(int(r['province_top1_ctc_first_mismatch']) for r in arr),
                    'ctc_first_ok_rear_bad': sum(int(r['ctc_first_ok_rear_bad']) for r in arr),
                    'error_types': dict(Counter(r['error_type'] for r in arr).most_common()),
                    'top_preds': dict(Counter(r['pred'] for r in arr).most_common(8)),
                    'occ_ge_087_n': len(hi),
                    'occ_ge_087_exact': div(sum(int(r['exact']) for r in hi), len(hi)),
                    'occ_ge_087_first': div(sum(int(r['first']) for r in hi), len(hi)),
                    'occ_ge_087_rear': div(sum(int(r['rear_4_8_acc']) for r in hi), len(hi)),
                }
    return summary


def compare_to_baseline(summary):
    decisions = {}
    for model, sm in summary.items():
        base = sm['baseline_beam']
        decisions[model] = {}
        for strat, buckets in sm.items():
            if strat == 'baseline_beam':
                continue
            improved = []
            regressed = []
            for bucket in ['cluster1', 'cluster2', 'cluster3', 'cluster3_tail']:
                if bucket not in buckets or bucket not in base:
                    continue
                b = base[bucket]; s = buckets[bucket]
                gain = (s['exact'] - b['exact']) + 0.25 * (s['first'] - b['first']) + 0.25 * (s['rear_4_8_acc'] - b['rear_4_8_acc'])
                if gain > 0.02:
                    improved.append(bucket)
                if gain < -0.02:
                    regressed.append(bucket)
            decisions[model][strat] = {
                'improved_buckets': improved,
                'regressed_buckets': regressed,
                'passes_probeB_gate': len(improved) >= 2 and len(regressed) == 0,
            }
    return decisions


def write_report(summary, decisions):
    lines = ['# GREEN_CLUSTER_SPECIAL_VALIDATION_PROBEB_ANCHOR_DECODE_REPORT', '', '日期：2026-04-26', '', '口径：Probe B，zero-training common-anchor decoding；不训练，不使用 GT oracle，所有策略只用现有 CTC logits / greedy / family-aware 输出 / province top-k。', '']
    for model in sorted(summary):
        lines.append(f'## {model}')
        lines.append('|strategy|ALL exact|ALL first|ALL rear|c1 exact/first/rear|c2 exact/first/rear|c3 exact/first/rear|tail exact/first/rear|empty|len_collapse|')
        lines.append('|---|---:|---:|---:|---|---|---|---|---:|---:|')
        for strat in sorted(summary[model]):
            sm = summary[model][strat]
            def tri(b):
                x = sm[b]
                return f"{x['exact']:.3f}/{x['first']:.3f}/{x['rear_4_8_acc']:.3f}"
            allb = sm['ALL']
            lines.append(f"|{strat}|{allb['exact']:.3f}|{allb['first']:.3f}|{allb['rear_4_8_acc']:.3f}|{tri('cluster1')}|{tri('cluster2')}|{tri('cluster3')}|{tri('cluster3_tail')}|{allb['empty_count']}|{allb['length_collapse_count']}|")
        lines.append('')
        lines.append('### Gate decisions')
        lines.append('```json')
        lines.append(json.dumps(decisions[model], ensure_ascii=False, indent=2))
        lines.append('```')
        lines.append('')
    lines += ['## 结论摘要', '', '- 若某策略没有 passes_probeB_gate=true，则说明仅靠 zero-training decoding anchor 不能同时修复至少两个 cluster 且不退化。', '- 这类失败意味着现有 logits 里可恢复信息不足，后续应转 Probe C：按 special validation profile 生成真实域 QA，而不是先做大训练。', '', '## 产物', '', f'- 输出目录：{OUT}', f'- Windows QA：{WIN}', f'- rows：{OUT/"all_rows.csv"}', f'- summary：{OUT/"summary.json"}', f'- decisions：{OUT/"probeB_decisions.json"}']
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    rows = read_csv(MANIFEST)
    all_rows = []
    for model, path in MODELS.items():
        if path.exists():
            all_rows.extend(eval_model(model, path, rows))
    write_csv(OUT / 'all_rows.csv', all_rows)
    summary = summarize(all_rows)
    decisions = compare_to_baseline(summary)
    (OUT / 'summary.json').write_text(json.dumps({'device': str(DEVICE), 'summary': summary}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'probeB_decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_report(summary, decisions)
    import shutil
    for p in OUT.iterdir():
        if p.is_file():
            shutil.copy2(p, WIN / p.name)
    shutil.copy2(REPORT, WIN / REPORT.name)
    print(json.dumps({'out_dir': str(OUT), 'windows_dir': str(WIN), 'decisions': decisions}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
