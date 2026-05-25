#!/usr/bin/env python3
"""StageB1A-E1 province/pos0 fusion probe on old/new extreme proxies.

Zero-training probe: keep OCR logits fixed, replace/fuse only the first char
from auxiliary province/pos0 branches, and compare whether the accepted
moderate extreme failure is primarily a first-char anchor problem.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src/evaluation', ROOT / 'src/training', ROOT / 'src/utils']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import UnifiedManifestDataset, CHARS  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa: E402
from train_LPRNet import _select_family_logits_from_dict  # noqa: E402
from test_LPRNet import collate_fn  # noqa: E402
from eval_lpr_detailed import decode_logits  # noqa: E402
from firstchar_fusion import extract_province_logits, extract_pos0_logits  # noqa: E402

MODEL = ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth'
OLD_MAN = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv'
NEW_MAN = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_edgefit_extreme.csv'
OUT = ROOT / 'reports/stageB1A_E1_province_pos0_fusion_probe_20260426'
PROVINCES = CHARS[:31]
MODES = [
    ('base', None, 'none', 0.0),
    ('province_replace_all', 'province', 'replace_all', 0.0),
    ('province_conf_055', 'province', 'replace_if_confident', 0.55),
    ('province_conf_070', 'province', 'replace_if_confident', 0.70),
    ('pos0_replace_all', 'pos0', 'replace_all', 0.0),
    ('pos0_conf_055', 'pos0', 'replace_if_confident', 0.55),
    ('pos0_conf_070', 'pos0', 'replace_if_confident', 0.70),
]


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def load_model(path, device):
    state = torch.load(path, map_location=device)
    net, cfg = build_lprnet_multihead_from_state_dict(
        state,
        lpr_max_len=8,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0,
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device)
    net.eval()
    return net, cfg


def build_dataset(manifest):
    full = UnifiedManifestDataset(
        manifest_path=str(manifest),
        img_size=[94, 24],
        lpr_max_len=8,
        split_filter='test',
        ocr_channel_order='bgr',
        ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_min_occ_ratio=0.90,
        ocr_quad_pad_ratio=0.0,
    )
    idx = [
        i for i, row in enumerate(full.records)
        if (row.get('family') or '').strip() == 'green8'
        and row.get('img_path')
        and Path(row.get('img_path')).exists()
    ]
    return full, idx, DataLoader(Subset(full, idx), batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn)


def ids_to_text(ids):
    return ''.join(CHARS[int(c)] for c in ids)


def fuse(base_text, aux_char, aux_conf, mode, threshold):
    if mode == 'none' or aux_char is None:
        return base_text, False, 'base'
    if not base_text or not aux_char:
        return base_text, False, 'empty'
    if mode == 'replace_all':
        if base_text[0] == aux_char:
            return base_text, False, 'same'
        return aux_char + base_text[1:], True, 'replace_all'
    if mode == 'replace_if_confident':
        if aux_conf < threshold:
            return base_text, False, 'below_threshold'
        if base_text[0] == aux_char:
            return base_text, False, 'same'
        return aux_char + base_text[1:], True, 'replace_if_confident'
    raise ValueError(mode)


def err_type(gt, pred):
    if pred == gt:
        return 'exact'
    if not pred:
        return 'empty'
    if pred[0] != gt[0]:
        return 'first_char'
    if len(pred) != len(gt):
        return 'length_after_first_ok'
    diffs = [i for i, (a, b) in enumerate(zip(gt, pred)) if a != b]
    if diffs == [1]:
        return 'pos1_letter'
    if diffs == [2]:
        return 'pos2_family'
    if all(i >= 3 for i in diffs):
        return 'rear_only'
    return 'mixed'


def summarize(rows):
    n = len(rows)
    by_prov = defaultdict(lambda: {'n': 0, 'exact': 0, 'first': 0})
    by_tier = defaultdict(lambda: {'n': 0, 'exact': 0, 'first': 0})
    by_dir = defaultdict(lambda: {'n': 0, 'exact': 0, 'first': 0})
    for r in rows:
        p = r['gt'][:1]
        for bucket, key in [(by_prov, p), (by_tier, r.get('tier') or ''), (by_dir, r.get('direction') or '')]:
            bucket[key]['n'] += 1
            bucket[key]['exact'] += int(r['exact'])
            bucket[key]['first'] += int(r['first_ok'])
    def fmt_bucket(b):
        return {
            k: {
                'n': v['n'],
                'exact': safe_div(v['exact'], v['n']),
                'first': safe_div(v['first'], v['n']),
            } for k, v in sorted(b.items())
        }
    return {
        'n': n,
        'exact': safe_div(sum(r['exact'] for r in rows), n),
        'first': safe_div(sum(r['first_ok'] for r in rows), n),
        'pos2': safe_div(sum(r['pos2_ok'] for r in rows), n),
        'pos3plus': safe_div(sum(r['pos3plus_ok_sum'] for r in rows), sum(r['pos3plus_total'] for r in rows)),
        'changed': sum(r['changed'] for r in rows),
        'oracle_first_exact': safe_div(sum(r['oracle_first_exact'] for r in rows), n),
        'aux_first': safe_div(sum(r.get('aux_first_ok', 0) for r in rows), n),
        'reasons': dict(Counter(r['reason'] for r in rows)),
        'err_types': dict(Counter(r['err_type'] for r in rows)),
        'pred_first_top': dict(Counter((r['pred'][:1] or '<empty>') for r in rows).most_common(15)),
        'aux_first_top': dict(Counter((r.get('aux_char') or '<none>') for r in rows).most_common(15)),
        'by_province': fmt_bucket(by_prov),
        'by_tier': fmt_bucket(by_tier),
        'by_direction': fmt_bucket(by_dir),
    }


def top_info(prob_row):
    order = np.argsort(prob_row)[::-1]
    top_idx = int(order[0])
    top_char = PROVINCES[top_idx] if top_idx < len(PROVINCES) else ''
    conf = float(prob_row[top_idx])
    top5 = ';'.join(f'{PROVINCES[int(i)]}:{float(prob_row[int(i)]):.3f}' for i in order[:5] if int(i) < len(PROVINCES))
    return top_idx, top_char, conf, top5


def eval_manifest(name, manifest, net, device):
    full, idx, loader = build_dataset(manifest)
    out_rows = []
    rec_cursor = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            targets = []
            st = 0
            for le in lengths:
                targets.append(labels[st:st + le].numpy())
                st += le
            images = images.to(device)
            fams = list(families)
            raw = net(images)
            logits = _select_family_logits_from_dict(raw, sample_families=fams).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=fams)
            prov_logits = extract_province_logits(raw, fams)
            pos0_logits = extract_pos0_logits(raw, fams)
            prov_prob = F.softmax(prov_logits, dim=1).detach().cpu().numpy() if prov_logits is not None else None
            pos0_prob = F.softmax(pos0_logits, dim=1).detach().cpu().numpy() if pos0_logits is not None else None
            for j, (pred_ids, gt_ids) in enumerate(zip(decoded, targets)):
                row = full.records[idx[rec_cursor]]
                rec_cursor += 1
                gt = ids_to_text(gt_ids.tolist())
                base = ids_to_text(pred_ids)
                aux = {}
                if prov_prob is not None:
                    _, ch, cf, top5 = top_info(prov_prob[j])
                    aux['province'] = (ch, cf, top5)
                if pos0_prob is not None:
                    _, ch, cf, top5 = top_info(pos0_prob[j])
                    aux['pos0'] = (ch, cf, top5)
                for mode_name, source, fuse_mode, threshold in MODES:
                    aux_char = aux_conf = aux_top5 = None
                    if source is not None and source in aux:
                        aux_char, aux_conf, aux_top5 = aux[source]
                    pred, changed, reason = fuse(base, aux_char, aux_conf or 0.0, fuse_mode, threshold)
                    pos3_total = max(0, len(gt) - 2)
                    pos3_ok = sum(1 for pos in range(2, len(gt)) if len(pred) > pos and pred[pos] == gt[pos])
                    oracle = (gt[:1] + base[1:]) if gt else base
                    out_rows.append({
                        'eval': name,
                        'mode': mode_name,
                        'source': source or 'base',
                        'fusion_mode': fuse_mode,
                        'threshold': threshold,
                        'idx': rec_cursor - 1,
                        'gt': gt,
                        'base_pred': base,
                        'pred': pred,
                        'exact': int(pred == gt),
                        'first_ok': int(bool(pred) and bool(gt) and pred[0] == gt[0]),
                        'pos2_ok': int(len(gt) > 1 and len(pred) > 1 and pred[1] == gt[1]),
                        'pos3plus_ok_sum': pos3_ok,
                        'pos3plus_total': pos3_total,
                        'oracle_first_exact': int(oracle == gt),
                        'changed': int(changed),
                        'reason': reason,
                        'err_type': err_type(gt, pred),
                        'aux_char': aux_char or '',
                        'aux_conf': float(aux_conf or 0.0),
                        'aux_top5': aux_top5 or '',
                        'aux_first_ok': int(bool(aux_char) and bool(gt) and aux_char == gt[0]),
                        'province_top5': aux.get('province', ('', 0.0, ''))[2] if 'province' in aux else '',
                        'pos0_top5': aux.get('pos0', ('', 0.0, ''))[2] if 'pos0' in aux else '',
                        'tier': row.get('difficulty_tier', 'old'),
                        'direction': row.get('extreme_direction', 'old'),
                        'path': row.get('img_path', ''),
                    })
    return out_rows


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net, cfg = load_model(MODEL, device)
    all_rows = []
    summary = {'model': str(MODEL), 'model_cfg': cfg, 'proxies': {}}
    for name, man in [('old_proxy', OLD_MAN), ('new_proxy', NEW_MAN)]:
        rows = eval_manifest(name, man, net, device)
        all_rows.extend(rows)
        write_csv(OUT / f'{name}_fusion_rows.csv', rows)
        proxy_summary = {}
        for mode in [m[0] for m in MODES]:
            proxy_summary[mode] = summarize([r for r in rows if r['mode'] == mode])
        summary['proxies'][name] = proxy_summary
    write_csv(OUT / 'all_fusion_rows.csv', all_rows)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# GREEN StageB1A-E1 Province/Pos0 Fusion Probe', '', f'Model: `{MODEL}`', '', '## Compact results', '']
    for proxy, ps in summary['proxies'].items():
        lines.append(f'### {proxy}')
        lines.append('mode | exact | first | pos2 | pos3+ | aux_first | changed | oracle_first_exact')
        lines.append('--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:')
        for mode, st in ps.items():
            lines.append(f"{mode} | {st['exact']:.4f} | {st['first']:.4f} | {st['pos2']:.4f} | {st['pos3plus']:.4f} | {st['aux_first']:.4f} | {st['changed']} | {st['oracle_first_exact']:.4f}")
        lines.append('')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
