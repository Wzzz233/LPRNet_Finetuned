#!/usr/bin/env python3
"""A/B/C for StageB accepted-moderate extreme domain.

A: learnability stratification on accepted moderate train/new proxy using E1/E2 models.
B: micro-overfit sanity check on a tiny accepted moderate subset.
C: write evidence-backed plan/report for geometry-aware next step under paradigm-3.
"""
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src/evaluation', ROOT / 'src/training', ROOT / 'src/utils']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import UnifiedManifestDataset, CHARS  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa: E402
from train_LPRNet import _select_family_logits_from_dict  # noqa: E402
from test_LPRNet import collate_fn  # noqa: E402
from eval_pos0_fusion import family_prefix_valid, family_full_valid, family_target_length  # noqa: E402

OUT = ROOT / 'reports/stageB1A_ABC_accepted_domain_20260426'
OUT.mkdir(parents=True, exist_ok=True)
WIN = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/stageB1A_ABC_accepted_domain_20260426')
WIN.mkdir(parents=True, exist_ok=True)

MAN_TRAIN = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv'
MAN_NEW_PROXY = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_edgefit_extreme.csv'
MAN_OLD_PROXY = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv'
MODEL_E1 = ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth'
MODEL_E2 = ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E2_provanchor_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PROVINCES = CHARS[:31]
BLANK_IDX = len(CHARS) - 1
SEED = 47
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def ids_to_text(ids):
    return ''.join(CHARS[int(i)] for i in ids)


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def row_bucket(gt, beam_top1, gt_in_topk, first_rank, ed):
    if beam_top1 == gt:
        return 'exact_top1'
    if gt_in_topk and ed <= 2:
        return 'near_miss_topk'
    if first_rank <= 3 and ed <= 4:
        return 'firstchar_recoverable'
    if len(beam_top1) == len(gt) and sum(1 for i in range(min(len(gt), len(beam_top1))) if i >= 2 and gt[i] == beam_top1[i]) >= max(2, len(gt) - 4):
        return 'rear_partial'
    return 'unlearnable_like'


def summarize_rows(rows):
    n = len(rows)
    tier = defaultdict(lambda: {'n': 0, 'exact': 0, 'topk': 0})
    bucket = Counter()
    first_rank_hist = Counter()
    for r in rows:
        t = r.get('tier') or 'unknown'
        tier[t]['n'] += 1
        tier[t]['exact'] += int(r['beam_top1'] == r['gt'])
        tier[t]['topk'] += int(r['gt_in_top10'])
        bucket[r['learnability_bucket']] += 1
        first_rank_hist[str(r['gt_first_rank'])] += 1
    return {
        'n': n,
        'top1_exact': safe_div(sum(int(r['beam_top1'] == r['gt']) for r in rows), n),
        'gt_in_top10': safe_div(sum(int(r['gt_in_top10']) for r in rows), n),
        'mean_edit_top1': safe_div(sum(r['edit_top1'] for r in rows), n),
        'mean_edit_best_top10': safe_div(sum(r['best_edit_top10'] for r in rows), n),
        'first_rank_le1': safe_div(sum(int(r['gt_first_rank'] <= 1) for r in rows), n),
        'first_rank_le3': safe_div(sum(int(r['gt_first_rank'] <= 3) for r in rows), n),
        'first_rank_le5': safe_div(sum(int(r['gt_first_rank'] <= 5) for r in rows), n),
        'bucket_counts': dict(bucket),
        'bucket_ratio': {k: safe_div(v, n) for k, v in bucket.items()},
        'tier_stats': {
            k: {
                'n': v['n'],
                'top1_exact': safe_div(v['exact'], v['n']),
                'gt_in_top10': safe_div(v['topk'], v['n']),
            }
            for k, v in sorted(tier.items())
        },
        'first_rank_hist': dict(first_rank_hist),
        'top1_firstchar_top': dict(Counter((r['beam_top1'][:1] or '<empty>') for r in rows).most_common(10)),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_model(path):
    state = torch.load(path, map_location=DEVICE)
    net, cfg = build_lprnet_multihead_from_state_dict(
        state,
        lpr_max_len=8,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0,
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(DEVICE)
    net.eval()
    return net, cfg


def constrained_ctc_beams(logits_ct, family, beam_size=10, topk=12):
    log_probs = logits_ct - np.logaddexp.reduce(logits_ct, axis=0, keepdims=True)
    beams = {'': (0.0, -1e18)}
    max_len = family_target_length(family)
    for t in range(log_probs.shape[1]):
        next_beams = {}
        col = log_probs[:, t]
        idxs = np.argpartition(col, -topk)[-topk:]
        idxs = idxs[np.argsort(col[idxs])[::-1]]
        if BLANK_IDX not in idxs:
            idxs = np.append(idxs, BLANK_IDX)
        for prefix, (pb, pnb) in beams.items():
            nb_pb, nb_pnb = next_beams.get(prefix, (-1e18, -1e18))
            nb_pb = np.logaddexp(nb_pb, np.logaddexp(pb, pnb) + col[BLANK_IDX])
            next_beams[prefix] = (nb_pb, nb_pnb)
            for c in idxs:
                if int(c) == BLANK_IDX:
                    continue
                ch = CHARS[int(c)]
                new_prefix = prefix + ch
                if max_len is not None and len(new_prefix) > max_len:
                    continue
                if not family_prefix_valid(family, new_prefix):
                    continue
                npb, npnb = next_beams.get(new_prefix, (-1e18, -1e18))
                if prefix and prefix[-1] == ch:
                    score = pb + col[int(c)]
                else:
                    score = np.logaddexp(pb, pnb) + col[int(c)]
                npnb = np.logaddexp(npnb, score)
                next_beams[new_prefix] = (npb, npnb)
                if prefix and prefix[-1] == ch:
                    rpb, rpnb = next_beams.get(prefix, (-1e18, -1e18))
                    rpnb = np.logaddexp(rpnb, pnb + col[int(c)])
                    next_beams[prefix] = (rpb, rpnb)
        items = sorted(next_beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]), reverse=True)
        beams = dict(items[:beam_size])
    ranked = []
    for prefix, (pb, pnb) in beams.items():
        if not family_prefix_valid(family, prefix):
            continue
        score = float(np.logaddexp(pb, pnb))
        ranked.append((score, prefix, bool(family_full_valid(family, prefix))))
    ranked.sort(key=lambda x: x[0], reverse=True)
    full = [r for r in ranked if r[2]]
    if full:
        ordered = full + [r for r in ranked if not r[2]]
    else:
        ordered = ranked
    uniq = []
    seen = set()
    for score, prefix, valid in ordered:
        if prefix in seen:
            continue
        seen.add(prefix)
        uniq.append({'score': score, 'text': prefix, 'valid_full': valid})
    return uniq[:beam_size]


def build_dataset(manifest, split_filter=None, source_contains=None):
    full = UnifiedManifestDataset(
        manifest_path=str(manifest),
        img_size=[94, 24],
        lpr_max_len=8,
        split_filter=split_filter,
        ocr_channel_order='bgr',
        ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_min_occ_ratio=0.90,
        ocr_quad_pad_ratio=0.0,
    )
    idx = []
    for i, row in enumerate(full.records):
        if (row.get('family') or '').strip() != 'green8':
            continue
        source = row.get('source') or ''
        if source_contains and source_contains not in source:
            continue
        path = row.get('img_path')
        if not path or not Path(path).exists():
            continue
        idx.append(i)
    return full, idx


def eval_manifest_with_model(eval_name, manifest, split_filter, source_contains, model_name, net):
    full, idx = build_dataset(manifest, split_filter=split_filter, source_contains=source_contains)
    loader = DataLoader(Subset(full, idx), batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn)
    rows = []
    rec_cursor = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            targets = []
            st = 0
            for le in lengths:
                targets.append(labels[st:st + le].numpy())
                st += le
            images = images.to(DEVICE)
            raw = net(images)
            logits = _select_family_logits_from_dict(raw, sample_families=list(families)).detach().cpu().numpy()
            if 'province_family_heads.green8.4.weight' in net.state_dict():
                prov_raw = raw.get('province_family_logits')
            else:
                prov_raw = None
            prov_prob = None
            if isinstance(prov_raw, dict) and 'green8' in prov_raw:
                prov_prob = F.softmax(prov_raw['green8'], dim=1).detach().cpu().numpy()
            elif torch.is_tensor(prov_raw):
                prov_prob = F.softmax(prov_raw, dim=1).detach().cpu().numpy()
            for j, gt_ids in enumerate(targets):
                row = full.records[idx[rec_cursor]]
                rec_cursor += 1
                gt = ids_to_text(gt_ids.tolist())
                beams = constrained_ctc_beams(logits[j], 'green8', beam_size=10, topk=12)
                beam_texts = [b['text'] for b in beams]
                beam_top1 = beam_texts[0] if beam_texts else ''
                eds = [edit_distance(gt, t) for t in beam_texts] or [len(gt)]
                best_edit = min(eds)
                gt_in_top10 = gt in beam_texts
                gt_first = gt[:1]
                first_rank = 999
                first_top5 = ''
                if prov_prob is not None:
                    order = np.argsort(prov_prob[j])[::-1]
                    first_top5 = ';'.join(f'{PROVINCES[int(i)]}:{float(prov_prob[j][int(i)]):.3f}' for i in order[:5] if int(i) < len(PROVINCES))
                    for rank, i in enumerate(order[:31], start=1):
                        if PROVINCES[int(i)] == gt_first:
                            first_rank = rank
                            break
                else:
                    for rank, ch in enumerate([t[:1] or '' for t in beam_texts], start=1):
                        if ch == gt_first:
                            first_rank = rank
                            break
                learn_bucket = row_bucket(gt, beam_top1, gt_in_top10, first_rank, edit_distance(gt, beam_top1))
                rows.append({
                    'eval_name': eval_name,
                    'model_name': model_name,
                    'gt': gt,
                    'beam_top1': beam_top1,
                    'beam_top10': ' || '.join(beam_texts),
                    'gt_in_top10': int(gt_in_top10),
                    'edit_top1': edit_distance(gt, beam_top1),
                    'best_edit_top10': best_edit,
                    'gt_first_rank': first_rank,
                    'learnability_bucket': learn_bucket,
                    'tier': row.get('difficulty_tier', ''),
                    'direction': row.get('extreme_direction', ''),
                    'source': row.get('source', ''),
                    'img_path': row.get('img_path', ''),
                    'province_top5': first_top5,
                })
    return rows


class FixedRecordDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return rec['image'], rec['label_ids'], len(rec['label_ids']), rec['family']


def gather_fixed_records(manifest, limit_per_tier=24):
    full, idx = build_dataset(manifest, split_filter='train', source_contains='green_edgefit_extreme_E1_moderate_lmh_ccpdboard')
    buckets = defaultdict(list)
    for i in idx:
        r = full.records[i]
        buckets[r.get('difficulty_tier', 'unknown')].append(i)
    chosen = []
    for tier in ['low', 'mid', 'high']:
        arr = buckets.get(tier, [])
        arr = sorted(arr)
        chosen.extend(arr[:limit_per_tier])
    fixed = []
    for i in chosen:
        sample = full[i]
        img, label_ids, _, family = sample
        rec = full.records[i]
        fixed.append({
            'image': img,
            'label_ids': label_ids,
            'family': family,
            'gt': rec.get('text', ''),
            'img_path': rec.get('img_path', ''),
            'tier': rec.get('difficulty_tier', ''),
        })
    return fixed


def split_fixed_records(records):
    by_tier = defaultdict(list)
    for r in records:
        by_tier[r['tier']].append(r)
    train, val = [], []
    for tier in ['low', 'mid', 'high']:
        arr = by_tier[tier]
        train.extend(arr[:16])
        val.extend(arr[16:24])
    return train, val


def build_probe_model(ref_model_path):
    state = torch.load(ref_model_path, map_location='cpu')
    net, cfg = build_lprnet_multihead_from_state_dict(
        state,
        lpr_max_len=8,
        phase=True,
        class_num=len(CHARS),
        dropout_rate=0,
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    for p in net.parameters():
        p.requires_grad = False
    for name, p in net.named_parameters():
        if name.startswith('containers.green8') or name.startswith('family_adapters.green8'):
            p.requires_grad = True
    net.to(DEVICE)
    return net, cfg


def labels_to_targets(lengths, labels):
    targets = []
    st = 0
    for le in lengths:
        targets.append(labels[st:st + le].tolist())
        st += le
    return targets


def batch_exact(logits_np, targets):
    exact = 0
    for arr, gt_ids in zip(logits_np, targets):
        beams = constrained_ctc_beams(arr, 'green8', beam_size=1, topk=12)
        pred = beams[0]['text'] if beams else ''
        if pred == ids_to_text(gt_ids):
            exact += 1
    return exact


def run_micro_overfit():
    records = gather_fixed_records(MAN_TRAIN, limit_per_tier=24)
    train_records, val_records = split_fixed_records(records)
    train_ds = FixedRecordDataset(train_records)
    val_ds = FixedRecordDataset(val_records)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)
    net, cfg = build_probe_model(MODEL_E1)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    criterion = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    history = []
    for epoch in range(1, 41):
        net.train()
        tr_loss = 0.0
        tr_n = 0
        tr_exact = 0
        for images, labels, lengths, families in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            raw = net(images)
            logits = _select_family_logits_from_dict(raw, sample_families=list(families))
            log_probs = logits.log_softmax(1).permute(2, 0, 1)
            input_lengths = torch.full((images.size(0),), logits.size(2), dtype=torch.long, device=DEVICE)
            target_lengths = torch.tensor(lengths, dtype=torch.long, device=DEVICE)
            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()) * images.size(0)
            tr_n += images.size(0)
            tr_exact += batch_exact(logits.detach().cpu().numpy(), labels_to_targets(lengths, labels.detach().cpu()))
        net.eval()
        with torch.no_grad():
            val_exact = 0
            val_n = 0
            for images, labels, lengths, families in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                raw = net(images)
                logits = _select_family_logits_from_dict(raw, sample_families=list(families))
                val_exact += batch_exact(logits.detach().cpu().numpy(), labels_to_targets(lengths, labels.detach().cpu()))
                val_n += images.size(0)
        history.append({
            'epoch': epoch,
            'train_loss': safe_div(tr_loss, tr_n),
            'train_exact': safe_div(tr_exact, tr_n),
            'val_exact': safe_div(val_exact, val_n),
        })
    result = {
        'config': {
            'train_n': len(train_records),
            'val_n': len(val_records),
            'epochs': 40,
            'lr': 1e-3,
            'trainable_prefixes': ['containers.green8', 'family_adapters.green8'],
        },
        'history': history,
        'final': history[-1],
        'best_train_exact': max(h['train_exact'] for h in history),
        'best_val_exact': max(h['val_exact'] for h in history),
        'sample_paths_train': [r['img_path'] for r in train_records[:10]],
        'sample_paths_val': [r['img_path'] for r in val_records[:10]],
        'tier_counts_train': dict(Counter(r['tier'] for r in train_records)),
        'tier_counts_val': dict(Counter(r['tier'] for r in val_records)),
    }
    return result


def write_report(a_summary, b_result):
    lines = []
    lines.append('# GREEN StageB1A Accepted Domain ABC Report')
    lines.append('')
    lines.append('日期：2026-04-26')
    lines.append('')
    lines.append('## A 可学性分层诊断')
    lines.append('')
    for key, val in a_summary.items():
        lines.append(f'### {key}')
        lines.append('```json')
        lines.append(json.dumps(val, ensure_ascii=False, indent=2))
        lines.append('```')
        lines.append('')
    lines.append('## B micro-overfit sanity check')
    lines.append('')
    lines.append('```json')
    lines.append(json.dumps(b_result, ensure_ascii=False, indent=2))
    lines.append('```')
    lines.append('')
    lines.append('## C 结论与范式三后续建议')
    lines.append('')
    learnable = a_summary['new_proxy_E1']['bucket_ratio'].get('near_miss_topk', 0.0) + a_summary['new_proxy_E1']['bucket_ratio'].get('firstchar_recoverable', 0.0)
    train_micro = b_result['best_train_exact']
    val_micro = b_result['best_val_exact']
    lines.append(f'- accepted moderate new proxy 在 E1 上的可恢复样本占比（near_miss_topk + firstchar_recoverable）= {learnable:.4f}。')
    lines.append(f'- micro-overfit best train exact = {train_micro:.4f}，best val exact = {val_micro:.4f}。')
    lines.append('- 若 tiny set 都难以记住，说明仅靠当前 final94→CTC 主链不足，范式三后续不能继续只调采样/省份辅助。')
    lines.append('- 下一步最小可执行方案：保持板端一致 final94，不改主输入；新增 geometry-aware auxiliary（direction/angle/margin/occupancy bins）或极轻量 rectification-aware adapter probe。')
    lines.append('- 新 curriculum 不再按 low/mid/high 生成桶直接喂入，而应按 A 的模型可学性分桶：exact/near-miss/firstchar-recoverable/rear-partial/unlearnable_like。')
    lines.append('- 训练顺序建议：先用 exact+near-miss+firstchar-recoverable 做 B1，再小步加入 rear_partial，最后才碰 unlearnable_like。')
    report_path = ROOT / 'reports/GREEN_STAGEB1A_ABC_ACCEPTED_DOMAIN_REPORT.md'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def main():
    summary = {'artifacts': {}}
    models = {}
    for name, path in [('E1', MODEL_E1), ('E2', MODEL_E2)]:
        net, cfg = load_model(path)
        models[name] = net
        summary['artifacts'][f'model_{name}'] = str(path)
        summary['artifacts'][f'model_{name}_cfg'] = cfg

    a_rows = []
    for model_name, net in models.items():
        rows_train = eval_manifest_with_model('train_accepted_extreme', MAN_TRAIN, 'train', 'green_edgefit_extreme_E1_moderate_lmh_ccpdboard', model_name, net)
        rows_new = eval_manifest_with_model('new_proxy', MAN_NEW_PROXY, 'test', None, model_name, net)
        rows_old = eval_manifest_with_model('old_proxy', MAN_OLD_PROXY, 'test', None, model_name, net)
        for label, rows in [('train', rows_train), ('new_proxy', rows_new), ('old_proxy', rows_old)]:
            write_csv(OUT / f'A_{label}_{model_name}.csv', rows)
        a_rows.extend(rows_train + rows_new + rows_old)

    a_summary = {}
    for eval_name in ['train_accepted_extreme', 'new_proxy', 'old_proxy']:
        for model_name in ['E1', 'E2']:
            rows = [r for r in a_rows if r['eval_name'] == eval_name and r['model_name'] == model_name]
            a_summary[f'{eval_name}_{model_name}'] = summarize_rows(rows)
    write_csv(OUT / 'A_all_rows.csv', a_rows)
    (OUT / 'A_summary.json').write_text(json.dumps(a_summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    b_result = run_micro_overfit()
    (OUT / 'B_micro_overfit.json').write_text(json.dumps(b_result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    report_path = write_report(a_summary, b_result)
    final = {
        'A_summary': a_summary,
        'B_result': b_result,
        'report_path': str(report_path),
        'out_dir': str(OUT),
    }
    (OUT / 'summary.json').write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    import shutil
    for p in [OUT / 'A_summary.json', OUT / 'B_micro_overfit.json', OUT / 'summary.json', report_path]:
        if p.exists():
            shutil.copy2(p, WIN / p.name)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
