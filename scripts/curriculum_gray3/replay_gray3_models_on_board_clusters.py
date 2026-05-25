#!/usr/bin/env python3
"""Replay Gray3 representative checkpoints on board OCR dump clusters with explicit gray3 preprocessing.

The board ocrin_*.ppm images are already final 94x24 board inputs. For Gray3
checkpoints, this script converts them to grayscale triplicate before LPRNet
normalization, then evaluates family-aware beam and greedy CTC outputs.
"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src/evaluation', ROOT / 'src/training', ROOT / 'src/utils']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import CHARS, read_ppm_p6_payload, ocr_preprocess_bgr888  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa: E402
from train_LPRNet import _select_family_logits_from_dict  # noqa: E402
from eval_lpr_detailed import decode_logits  # noqa: E402

OUT = ROOT / 'reports/gray3_board_cluster_replay_20260426'
OUT.mkdir(parents=True, exist_ok=True)
REPORT = ROOT / 'reports/GREEN_GRAY3_BOARD_CLUSTER_REPLAY_REPORT.md'

CLUSTERS = {
    'cluster1': ROOT / 'tmp/green_board_native_cluster1_benchmark_manifest_20260413.csv',
    'cluster2': ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv',
    'cluster3': ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv',
    'cluster3_tail': ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster3_tail_collapse_wsl.csv',
}
MODELS = {
    'A1D_iter2000': ROOT / 'experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth',
    'B1A_iter2000': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservativeLPRNet__iteration_2000.pth',
    'B1A_final': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservative/Final_LPRNet_model.pth',
    'E1_final': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth',
    'E2_final': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E2_provanchor_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth',
}
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PROVINCES = CHARS[:31]
BLANK = len(CHARS) - 1


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def load_model(path):
    state = torch.load(path, map_location=DEVICE)
    net, cfg = build_lprnet_multihead_from_state_dict(state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(DEVICE)
    net.eval()
    return net, cfg


def read_cluster_rows(cluster, path):
    rows = []
    with path.open('r', encoding='utf-utf-8-sig' if False else 'utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if cluster == 'cluster1':
                img = r.get('img_path') or ''
                gt = r.get('text') or ''
                sample_id = str(i)
                frame_id = Path(img).stem
                app_text = ''
                occ = ''
            else:
                img = r.get('local_ocrin_path') or r.get('ocr_input_path') or r.get('img_path') or ''
                gt = r.get('gt_text') or r.get('text') or ''
                sample_id = r.get('sample_id') or str(i)
                frame_id = r.get('frame_id') or ''
                app_text = r.get('app_text') or ''
                occ = r.get('app_occ_ratio') or ''
            if not img or not gt or not Path(img).exists():
                continue
            rows.append({
                'cluster': cluster,
                'sample_id': sample_id,
                'frame_id': frame_id,
                'img_path': img,
                'gt': gt,
                'app_text': app_text,
                'app_occ_ratio': occ,
                'failure_type': r.get('failure_type', ''),
                'note': r.get('note', ''),
            })
    return rows


def preprocess_ocrin_gray3(path):
    img = read_ppm_p6_payload(path)
    if img.shape[0] != 24 or img.shape[1] != 94:
        raise RuntimeError(f'ocrin size mismatch {path}: {img.shape}')
    # Board dump PPM is treated as BGR consistently with the project board_dump manifest口径.
    img = ocr_preprocess_bgr888(img, 'gray3')
    x = img.astype('float32')
    x -= 127.5
    x *= 0.0078125
    return np.transpose(x, (2, 0, 1))


def greedy_decode_single(logits_ct):
    labels = []
    prev = None
    for t in range(logits_ct.shape[1]):
        c = int(np.argmax(logits_ct[:, t]))
        if c != BLANK and c != prev:
            labels.append(c)
        prev = c
    return ''.join(CHARS[c] for c in labels)


def province_topk_from_logits(logits_ct, k=5):
    # Same diagnostic convention used before: average first four timesteps over province logits.
    first_logits = torch.tensor(logits_ct[:31, :4], dtype=torch.float32).mean(dim=1)
    prob = F.softmax(first_logits, dim=0).cpu().numpy()
    order = np.argsort(prob)[::-1]
    return [{'char': PROVINCES[int(i)], 'prob': float(prob[int(i)])} for i in order[:k]], int(np.where([PROVINCES[int(i)] == '' for i in order[:31]])[0][0]) if False else order


def eval_rows(model_name, net, rows):
    out = []
    with torch.no_grad():
        for r in rows:
            x = preprocess_ocrin_gray3(r['img_path'])
            images = torch.from_numpy(x[None, ...]).to(DEVICE)
            raw = net(images)
            logits = _select_family_logits_from_dict(raw, sample_families=['green8']).detach().cpu().numpy()[0]
            fam = ['green8']
            beam_ids = decode_logits(logits[None, ...], 'family_aware_beam', 20, 12, sample_families=fam)[0]
            beam = ''.join(CHARS[int(c)] for c in beam_ids)
            greedy = greedy_decode_single(logits)
            top5, order = province_topk_from_logits(logits, 5)
            gt_first = r['gt'][:1]
            gt_first_rank = 999
            for rank, idx in enumerate(order[:31], start=1):
                if PROVINCES[int(idx)] == gt_first:
                    gt_first_rank = rank
                    break
            top5_s = ';'.join(f"{d['char']}:{d['prob']:.3f}" for d in top5)
            out.append({
                **r,
                'model': model_name,
                'pred_beam': beam,
                'pred_greedy': greedy,
                'beam_exact': int(beam == r['gt']),
                'beam_first': int(bool(beam) and beam[0] == r['gt'][0]),
                'greedy_exact': int(greedy == r['gt']),
                'greedy_first': int(bool(greedy) and greedy[0] == r['gt'][0]),
                'edit_beam': edit_distance(r['gt'], beam),
                'edit_greedy': edit_distance(r['gt'], greedy),
                'gt_first_rank': gt_first_rank,
                'province_top5': top5_s,
            })
    return out


def summarize(rows):
    n = len(rows)
    by_cluster = {}
    for cluster in sorted(set(r['cluster'] for r in rows)):
        arr = [r for r in rows if r['cluster'] == cluster]
        by_cluster[cluster] = summarize_flat(arr)
    return {'n': n, 'by_cluster': by_cluster}


def summarize_flat(arr):
    n = len(arr)
    return {
        'n': n,
        'beam_exact': safe_div(sum(r['beam_exact'] for r in arr), n),
        'beam_first': safe_div(sum(r['beam_first'] for r in arr), n),
        'greedy_exact': safe_div(sum(r['greedy_exact'] for r in arr), n),
        'greedy_first': safe_div(sum(r['greedy_first'] for r in arr), n),
        'mean_edit_beam': safe_div(sum(r['edit_beam'] for r in arr), n),
        'gt_first_rank_le1': safe_div(sum(r['gt_first_rank'] <= 1 for r in arr), n),
        'gt_first_rank_le5': safe_div(sum(r['gt_first_rank'] <= 5 for r in arr), n),
        'beam_top_predictions': dict(Counter(r['pred_beam'] for r in arr).most_common(8)),
        'greedy_top_predictions': dict(Counter(r['pred_greedy'] for r in arr).most_common(8)),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def write_report(summary, all_rows):
    lines = ['# GREEN Gray3 Board Cluster Replay Report', '', '日期：2026-04-26', '', '## 口径', '', '- 输入：三个 cluster 的板端 `ocrin_*.ppm`，直接作为最终 94x24 输入。', '- 预处理：显式执行 `gray3` 灰度三通道化，然后按 LPRNet 标准 `x=(x-127.5)*0.0078125` 归一化。', '- 解码：同时输出 family-aware beam 与板端 greedy CTC。', '- 注意：`app_text` 只作板端弱对照；GT 使用 cluster csv/manifest 中人工标注的 `gt_text/text`。', '', '## 汇总', '']
    for model, st in summary.items():
        lines.append(f'### {model}')
        lines.append('| cluster | n | beam exact | beam first | greedy exact | greedy first | mean edit | top beam preds |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---|')
        for cluster, c in st['by_cluster'].items():
            top = ', '.join(f'{k}×{v}' for k, v in c['beam_top_predictions'].items())
            lines.append(f"| {cluster} | {c['n']} | {c['beam_exact']:.4f} | {c['beam_first']:.4f} | {c['greedy_exact']:.4f} | {c['greedy_first']:.4f} | {c['mean_edit_beam']:.2f} | {top} |")
        lines.append('')
    # cluster3 timeline compact for best visibility
    lines.append('## cluster3 时间线观察')
    lines.append('')
    lines.append('cluster3 的 `app_occ_ratio` 已随轨迹从约 0.95 下降到 0.63；本报告的逐帧 CSV 可直接看每个模型在哪个 frame 开始从苏BF01111掉出。')
    lines.append('')
    lines.append('## 产物')
    lines.append('')
    lines.append(f'- 明细 CSV：`{OUT / "all_rows.csv"}`')
    lines.append(f'- 汇总 JSON：`{OUT / "summary.json"}`')
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    cluster_rows = []
    for name, path in CLUSTERS.items():
        if path.exists():
            cluster_rows.extend(read_cluster_rows(name, path))
    all_rows = []
    summary = {}
    for name, path in MODELS.items():
        if not path.exists():
            continue
        net, cfg = load_model(path)
        rows = eval_rows(name, net, cluster_rows)
        all_rows.extend(rows)
        summary[name] = summarize(rows)
        write_csv(OUT / f'{name}_rows.csv', rows)
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(OUT / 'all_rows.csv', all_rows)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_report(summary, all_rows)
    print(json.dumps({'summary': summary, 'out_dir': str(OUT), 'report': str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
