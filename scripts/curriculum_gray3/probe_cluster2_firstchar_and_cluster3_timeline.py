#!/usr/bin/env python3
"""Small probes for real board clusters.

1) cluster2 first-char evidence: compare ocrin-left vs full-crop-left and
   model province evidence from current Gray3 checkpoints + A4C full_crop+gray3.
2) cluster3 timeline: frame-sorted occ_ratio / pred / top-k / edit-distance table.

No training is launched here.
"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src/evaluation', ROOT / 'src/training', ROOT / 'src/utils']:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from load_data import CHARS, read_ppm_p6_payload, ocr_preprocess_bgr888  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa: E402
from train_LPRNet import _select_family_logits_from_dict  # noqa: E402
from eval_lpr_detailed import decode_logits  # noqa: E402
from train_tiny_province_net import TinyProvinceNet  # noqa: E402

DATE = '20260426'
OUT = ROOT / f'reports/cluster_probe_firstchar_timeline_{DATE}'
WIN = Path(f'/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/cluster_probe_firstchar_timeline_{DATE}')
REPORT = ROOT / 'reports/GREEN_CLUSTER2_FIRSTCHAR_AND_CLUSTER3_TIMELINE_PROBE_REPORT.md'
for d in [OUT, WIN]:
    d.mkdir(parents=True, exist_ok=True)

CLUSTER2_CSV = ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv'
CLUSTER3_CSV = ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv'
A4C = ROOT / 'experiments/firstchar_tiny_A4C_gray3_fullcrop_bal31_bs4_nw0_ep30_fix1/best.pt'
MODELS = {
    'A1D_iter2000': ROOT / 'experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth',
    'B1A_final': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservative/Final_LPRNet_model.pth',
    'E1_final': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth',
    'E2_final': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E2_provanchor_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth',
}
PROVINCES = CHARS[:31]
BLANK = len(CHARS) - 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

FONT_PATHS = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/unifont/unifont.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]
FONT_PATH = next((p for p in FONT_PATHS if Path(p).exists()), None)
FONT = ImageFont.truetype(FONT_PATH, 14) if FONT_PATH else ImageFont.load_default()
SMALL = ImageFont.truetype(FONT_PATH, 11) if FONT_PATH else ImageFont.load_default()


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()}) if rows else ['empty']
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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


def load_lpr(path):
    state = torch.load(str(path), map_location=DEVICE)
    net, cfg = build_lprnet_multihead_from_state_dict(state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(DEVICE).eval()
    return net


def load_a4c(path):
    state = torch.load(str(path), map_location=DEVICE)
    in_ch = int(state['features.0.weight'].shape[1])
    net = TinyProvinceNet(in_channels=in_ch)
    net.load_state_dict(state, strict=True)
    net.to(DEVICE).eval()
    return net, in_ch


def preprocess_ocrin_gray3(path):
    img = read_ppm_p6_payload(str(path))
    img = ocr_preprocess_bgr888(img, 'gray3')
    x = img.astype('float32')
    x = (x - 127.5) * 0.0078125
    return torch.from_numpy(np.transpose(x, (2, 0, 1))[None, ...]).to(DEVICE), img


def topk_from_province_logits(vec, k=8):
    prob = F.softmax(vec.float(), dim=0).detach().cpu().numpy()
    order = np.argsort(prob)[::-1]
    return [{'char': PROVINCES[int(i)], 'prob': float(prob[int(i)]), 'rank': r + 1} for r, i in enumerate(order[:k])], prob, order


def province_diag_from_lpr(net, ocrin_path):
    x, img = preprocess_ocrin_gray3(ocrin_path)
    with torch.no_grad():
        raw = net(x)
        logits = _select_family_logits_from_dict(raw, sample_families=['green8']).detach().cpu().numpy()[0]
    vec = torch.tensor(logits[:31, :4], dtype=torch.float32).mean(dim=1)
    topk, prob, order = topk_from_province_logits(vec, 8)
    beam_ids = decode_logits(logits[None, ...], 'family_aware_beam', 20, 12, sample_families=['green8'])[0]
    pred = ''.join(CHARS[int(c)] for c in beam_ids)
    return {'pred': pred, 'topk': topk, 'prob': prob, 'order': order, 'ocrin_gray3': img, 'logits': logits}


def a4c_input_from_crop(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    full = img.copy()
    rs = cv2.resize(img, (171, 64), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x = torch.from_numpy(gray3.astype('float32') / 255.0).permute(2, 0, 1)[None, ...].to(DEVICE)
    return full, gray3, x


def a4c_diag(net, in_ch, crop_path):
    full, gray3, x = a4c_input_from_crop(crop_path)
    if in_ch == 1:
        x = x[:, 0:1] * 0.1140 + x[:, 1:2] * 0.5870 + x[:, 2:3] * 0.2990
    with torch.no_grad():
        logits = net(x)[0]
    topk, prob, order = topk_from_province_logits(logits[:31], 8)
    return {'topk': topk, 'prob': prob, 'order': order, 'crop_bgr': full, 'a4c_gray3': gray3}


def rank_prob(prob, ch):
    if ch not in PROVINCES:
        return 999, 0.0
    idx = PROVINCES.index(ch)
    order = np.argsort(prob)[::-1]
    rank = int(np.where(order == idx)[0][0]) + 1
    return rank, float(prob[idx])


def left_metrics(img_bgr, kind):
    h, w = img_bgr.shape[:2]
    if kind == 'ocrin':
        # Approximate province area in 94x24 final input.
        x1, x2 = 0, max(1, int(round(w * 0.18)))
    else:
        # Full crop/A4C fixed input: use first character band heuristic.
        x1, x2 = 0, max(1, int(round(w * 0.18)))
    roi = img_bgr[:, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 40, 120)
    non_bg = np.mean((gray < 245) & (gray > 10))
    dark = np.mean(gray < 100)
    bright = np.mean(gray > 220)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    return {
        f'{kind}_left_x2': x2,
        f'{kind}_left_brightness': float(gray.mean()),
        f'{kind}_left_std': float(gray.std()),
        f'{kind}_left_edge_density': float((edge > 0).mean()),
        f'{kind}_left_gx_mean': float(np.mean(np.abs(gx))),
        f'{kind}_left_occupancy': float(non_bg),
        f'{kind}_left_dark_ratio': float(dark),
        f'{kind}_left_bright_ratio': float(bright),
    }


def pil_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def fit_pil_bgr(img, size):
    im = pil_bgr(img)
    w, h = im.size
    scale = min(size[0] / max(1, w), size[1] / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    rs = im.resize((nw, nh), Image.Resampling.BILINEAR)
    can = Image.new('RGB', size, (245, 245, 245))
    can.paste(rs, ((size[0] - nw) // 2, (size[1] - nh) // 2))
    return can


def make_cluster2_sheet(rows, path):
    cell_w, cell_h = 620, 210
    can = Image.new('RGB', (cell_w, cell_h * len(rows)), (255, 255, 255))
    d = ImageDraw.Draw(can)
    for i, r in enumerate(rows):
        y = i * cell_h
        d.rectangle([0, y, cell_w - 1, y + cell_h - 1], outline=(200, 200, 200))
        crop = cv2.imread(r['local_crop_path'], cv2.IMREAD_COLOR)
        ocr = read_ppm_p6_payload(r['local_ocrin_path'])
        ocr_gray = ocr_preprocess_bgr888(ocr, 'gray3')
        crop_left = crop[:, :max(1, int(crop.shape[1] * 0.18))]
        ocr_left = ocr_gray[:, :max(1, int(ocr_gray.shape[1] * 0.18))]
        can.paste(fit_pil_bgr(crop, (175, 74)), (8, y + 8))
        can.paste(fit_pil_bgr(crop_left, (90, 74)), (190, y + 8))
        can.paste(fit_pil_bgr(cv2.resize(ocr_gray, (376, 96), interpolation=cv2.INTER_NEAREST), (188, 48)), (288, y + 8))
        can.paste(fit_pil_bgr(cv2.resize(ocr_left, (160, 96), interpolation=cv2.INTER_NEAREST), (96, 48)), (484, y + 8))
        txt = f"sid={r['sample_id']} frame={r['frame_id']} gt={r['gt_text']} app={r['app_text']} occ={r.get('app_occ_ratio','')}"
        d.text((8, y + 88), txt, font=SMALL, fill=(0, 0, 0))
        d.text((8, y + 106), f"Gray3:{r['gray3_model']} pred={r['gray3_pred']} 京rank={r['gray3_jing_rank']} 皖-京={float(r['gray3_wan_minus_jing']):+.3f} top5={r['gray3_top5']}", font=SMALL, fill=(0, 0, 120))
        d.text((8, y + 124), f"A4C full_crop+gray3 pred={r['a4c_top1']} 京rank={r['a4c_jing_rank']} 皖-京={float(r['a4c_wan_minus_jing']):+.3f} top5={r['a4c_top5']}", font=SMALL, fill=(120, 0, 0))
        d.text((8, y + 142), f"ocrin edge={float(r['ocrin_left_edge_density']):.3f} bright={float(r['ocrin_left_brightness']):.1f} occ={float(r['ocrin_left_occupancy']):.3f}; crop edge={float(r['crop_left_edge_density']):.3f} bright={float(r['crop_left_brightness']):.1f} occ={float(r['crop_left_occupancy']):.3f}", font=SMALL, fill=(20, 90, 20))
    can.save(path, quality=92)


def do_cluster2(models, a4c_net, a4c_in_ch):
    rows = read_csv(CLUSTER2_CSV)
    chosen_model_name = 'E2_final' if 'E2_final' in models else next(iter(models))
    chosen_net = models[chosen_model_name]
    out_rows = []
    for r in rows:
        ocr_path = Path(r['local_ocrin_path'])
        crop_path = Path(r['local_crop_path'])
        g = province_diag_from_lpr(chosen_net, ocr_path)
        a = a4c_diag(a4c_net, a4c_in_ch, crop_path)
        jing_rank, jing_p = rank_prob(g['prob'], '京')
        wan_rank, wan_p = rank_prob(g['prob'], '皖')
        a_jing_rank, a_jing_p = rank_prob(a['prob'], '京')
        a_wan_rank, a_wan_p = rank_prob(a['prob'], '皖')
        crop_img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        rec = dict(r)
        rec.update({
            'gray3_model': chosen_model_name,
            'gray3_pred': g['pred'],
            'gray3_jing_rank': jing_rank,
            'gray3_jing_prob': jing_p,
            'gray3_wan_rank': wan_rank,
            'gray3_wan_prob': wan_p,
            'gray3_wan_minus_jing': wan_p - jing_p,
            'gray3_top5': ';'.join(f"{x['char']}:{x['prob']:.3f}" for x in g['topk'][:5]),
            'a4c_model': str(A4C),
            'a4c_top1': a['topk'][0]['char'],
            'a4c_jing_rank': a_jing_rank,
            'a4c_jing_prob': a_jing_p,
            'a4c_wan_rank': a_wan_rank,
            'a4c_wan_prob': a_wan_p,
            'a4c_wan_minus_jing': a_wan_p - a_jing_p,
            'a4c_top5': ';'.join(f"{x['char']}:{x['prob']:.3f}" for x in a['topk'][:5]),
        })
        rec.update(left_metrics(g['ocrin_gray3'], 'ocrin'))
        rec.update(left_metrics(crop_img, 'crop'))
        rec.update(left_metrics(a['a4c_gray3'], 'a4c'))
        out_rows.append(rec)
    write_csv(OUT / 'cluster2_firstchar_evidence.csv', out_rows)
    make_cluster2_sheet(out_rows, OUT / 'cluster2_firstchar_evidence_contact.jpg')
    return out_rows


def do_cluster3(models):
    rows = sorted(read_csv(CLUSTER3_CSV), key=lambda r: int(r.get('frame_id') or 0))
    out = []
    for model_name, net in models.items():
        for r in rows:
            diag = province_diag_from_lpr(net, Path(r['local_ocrin_path']))
            pred = diag['pred']
            gt = r['gt_text']
            su_rank, su_p = rank_prob(diag['prob'], '苏')
            out.append({
                'model': model_name,
                'sample_id': r.get('sample_id', ''),
                'frame_id': int(r.get('frame_id') or 0),
                'ts_us': r.get('ts_us', ''),
                'app_occ_ratio': float(r.get('app_occ_ratio') or 0.0),
                'app_text': r.get('app_text', ''),
                'gt_text': gt,
                'pred': pred,
                'first_correct': int(bool(pred) and pred[0] == gt[0]),
                'exact': int(pred == gt),
                'edit_distance': edit_distance(gt, pred),
                'contains_full_gt': int(gt in pred or pred == gt),
                'su_rank': su_rank,
                'su_prob': su_p,
                'top5': ';'.join(f"{x['char']}:{x['prob']:.3f}" for x in diag['topk'][:5]),
                'local_ocrin_path': r.get('local_ocrin_path', ''),
                'local_crop_path': r.get('local_crop_path', ''),
                'failure_type': r.get('failure_type', ''),
            })
    write_csv(OUT / 'cluster3_timeline_topk_occ.csv', out)
    # Boundary summary per model.
    summary = {}
    for model_name in sorted(set(x['model'] for x in out)):
        arr = [x for x in out if x['model'] == model_name]
        arr_sorted = sorted(arr, key=lambda x: x['frame_id'])
        exact_frames = [x for x in arr_sorted if x['exact']]
        first_bad = next((x for x in arr_sorted if not x['first_correct']), None)
        exact_bad_after_good = None
        seen_good = False
        for x in arr_sorted:
            if x['exact']:
                seen_good = True
            elif seen_good and exact_bad_after_good is None:
                exact_bad_after_good = x
        summary[model_name] = {
            'n': len(arr_sorted),
            'occ_min': min(float(x['app_occ_ratio']) for x in arr_sorted),
            'occ_max': max(float(x['app_occ_ratio']) for x in arr_sorted),
            'exact_count': sum(x['exact'] for x in arr_sorted),
            'first_correct_count': sum(x['first_correct'] for x in arr_sorted),
            'mean_edit': sum(x['edit_distance'] for x in arr_sorted) / max(1, len(arr_sorted)),
            'exact_occ_values': [float(x['app_occ_ratio']) for x in exact_frames],
            'last_exact_frame': exact_frames[-1]['frame_id'] if exact_frames else None,
            'last_exact_occ': exact_frames[-1]['app_occ_ratio'] if exact_frames else None,
            'first_first_bad_frame': first_bad['frame_id'] if first_bad else None,
            'first_first_bad_occ': first_bad['app_occ_ratio'] if first_bad else None,
            'first_after_exact_drop_frame': exact_bad_after_good['frame_id'] if exact_bad_after_good else None,
            'first_after_exact_drop_occ': exact_bad_after_good['app_occ_ratio'] if exact_bad_after_good else None,
            'pred_counter': dict(Counter(x['pred'] for x in arr_sorted).most_common(10)),
        }
    (OUT / 'cluster3_timeline_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return out, summary


def write_report(cluster2_rows, cluster3_summary):
    n = len(cluster2_rows)
    a4c_jing_top5 = sum(int(r['a4c_jing_rank']) <= 5 for r in cluster2_rows)
    a4c_jing_top1 = sum(int(r['a4c_jing_rank']) == 1 for r in cluster2_rows)
    gray_jing_top5 = sum(int(r['gray3_jing_rank']) <= 5 for r in cluster2_rows)
    gray_jing_top1 = sum(int(r['gray3_jing_rank']) == 1 for r in cluster2_rows)
    avg_gray_gap = sum(float(r['gray3_wan_minus_jing']) for r in cluster2_rows) / max(1, n)
    avg_a4c_gap = sum(float(r['a4c_wan_minus_jing']) for r in cluster2_rows) / max(1, n)
    lines = []
    lines += ['# GREEN_CLUSTER2_FIRSTCHAR_AND_CLUSTER3_TIMELINE_PROBE_REPORT', '']
    lines += ['日期：2026-04-26', '目的：只做两个小 probe，不启动训练；判断 cluster2 京首字证据是否仍在 full_crop 中，以及 cluster3 苏BF01111 崩塌边界更接近哪个 occ_ratio/时间区间。', '']
    lines += ['## Probe 1: cluster2 首字证据', '']
    lines += [f'- 样本：{n} 张 cluster2 ocrin/crop，GT=京AD06088。']
    lines += [f'- Gray3 主链模型：E2_final，输入为板端 ocrin_94x24 显式 gray3。京 top1/top5 = {gray_jing_top1}/{gray_jing_top5}；平均(皖-京)概率差 = {avg_gray_gap:+.4f}。']
    lines += [f'- A4C 独立首字模型：{A4C}，严格按 full_crop + resize 171x64 + gray3 + 1ch 训练口径输入。京 top1/top5 = {a4c_jing_top1}/{a4c_jing_top5}；平均(皖-京)概率差 = {avg_a4c_gap:+.4f}。']
    if a4c_jing_top5 > gray_jing_top5:
        lines += ['- 初步判读：full_crop A4C 比 ocrin Gray3 主链保留更多“京”证据，firstchar/fullcrop auxiliary 或 fusion 有继续价值。']
    else:
        lines += ['- 初步判读：full_crop A4C 没比 ocrin Gray3 主链更能看到“京”；优先怀疑板端 crop/几何/左侧证据本身已经不足，不能只押 OCR head。']
    lines += ['', '## Probe 2: cluster3 时间线 top-k / occ_ratio', '']
    lines += ['| model | n | exact | first | occ range | last exact frame/occ | first first-bad frame/occ | mean edit |']
    lines += ['|---|---:|---:|---:|---|---|---|---:|']
    for model, s in cluster3_summary.items():
        lines.append(f"| {model} | {s['n']} | {s['exact_count']} | {s['first_correct_count']} | {s['occ_max']:.4f}->{s['occ_min']:.4f} | {s['last_exact_frame']}/{s['last_exact_occ']} | {s['first_first_bad_frame']}/{s['first_first_bad_occ']} | {s['mean_edit']:.2f} |")
    lines += ['', '## 产物', '']
    lines += [f'- cluster2 明细：{OUT / "cluster2_firstchar_evidence.csv"}']
    lines += [f'- cluster2 视觉对比：{OUT / "cluster2_firstchar_evidence_contact.jpg"}']
    lines += [f'- cluster3 时间线：{OUT / "cluster3_timeline_topk_occ.csv"}']
    lines += [f'- cluster3 汇总：{OUT / "cluster3_timeline_summary.json"}']
    lines += [f'- Windows QA 目录：{WIN}']
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def copy_outputs():
    for p in OUT.iterdir():
        if p.is_file() and p.suffix.lower() in ['.csv', '.json', '.jpg', '.md']:
            import shutil
            shutil.copy2(p, WIN / p.name)
    import shutil
    shutil.copy2(REPORT, WIN / REPORT.name)


def main():
    missing = [str(p) for p in [CLUSTER2_CSV, CLUSTER3_CSV, A4C] if not p.exists()]
    missing += [str(p) for p in MODELS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))
    models = {name: load_lpr(path) for name, path in MODELS.items()}
    a4c_net, a4c_in = load_a4c(A4C)
    cluster2_rows = do_cluster2(models, a4c_net, a4c_in)
    cluster3_rows, cluster3_summary = do_cluster3(models)
    summary = {
        'out_dir': str(OUT),
        'windows_dir': str(WIN),
        'device': str(DEVICE),
        'cluster2_n': len(cluster2_rows),
        'cluster3_n_rows': len(cluster3_rows),
        'models': list(models.keys()),
        'a4c_model': str(A4C),
        'cluster3_summary': cluster3_summary,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_report(cluster2_rows, cluster3_summary)
    copy_outputs()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
