#!/usr/bin/env python3
"""Diagnose StageB1A-E3 structural-anchor slot behavior on old/new extreme proxies.

This is a no-training replay. It checks whether the newly added slot head learned
usable per-position signal and whether that signal transfers to the CTC output.
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
from torch.utils.data import DataLoader, Subset

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src/evaluation', ROOT / 'src/training']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import UnifiedManifestDataset, CHARS, prepare_board_ocr_input_from_quad_bgr888  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa: E402
from train_LPRNet import _select_family_logits_from_dict  # noqa: E402
from test_LPRNet import collate_fn, greedy_decode_logits  # noqa: E402
from eval_lpr_detailed import decode_logits  # noqa: E402
from firstchar_fusion import extract_province_logits  # noqa: E402

DATE = '20260426'
EXP = ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_E3_structural_anchor_slot_lmh_ccpdboard_eval_original'
MODEL = EXP / 'Final_LPRNet_model.pth'
OLD_MAN = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv'
NEW_MAN = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_edgefit_extreme.csv'
OUT = ROOT / f'reports/stageB1A_E3_structural_anchor_slot_badcase_diagnosis_{DATE}'
WIN = Path(f'/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/stageB1A_E3_structural_anchor_slot_badcase_diagnosis_{DATE}')
REPORT = ROOT / 'reports/GREEN_STAGEB1A_E3_STRUCTURAL_ANCHOR_SLOT_BADCASE_DIAGNOSIS_REPORT.md'
OUT.mkdir(parents=True, exist_ok=True)
WIN.mkdir(parents=True, exist_ok=True)
FONT_PATH = next(p for p in [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/unifont/unifont.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
] if Path(p).exists())
FONT = ImageFont.truetype(FONT_PATH, 16)
SMALL = ImageFont.truetype(FONT_PATH, 12)
TINY = ImageFont.truetype(FONT_PATH, 10)
QKEYS = ['quad_1x', 'quad_1y', 'quad_2x', 'quad_2y', 'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y']
PROV = CHARS[:31]
BLANK = len(CHARS) - 1


def load_model(path, device):
    state = torch.load(path, map_location=device)
    net, cfg = build_lprnet_multihead_from_state_dict(
        state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device).eval()
    return net, cfg


def get_quad(row):
    vals = []
    for k in QKEYS:
        if not row.get(k):
            return None
        vals.append(float(row[k]))
    return np.array(vals, np.float32).reshape(4, 2)


def qmetrics(row):
    q = get_quad(row)
    if q is None:
        return {}
    top = q[1] - q[0]
    right = q[2] - q[1]
    bottom = q[2] - q[3]
    left = q[3] - q[0]

    def ang(v):
        return math.degrees(math.atan2(float(v[1]), float(v[0])))

    edges = [float(np.linalg.norm(top)), float(np.linalg.norm(right)), float(np.linalg.norm(bottom)), float(np.linalg.norm(left))]
    vals = [abs(ang(top)), abs(ang(bottom)), abs(abs(ang(left)) - 90), abs(abs(ang(right)) - 90)]
    return {
        'angle_score': max(vals),
        'ratio': max(edges[0], edges[2]) / max(1e-6, max(edges[1], edges[3])),
        'area': float(abs(cv2.contourArea(q.astype(np.float32)))),
        'min_edge': min(edges),
    }


def final94(row):
    img = cv2.imread(row['img_path'])
    if img is None:
        return np.zeros((24, 94, 3), np.uint8)
    q = get_quad(row)
    if q is not None:
        prep, _occ, _warped, _bbox, _quad = prepare_board_ocr_input_from_quad_bgr888(
            img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0
        )
        return prep
    return cv2.resize(img, (94, 24), interpolation=cv2.INTER_NEAREST)


def pil_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def decode_ids_to_text(ids):
    return ''.join(CHARS[int(c)] for c in ids if 0 <= int(c) < len(CHARS) and int(c) != BLANK)


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


def slot_pattern(s):
    out = []
    for i, ch in enumerate(s):
        if i == 0 and ch in PROV:
            out.append('P')
        elif ch.isdigit():
            out.append('D')
        elif ch.isascii() and ch.isalpha():
            out.append('A')
        elif ch in PROV:
            out.append('P')
        else:
            out.append('X')
    return ''.join(out)


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


def safe_char(s, i):
    return s[i] if i < len(s) else ''


def topk_text(prob, chars, k=5):
    order = np.argsort(prob)[::-1][:k]
    return ';'.join(f'{chars[int(i)]}:{float(prob[int(i)]):.3f}' for i in order)


def eval_manifest(name, manifest, net, device):
    ds = UnifiedManifestDataset(
        str(manifest),
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
    idx = [i for i, r in enumerate(ds.records) if (r.get('family') or '').strip() == 'green8' and r.get('img_path') and Path(r['img_path']).exists()]
    loader = DataLoader(Subset(ds, idx), batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn)
    rows = []
    rec_index = 0
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
            slot_logits = None
            if isinstance(raw, dict):
                slot_logits = raw.get('slot_green8')
            logits = _select_family_logits_from_dict(raw, sample_families=fams)
            arr = logits.detach().cpu().numpy()
            beam = decode_logits(arr, 'family_aware_beam', 20, 12, sample_families=fams)
            greedy = greedy_decode_logits(arr)
            prov_logits = extract_province_logits(raw, fams)
            prov_prob = F.softmax(prov_logits, dim=1).detach().cpu().numpy() if prov_logits is not None else None
            slot_prob = F.softmax(slot_logits, dim=2).detach().cpu().numpy() if slot_logits is not None else None
            for j, (bids, gids) in enumerate(zip(beam, targets)):
                row = ds.records[idx[rec_index]]
                rec_index += 1
                gt = ''.join(CHARS[int(c)] for c in gids.tolist())
                pred = ''.join(CHARS[int(c)] for c in bids)
                gpred = ''.join(CHARS[int(c)] for c in greedy[j])
                slot_pred = ''
                slot_top1 = [''] * 8
                slot_gt_hits = []
                slot_gt_probs = []
                slot_top5 = []
                if slot_prob is not None:
                    for pos in range(8):
                        p = slot_prob[j, pos]
                        top = int(np.argmax(p))
                        slot_top1[pos] = CHARS[top]
                        slot_top5.append(topk_text(p, CHARS, 5))
                        gt_idx = int(gids[pos]) if pos < len(gids) else BLANK
                        slot_gt_hits.append(int(top == gt_idx))
                        slot_gt_probs.append(float(p[gt_idx]))
                    slot_pred = ''.join(slot_top1)
                else:
                    slot_gt_hits = [0] * 8
                    slot_gt_probs = [0.0] * 8
                    slot_top5 = [''] * 8
                province_top5 = ''
                province_top1 = ''
                province_gt_rank = 999
                province_gt_prob = 0.0
                if prov_prob is not None:
                    pr = prov_prob[j]
                    order = np.argsort(pr)[::-1]
                    province_top1 = PROV[int(order[0])]
                    province_top5 = ';'.join(f'{PROV[int(i)]}:{float(pr[int(i)]):.3f}' for i in order[:5])
                    if gt[:1] in PROV:
                        gt_idx = PROV.index(gt[:1])
                        province_gt_rank = int(np.where(order == gt_idx)[0][0]) + 1
                        province_gt_prob = float(pr[gt_idx])
                rec = {
                    'eval': name,
                    'idx': rec_index - 1,
                    'gt': gt,
                    'pred': pred,
                    'greedy_pred': gpred,
                    'slot_pred': slot_pred,
                    'err_type': err_type(gt, pred),
                    'exact': int(pred == gt),
                    'first_ok': int(bool(pred) and pred[0] == gt[0]),
                    'slot_exact': int(slot_pred == gt),
                    'slot_first_ok': int(bool(slot_pred) and slot_pred[0] == gt[0]),
                    'slot_rear_4_8_ok': int(slot_pred[3:8] == gt[3:8]) if len(slot_pred) >= 8 and len(gt) >= 8 else 0,
                    'ctc_rear_4_8_ok': int(pred[3:8] == gt[3:8]) if len(pred) >= 8 and len(gt) >= 8 else 0,
                    'slot_mean_gt_prob': float(np.mean(slot_gt_probs)) if slot_gt_probs else 0.0,
                    'slot_hit_count': int(sum(slot_gt_hits)),
                    'edit': edit_distance(gt, pred),
                    'slot_edit': edit_distance(gt, slot_pred),
                    'pred_len': len(pred),
                    'greedy_len': len(gpred),
                    'slot_len': len(slot_pred),
                    'gt_pattern': slot_pattern(gt),
                    'pred_pattern': slot_pattern(pred),
                    'slot_pattern': slot_pattern(slot_pred),
                    'tier': row.get('difficulty_tier', 'old') or 'old',
                    'direction': row.get('extreme_direction', row.get('direction', 'old')) or 'old',
                    'path': row.get('img_path', ''),
                    'province_top1': province_top1,
                    'province_top5': province_top5,
                    'province_gt_rank': province_gt_rank,
                    'province_gt_prob': province_gt_prob,
                    **qmetrics(row),
                }
                for pos in range(8):
                    rec[f'ctc_pos{pos}_ok'] = int(safe_char(pred, pos) == safe_char(gt, pos))
                    rec[f'slot_pos{pos}_ok'] = int(slot_gt_hits[pos]) if pos < len(slot_gt_hits) else 0
                    rec[f'slot_pos{pos}_top1'] = slot_top1[pos]
                    rec[f'slot_pos{pos}_gt_prob'] = slot_gt_probs[pos] if pos < len(slot_gt_probs) else 0.0
                    rec[f'slot_pos{pos}_top5'] = slot_top5[pos] if pos < len(slot_top5) else ''
                rows.append(rec)
    return rows


def group(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r.get(key, '')].append(r)
    return d


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {}

    def avg(key):
        return sum(float(r.get(key, 0)) for r in rows) / n

    out = {
        'n': n,
        'ctc_exact': avg('exact'),
        'ctc_first': avg('first_ok'),
        'ctc_rear_4_8': avg('ctc_rear_4_8_ok'),
        'slot_exact': avg('slot_exact'),
        'slot_first': avg('slot_first_ok'),
        'slot_rear_4_8': avg('slot_rear_4_8_ok'),
        'slot_mean_gt_prob': avg('slot_mean_gt_prob'),
        'mean_edit': avg('edit'),
        'mean_slot_edit': avg('slot_edit'),
        'err_types': dict(Counter(r['err_type'] for r in rows).most_common()),
        'ctc_pred_first_top': dict(Counter((r['pred'][:1] or '<empty>') for r in rows).most_common(15)),
        'slot_pred_first_top': dict(Counter((r['slot_pred'][:1] or '<empty>') for r in rows).most_common(15)),
        'province_top1_top': dict(Counter((r['province_top1'] or '<none>') for r in rows).most_common(15)),
        'ctc_pred_len': dict(Counter(str(r['pred_len']) for r in rows).most_common()),
        'slot_pattern_top': dict(Counter(r['slot_pattern'] for r in rows).most_common(10)),
    }
    out['ctc_pos_acc'] = {f'pos{i}': avg(f'ctc_pos{i}_ok') for i in range(8)}
    out['slot_pos_acc'] = {f'pos{i}': avg(f'slot_pos{i}_ok') for i in range(8)}
    by_tier = {}
    for tier, arr in group(rows, 'tier').items():
        by_tier[tier] = {
            'n': len(arr),
            'ctc_exact': sum(r['exact'] for r in arr) / len(arr),
            'ctc_first': sum(r['first_ok'] for r in arr) / len(arr),
            'slot_exact': sum(r['slot_exact'] for r in arr) / len(arr),
            'slot_first': sum(r['slot_first_ok'] for r in arr) / len(arr),
            'slot_mean_gt_prob': sum(r['slot_mean_gt_prob'] for r in arr) / len(arr),
        }
    out['by_tier'] = by_tier
    return out


def write_csv(path, rows):
    fields = sorted({k for r in rows for k in r.keys()}) if rows else ['empty']
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def render_sheet(title, rows, out_path, limit=60):
    rows = rows[:limit]
    cols = 2
    cell_w, cell_h, title_h = 640, 210, 72
    canvas = Image.new('RGB', (cols * cell_w, title_h + math.ceil(len(rows) / cols) * cell_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), title, font=FONT, fill=(0, 0, 0))
    d.text((10, 34), 'final94 gray3 + CTC pred + SLOT pred + province/slot top-k', font=SMALL, fill=(60, 60, 60))
    for i, r in enumerate(rows):
        x = (i % cols) * cell_w
        y = title_h + (i // cols) * cell_h
        d.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(205, 205, 205))
        # Re-read manifest row via path is enough for plain fallback; for ccpd_board rows, the path image itself still provides visual context.
        img = cv2.imread(r['path'])
        if img is None:
            thumb = np.zeros((96, 376, 3), np.uint8)
        else:
            thumb = cv2.resize(img, (376, 96), interpolation=cv2.INTER_NEAREST)
        canvas.paste(pil_bgr(thumb), (x + 8, y + 8))
        color = (180, 0, 0) if not r['first_ok'] else (160, 90, 0)
        d.text((x + 392, y + 8), f"{i:02d} {r['tier']}/{r['direction']}", font=TINY, fill=(80, 80, 80))
        d.text((x + 392, y + 24), f"err={r['err_type']} edit={r['edit']} slot_edit={r['slot_edit']}", font=TINY, fill=color)
        d.text((x + 8, y + 112), f"GT   {r['gt']}", font=SMALL, fill=(0, 0, 0))
        d.text((x + 8, y + 132), f"CTC  {r['pred']}  greedy={r['greedy_pred']}", font=SMALL, fill=(120, 0, 0))
        d.text((x + 8, y + 152), f"SLOT {r['slot_pred']}  hits={r['slot_hit_count']}/8 gtprob={r['slot_mean_gt_prob']:.3f}", font=SMALL, fill=(0, 80, 0))
        d.text((x + 8, y + 174), f"prov={r['province_top5'][:92]}", font=TINY, fill=(0, 0, 120))
        d.text((x + 8, y + 190), Path(r['path']).name[:95], font=TINY, fill=(80, 80, 80))
    canvas.save(out_path, quality=92)


def write_report(summary):
    lines = [
        '# GREEN_STAGEB1A_E3_STRUCTURAL_ANCHOR_SLOT_BADCASE_DIAGNOSIS_REPORT',
        '',
        '日期：2026-04-26',
        '',
        '口径：固定 E3 Final，不训练；对 old proxy 与 new accepted proxy 做逐样本 replay，比较 CTC 输出、province head 与新增 slot head。',
        '',
    ]
    for name in ['old_proxy', 'new_proxy']:
        s = summary[name]
        lines += [
            f'## {name}',
            '',
            f"- n={s['n']}",
            f"- CTC exact={s['ctc_exact']:.4f}, first={s['ctc_first']:.4f}, rear_4_8={s['ctc_rear_4_8']:.4f}",
            f"- SLOT exact={s['slot_exact']:.4f}, first={s['slot_first']:.4f}, rear_4_8={s['slot_rear_4_8']:.4f}, mean_gt_prob={s['slot_mean_gt_prob']:.4f}",
            f"- CTC pred first top={s['ctc_pred_first_top']}",
            f"- SLOT pred first top={s['slot_pred_first_top']}",
            f"- Province top1 top={s['province_top1_top']}",
            f"- CTC pos acc={s['ctc_pos_acc']}",
            f"- SLOT pos acc={s['slot_pos_acc']}",
            f"- tier summary={s['by_tier']}",
            '',
        ]
    lines += [
        '## 产物',
        '',
        f'- 输出目录：{OUT}',
        f'- Windows QA：{WIN}',
        f'- summary：{OUT / "summary.json"}',
        f'- old rows：{OUT / "old_proxy_predictions.csv"}',
        f'- new rows：{OUT / "new_proxy_predictions.csv"}',
    ]
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net, cfg = load_model(MODEL, device)
    all_rows = []
    for name, man in [('old_proxy', OLD_MAN), ('new_proxy', NEW_MAN)]:
        rows = eval_manifest(name, man, net, device)
        all_rows.extend(rows)
        write_csv(OUT / f'{name}_predictions.csv', rows)
        render_sheet(f'{name} first-char failures', [r for r in rows if r['err_type'] == 'first_char'], OUT / f'{name}_first_char_failures.jpg')
        render_sheet(f'{name} slot-vs-ctc disagreements', sorted(rows, key=lambda r: (r['slot_hit_count'], r['exact'])), OUT / f'{name}_slot_ctc_disagreements.jpg')
    summary = {
        'model': str(MODEL),
        'model_cfg': cfg,
        'old_proxy': summarize([r for r in all_rows if r['eval'] == 'old_proxy']),
        'new_proxy': summarize([r for r in all_rows if r['eval'] == 'new_proxy']),
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(summary)
    import shutil
    for p in OUT.iterdir():
        if p.is_file():
            shutil.copy2(p, WIN / p.name)
    shutil.copy2(REPORT, WIN / REPORT.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
