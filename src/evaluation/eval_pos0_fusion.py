#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / 'src'
UTILS_DIR = SRC_DIR / 'utils'
EVAL_DIR = SRC_DIR / 'evaluation'
for p in [str(ROOT), str(SRC_DIR), str(UTILS_DIR), str(EVAL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.load_data import CHARS, UnifiedManifestDataset, PROVINCE_COUNT  # noqa: E402
from lpr_pipeline_policy import BOARD_PARAM_EXPECTED  # noqa: E402
from LPRNet_multihead import build_lprnet_multihead  # noqa: E402


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    families = []
    for sample in batch:
        if len(sample) == 4:
            img, label, length, family = sample
        else:
            img, label, length = sample
            family = 'normal7'
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
        families.append(family)
    labels = np.asarray(labels).flatten().astype(int)
    return torch.stack(imgs, 0), torch.from_numpy(labels), lengths, families


def greedy_decode_logits(prebs):
    preb_labels = []
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = []
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = []
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    return preb_labels


GREEN_PROVINCES = set(CHARS[:PROVINCE_COUNT])
GREEN_ALPHA = set('ABCDEFGHJKLMNPQRSTUVWXYZ')
GREEN_ALNUM = set('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
NORMAL7_ALPHA = set('ABCDEFGHJKLMNPQRSTUVWXYZ')
NORMAL7_ALNUM = set('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')


def green_prefix_valid(text):
    length = len(text)
    if length == 0:
        return True
    if text[0] not in GREEN_PROVINCES:
        return False
    if length >= 2 and text[1] not in GREEN_ALPHA:
        return False
    if length > 8:
        return False
    if length >= 3:
        # 新能源绿牌第三位不再强制限定为 D/F；D/F 耗尽后可启用其他合法字母段。
        # family-aware beam 只限制第 3 位及后续为绿牌合法字母/数字，不再强制末位 D/F。
        if any(c not in GREEN_ALNUM for c in text[2:]):
            return False
    return True


def green_full_valid(text):
    if len(text) != 8:
        return False
    if not green_prefix_valid(text):
        return False
    return all(c in GREEN_ALNUM for c in text[2:])


def normal7_prefix_valid(text):
    length = len(text)
    if length == 0:
        return True
    if text[0] not in GREEN_PROVINCES:
        return False
    if length >= 2 and text[1] not in NORMAL7_ALPHA:
        return False
    if length > 7:
        return False
    if any(c not in NORMAL7_ALNUM for c in text[2:]):
        return False
    return True


def normal7_full_valid(text):
    return len(text) == 7 and normal7_prefix_valid(text)


def family_prefix_valid(family, text):
    if family == 'green8':
        return green_prefix_valid(text)
    if family == 'normal7':
        return normal7_prefix_valid(text)
    return True


def family_full_valid(family, text):
    if family == 'green8':
        return green_full_valid(text)
    if family == 'normal7':
        return normal7_full_valid(text)
    return True


def family_target_length(family):
    if family == 'green8':
        return 8
    if family == 'normal7':
        return 7
    return None


def constrained_ctc_beam_decode_single(logits_ct, family, beam_size=30, topk=15):
    if family not in {'normal7', 'green8'}:
        return None
    blank_idx = len(CHARS) - 1
    topk = max(2, int(topk))
    beam_size = max(2, int(beam_size))
    log_probs = logits_ct - np.logaddexp.reduce(logits_ct, axis=0, keepdims=True)
    beams = {'': (0.0, -1e18)}
    max_len = family_target_length(family)
    for t in range(log_probs.shape[1]):
        next_beams = {}
        col = log_probs[:, t]
        idxs = np.argpartition(col, -topk)[-topk:]
        idxs = idxs[np.argsort(col[idxs])[::-1]]
        if blank_idx not in idxs:
            idxs = np.append(idxs, blank_idx)
        for prefix, (pb, pnb) in beams.items():
            nb_pb, nb_pnb = next_beams.get(prefix, (-1e18, -1e18))
            nb_pb = np.logaddexp(nb_pb, np.logaddexp(pb, pnb) + col[blank_idx])
            next_beams[prefix] = (nb_pb, nb_pnb)
            for c in idxs:
                if c == blank_idx:
                    continue
                ch = CHARS[int(c)]
                new_prefix = prefix + ch
                if max_len is not None and len(new_prefix) > max_len:
                    continue
                if not family_prefix_valid(family, new_prefix):
                    continue
                npb, npnb = next_beams.get(new_prefix, (-1e18, -1e18))
                if prefix and prefix[-1] == ch:
                    score = pb + col[c]
                else:
                    score = np.logaddexp(pb, pnb) + col[c]
                npnb = np.logaddexp(npnb, score)
                next_beams[new_prefix] = (npb, npnb)
                if prefix and prefix[-1] == ch:
                    rpb, rpnb = next_beams.get(prefix, (-1e18, -1e18))
                    rpnb = np.logaddexp(rpnb, pnb + col[c])
                    next_beams[prefix] = (rpb, rpnb)
        items = sorted(next_beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]), reverse=True)
        beams = dict(items[:beam_size])
    best_score = None
    best_text = ''
    for prefix, (pb, pnb) in beams.items():
        if not family_full_valid(family, prefix):
            continue
        score = np.logaddexp(pb, pnb)
        if best_score is None or score > best_score:
            best_score = score
            best_text = prefix
    if best_score is not None:
        return [CHARS.index(c) for c in best_text]
    fallback = []
    for prefix, (pb, pnb) in beams.items():
        if family_prefix_valid(family, prefix):
            score = np.logaddexp(pb, pnb)
            fallback.append((score, prefix))
    if fallback:
        fallback.sort(reverse=True)
        return [CHARS.index(c) for c in fallback[0][1]]
    return []


def green_constrained_ctc_beam_decode_single(logits_ct, beam_size=30, topk=15):
    blank_idx = len(CHARS) - 1
    topk = max(2, int(topk))
    beam_size = max(2, int(beam_size))
    log_probs = logits_ct - np.logaddexp.reduce(logits_ct, axis=0, keepdims=True)
    beams = {'': (0.0, -1e18)}
    for t in range(log_probs.shape[1]):
        next_beams = {}
        col = log_probs[:, t]
        idxs = np.argpartition(col, -topk)[-topk:]
        idxs = idxs[np.argsort(col[idxs])[::-1]]
        if blank_idx not in idxs:
            idxs = np.append(idxs, blank_idx)
        for prefix, (pb, pnb) in beams.items():
            nb_pb, nb_pnb = next_beams.get(prefix, (-1e18, -1e18))
            nb_pb = np.logaddexp(nb_pb, np.logaddexp(pb, pnb) + col[blank_idx])
            next_beams[prefix] = (nb_pb, nb_pnb)
            for c in idxs:
                if c == blank_idx:
                    continue
                ch = CHARS[int(c)]
                new_prefix = prefix + ch
                if not green_prefix_valid(new_prefix):
                    continue
                npb, npnb = next_beams.get(new_prefix, (-1e18, -1e18))
                if prefix and prefix[-1] == ch:
                    score = pb + col[c]
                else:
                    score = np.logaddexp(pb, pnb) + col[c]
                npnb = np.logaddexp(npnb, score)
                next_beams[new_prefix] = (npb, npnb)
                if prefix and prefix[-1] == ch:
                    rpb, rpnb = next_beams.get(prefix, (-1e18, -1e18))
                    rpnb = np.logaddexp(rpnb, pnb + col[c])
                    next_beams[prefix] = (rpb, rpnb)
        items = sorted(next_beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]), reverse=True)
        beams = dict(items[:beam_size])
    best_score = None
    best_text = ''
    for prefix, (pb, pnb) in beams.items():
        if not green_full_valid(prefix):
            continue
        score = np.logaddexp(pb, pnb)
        if best_score is None or score > best_score:
            best_score = score
            best_text = prefix
    if best_score is not None:
        return [CHARS.index(c) for c in best_text]
    fallback = []
    for prefix, (pb, pnb) in beams.items():
        if green_prefix_valid(prefix) and len(prefix) <= 8:
            score = np.logaddexp(pb, pnb)
            fallback.append((score, prefix))
    if fallback:
        fallback.sort(reverse=True)
        return [CHARS.index(c) for c in fallback[0][1]]
    return []


def decode_logits(prebs, mode, beam_size, beam_topk, sample_families=None):
    if mode == 'greedy':
        return greedy_decode_logits(prebs)
    if mode == 'green_ctc_beam':
        return [green_constrained_ctc_beam_decode_single(prebs[i], beam_size=beam_size, topk=beam_topk) for i in range(prebs.shape[0])]
    if mode == 'family_aware_beam':
        decoded = []
        for i in range(prebs.shape[0]):
            family = sample_families[i] if sample_families is not None and i < len(sample_families) else None
            seq = constrained_ctc_beam_decode_single(prebs[i], family, beam_size=beam_size, topk=beam_topk)
            if seq is None:
                seq = greedy_decode_logits(prebs[i:i + 1])[0]
            decoded.append(seq)
        return decoded
    raise ValueError(f'unsupported decode mode: {mode}')


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    ap = argparse.ArgumentParser(description='Evaluate pos0-based first-character fusion on green8 plates.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--img_size', default=[94, 24], nargs=2, type=int)
    ap.add_argument('--lpr_max_len', default=8, type=int)
    ap.add_argument('--dropout_rate', default=0.0, type=float)
    ap.add_argument('--enhanced_green_head', default='expD', choices=['', 'expD', 'expE'])
    ap.add_argument('--pos0_head_cols', default=-1, type=int)
    ap.add_argument('--pos0_num_classes', default=-1, type=int)
    ap.add_argument('--decode_mode', default='family_aware_beam', choices=['greedy', 'green_ctc_beam', 'family_aware_beam'])
    ap.add_argument('--beam_size', default=20, type=int)
    ap.add_argument('--beam_topk', default=12, type=int)
    ap.add_argument('--test_batch_size', default=200, type=int)
    ap.add_argument('--num_workers', default=0, type=int)
    ap.add_argument('--cuda', default=True, type=str2bool)
    ap.add_argument('--max_samples', default=0, type=int)
    ap.add_argument('--seed', default=20260320, type=int)
    ap.add_argument('--deterministic', default=True, type=str2bool)
    ap.add_argument('--split_filter', default='test')
    ap.add_argument('--focus_family', default='green8')
    ap.add_argument('--target_provinces', default='苏,沪,粤,浙,晋,黑')
    ap.add_argument('--fusion_mode', default='replace_if_confident', choices=['replace_all', 'replace_if_confident', 'replace_if_not_cjk'])
    ap.add_argument('--pos0_conf_threshold', default=0.55, type=float)
    ap.add_argument('--ocr_channel_order', default='bgr', choices=['rgb', 'bgr'])
    ap.add_argument('--ocr_crop_mode', default='obb_warp', choices=['fixed', 'box', 'tight', 'box-pad', 'match', 'obb_warp'])
    ap.add_argument('--ocr_resize_mode', default='letterbox', choices=['stretch', 'letterbox'])
    ap.add_argument('--ocr_resize_kernel', default='nn', choices=['nn', 'bilinear'])
    ap.add_argument('--ocr_preproc', default='none', choices=['none', 'raw', 'gray', 'gray3', 'bin'])
    ap.add_argument('--ocr_min_occ_ratio', default=0.90, type=float)
    ap.add_argument('--ocr_quad_pad_ratio', default=0.0, type=float)
    return ap.parse_args()


def configure_runtime(seed, deterministic):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def enforce_board_alignment_args(args):
    mismatches = []
    for key, expected in BOARD_PARAM_EXPECTED.items():
        actual = getattr(args, key, None)
        if isinstance(expected, float):
            ok = abs(float(actual) - float(expected)) <= 1e-6
        else:
            ok = (actual == expected)
        if not ok:
            mismatches.append(f'{key}={actual} expected {expected}')
    if mismatches:
        raise RuntimeError('Board-aligned OCR params must stay fixed: ' + '; '.join(mismatches))


def detect_pos0_head(state_dict):
    for k, v in state_dict.items():
        if k == 'pos0_head.4.weight':
            return {'cols': 4, 'num_classes': int(v.shape[0])}
    return None


def build_dataset(args):
    dataset = UnifiedManifestDataset(
        manifest_path=args.manifest,
        img_size=args.img_size,
        lpr_max_len=args.lpr_max_len,
        split_filter=args.split_filter,
        ocr_channel_order=args.ocr_channel_order,
        ocr_crop_mode=args.ocr_crop_mode,
        ocr_resize_mode=args.ocr_resize_mode,
        ocr_resize_kernel=args.ocr_resize_kernel,
        ocr_preproc=args.ocr_preproc,
        ocr_min_occ_ratio=args.ocr_min_occ_ratio,
        ocr_quad_pad_ratio=args.ocr_quad_pad_ratio,
    )
    valid_indices = []
    records = getattr(dataset, 'records', [])
    for idx, row in enumerate(records):
        img_path = row.get('img_path') if isinstance(row, dict) else None
        if img_path and os.path.exists(img_path):
            valid_indices.append(idx)
    if len(valid_indices) != len(records):
        print(f'[Info] skip missing-image rows: keep {len(valid_indices)}/{len(records)}')
    dataset = Subset(dataset, valid_indices)
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.max_samples)))
    return dataset


def is_cjk(tok):
    if not tok:
        return False
    cp = ord(tok[0])
    return 0x4E00 <= cp <= 0x9FFF


def select_family_logits(raw_dict, families):
    selected = []
    for i, family in enumerate(families):
        key = family if family in raw_dict else 'normal7'
        selected.append(raw_dict[key][i:i + 1])
    return torch.cat(selected, dim=0)


def ids_to_text(ids):
    return ''.join(CHARS[int(c)] for c in ids)


def text_to_ids(text):
    out = []
    for ch in text:
        try:
            out.append(CHARS.index(ch))
        except ValueError:
            return None
    return out


def fuse_first_char(base_text, pos0_char, pos0_conf, mode, threshold):
    if not base_text or not pos0_char:
        return base_text, False, 'empty'
    if mode == 'replace_all':
        if base_text[0] == pos0_char:
            return base_text, False, 'same'
        return pos0_char + base_text[1:], True, 'replace_all'
    if mode == 'replace_if_not_cjk':
        if is_cjk(base_text[0]):
            return base_text, False, 'base_is_cjk'
        if base_text[0] == pos0_char:
            return base_text, False, 'same'
        return pos0_char + base_text[1:], True, 'replace_if_not_cjk'
    if pos0_conf < threshold:
        return base_text, False, 'below_threshold'
    if base_text[0] == pos0_char:
        return base_text, False, 'same'
    return pos0_char + base_text[1:], True, 'replace_if_confident'


def empty_bucket():
    return {
        'sample_count': 0,
        'exact_count': 0,
        'first_char_correct': 0,
        'gt_first_counter': Counter(),
        'pred_first_counter': Counter(),
        'gt_pred_counter': defaultdict(Counter),
    }


def update_metrics(bucket, gt_text, pred_text):
    bucket['sample_count'] += 1
    bucket['exact_count'] += int(gt_text == pred_text)
    bucket['first_char_correct'] += int(bool(gt_text) and bool(pred_text) and gt_text[0] == pred_text[0])
    gt_first = gt_text[0] if gt_text else ''
    pred_first = pred_text[0] if pred_text else ''
    bucket['gt_first_counter'][gt_first] += 1
    bucket['pred_first_counter'][pred_first] += 1
    if gt_first:
        bucket['gt_pred_counter'][gt_first][pred_first] += 1


def summarize_metrics(bucket, target_provinces):
    sample_count = int(bucket['sample_count'])
    summary = {
        'sample_count': sample_count,
        'exact_plate_acc': float(bucket['exact_count'] / sample_count) if sample_count else 0.0,
        'first_char_acc': float(bucket['first_char_correct'] / sample_count) if sample_count else 0.0,
        'pred_first_counter': dict(bucket['pred_first_counter'].most_common()),
    }
    target_rows = {}
    for prov in target_provinces:
        gt_total = int(bucket['gt_first_counter'].get(prov, 0))
        preds = bucket['gt_pred_counter'].get(prov, Counter())
        target_rows[prov] = {
            'sample_count': gt_total,
            'pred_true_count': int(preds.get(prov, 0)),
            'pred_anhui_count': int(preds.get('皖', 0)),
            'pred_true_minus_anhui': int(preds.get(prov, 0) - preds.get('皖', 0)),
            'pred_top5': dict(preds.most_common(5)),
        }
    summary['target_provinces'] = target_rows
    return summary


def main():
    args = parse_args()
    configure_runtime(args.seed, args.deterministic)
    enforce_board_alignment_args(args)

    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    state = torch.load(args.weights, map_location=device)
    pos0_info = detect_pos0_head(state)
    pos0_cols = pos0_info['cols'] if args.pos0_head_cols < 0 and pos0_info else max(args.pos0_head_cols, 0)
    pos0_num_classes = pos0_info['num_classes'] if args.pos0_num_classes < 0 and pos0_info else (args.pos0_num_classes if args.pos0_num_classes > 0 else 31)
    if pos0_cols <= 0:
        raise RuntimeError(f'Weights do not contain a usable pos0 head: {args.weights}')

    net = build_lprnet_multihead(
        lpr_max_len=args.lpr_max_len,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=args.dropout_rate,
        enhanced_green_head=args.enhanced_green_head,
        pos0_head_cols=pos0_cols,
        pos0_num_classes=pos0_num_classes,
    )
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print('[Warn] missing keys:', missing)
        print('[Warn] unexpected keys:', unexpected)
    net.to(device)
    net.eval()

    dataset = build_dataset(args)
    loader = DataLoader(dataset, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)
    target_provinces = [x.strip() for x in args.target_provinces.split(',') if x.strip()]

    baseline_bucket = empty_bucket()
    fused_bucket = empty_bucket()
    fusion_reasons = Counter()
    changed_cases = []
    seen = 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels, lengths, families in loader:
            start = 0
            targets = []
            for length in lengths:
                label = labels[start:start + length]
                targets.append(label.numpy().tolist())
                start += length
            images = Variable(images.to(device))
            raw = net(images)
            pos0_logits = raw.get('pos0')
            if pos0_logits is None:
                raise RuntimeError('Model forward did not return pos0 logits')
            ctc_logits = select_family_logits(raw, families)
            ctc_np = ctc_logits.cpu().numpy()
            pos0_prob = torch.softmax(pos0_logits, dim=1).cpu().numpy()
            decoded = decode_logits(ctc_np, args.decode_mode, args.beam_size, args.beam_topk, sample_families=list(families))
            for i, base_ids in enumerate(decoded):
                family = families[i]
                gt_text = ids_to_text(targets[i])
                seen += 1
                if family != args.focus_family:
                    continue
                base_text = ids_to_text(base_ids)
                pos0_idx = int(np.argmax(pos0_prob[i]))
                pos0_conf = float(pos0_prob[i][pos0_idx])
                pos0_char = CHARS[pos0_idx] if pos0_idx < len(CHARS) else ''
                fused_text, changed, reason = fuse_first_char(base_text, pos0_char, pos0_conf, args.fusion_mode, args.pos0_conf_threshold)
                if changed and text_to_ids(fused_text) is None:
                    fused_text = base_text
                    changed = False
                    reason = 'invalid_fused_text'
                fusion_reasons[reason] += 1
                update_metrics(baseline_bucket, gt_text, base_text)
                update_metrics(fused_bucket, gt_text, fused_text)
                if changed and len(changed_cases) < 50:
                    changed_cases.append({
                        'gt': gt_text,
                        'base_pred': base_text,
                        'fused_pred': fused_text,
                        'pos0_char': pos0_char,
                        'pos0_conf': round(pos0_conf, 4),
                    })

    elapsed = time.time() - start_time
    baseline_summary = summarize_metrics(baseline_bucket, target_provinces)
    fused_summary = summarize_metrics(fused_bucket, target_provinces)
    result = {
        'weights': str(args.weights),
        'manifest': str(args.manifest),
        'focus_family': args.focus_family,
        'decode_mode': args.decode_mode,
        'fusion_mode': args.fusion_mode,
        'pos0_conf_threshold': args.pos0_conf_threshold,
        'enhanced_green_head': args.enhanced_green_head,
        'pos0_head_cols': pos0_cols,
        'pos0_num_classes': pos0_num_classes,
        'target_provinces': target_provinces,
        'elapsed_sec': elapsed,
        'baseline': baseline_summary,
        'fused': fused_summary,
        'delta': {
            'exact_plate_acc': fused_summary['exact_plate_acc'] - baseline_summary['exact_plate_acc'],
            'first_char_acc': fused_summary['first_char_acc'] - baseline_summary['first_char_acc'],
        },
        'fusion_reasons': dict(fusion_reasons),
        'changed_cases': changed_cases,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
