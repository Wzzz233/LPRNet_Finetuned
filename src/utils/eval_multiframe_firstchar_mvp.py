#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'training'), str(_SRC_DIR / 'evaluation'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import UnifiedManifestDataset, CHARS, CHARS_DICT, PROVINCE_COUNT
from eval_lpr_detailed import decode_logits
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from test_LPRNet import collate_fn, greedy_decode_logits
from train_LPRNet import forward_family_logits

MANIFEST_FIELDS = [
    'img_path', 'img_rel_path', 'dataset_name', 'split', 'text', 'plate_len', 'family', 'sub_type', 'source',
    'is_real', 'need_tilt_aug', 'preprocess_group', 'has_bbox', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'bbox_source', 'quad_source', 'ocr_channel_order', 'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_min_occ_ratio', 'ocr_quad_pad_ratio'
]

BLANK_IDX = len(CHARS) - 1
PROVINCE_CHARS = CHARS[:PROVINCE_COUNT]


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def text_to_ids(text):
    out = []
    for ch in text:
        idx = CHARS_DICT.get(ch)
        if idx is None:
            return None
        out.append(idx)
    return out


def build_temp_manifest(input_csvs, temp_manifest: Path):
    rows = []
    meta_rows = []
    for csv_path in input_csvs:
        with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                img_path = row.get('local_ocrin_path') or row.get('ocr_input_path') or row.get('img_path')
                gt = (row.get('gt_text') or '').strip()
                if not img_path or not gt:
                    continue
                img_path = str(Path(img_path))
                if not Path(img_path).exists():
                    continue
                track_id = row.get('track_id') or f'{csv_path.stem}:{gt}'
                rows.append({
                    'img_path': img_path,
                    'img_rel_path': img_path,
                    'dataset_name': 'dump_track_replay',
                    'split': 'test',
                    'text': gt,
                    'plate_len': len(gt),
                    'family': 'green8',
                    'sub_type': 'green',
                    'source': 'dump_track_replay',
                    'is_real': 1,
                    'need_tilt_aug': 0,
                    'preprocess_group': 'dump_track_replay',
                    'has_bbox': 0,
                    'has_quad': 0,
                    'can_parse_ccpd_geom': 0,
                    'can_perspective': 0,
                    'bbox_source': 'none',
                    'quad_source': 'none',
                    'ocr_channel_order': 'bgr',
                    'ocr_crop_mode': 'obb_warp',
                    'ocr_resize_mode': 'letterbox',
                    'ocr_resize_kernel': 'nn',
                    'ocr_preproc': 'none',
                    'ocr_min_occ_ratio': 0.9,
                    'ocr_quad_pad_ratio': 0.0,
                })
                meta = dict(row)
                meta['_src_csv'] = str(csv_path)
                meta['_src_stem'] = csv_path.stem
                meta['_row_index'] = i
                meta['_img_path'] = img_path
                meta['_track_id'] = track_id
                meta_rows.append(meta)
    temp_manifest.parent.mkdir(parents=True, exist_ok=True)
    with temp_manifest.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return meta_rows


def load_model(model_path: Path, device):
    state = torch.load(str(model_path), map_location=device)
    net, _cfg = build_lprnet_multihead_from_state_dict(
        state,
        lpr_max_len=8,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0,
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device)
    net.eval()
    return net


def topk_chars(probs, k=5):
    order = np.argsort(-probs)[:k]
    return [{'char': CHARS[int(i)], 'prob': float(probs[int(i)])} for i in order]


def ctc_candidate_scores(frame_logits, candidate_texts, ctc_loss):
    # frame_logits: [C, T]
    candidate_ids = []
    target_lengths = []
    valid_texts = []
    for text in candidate_texts:
        ids = text_to_ids(text)
        if ids is None or len(ids) == 0:
            continue
        candidate_ids.extend(ids)
        target_lengths.append(len(ids))
        valid_texts.append(text)
    if not valid_texts:
        return {}
    logits_t = torch.from_numpy(frame_logits).float().transpose(0, 1).contiguous()  # [T, C]
    log_probs = logits_t.log_softmax(dim=1).unsqueeze(1).repeat(1, len(valid_texts), 1)  # [T, N, C]
    targets = torch.tensor(candidate_ids, dtype=torch.long)
    input_lengths = torch.full((len(valid_texts),), logits_t.shape[0], dtype=torch.long)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)
    losses = ctc_loss(log_probs, targets, input_lengths, target_lengths_t)
    scores = -losses.detach().cpu().numpy()
    return {text: float(score) for text, score in zip(valid_texts, scores)}


def summarize_track(track_id, rows, args, ctc_loss):
    rows = sorted(rows, key=lambda r: (int(r.get('frame_id') or 0), int(r.get('ts_us') or 0), int(r.get('sample_id') or 0)))
    gt_text = rows[0].get('gt_text', '')
    baseline_pred_counter = Counter(r['family_aware_pred'] for r in rows)
    baseline_pred = baseline_pred_counter.most_common(1)[0][0] if baseline_pred_counter else ''
    suffix_counter = Counter()
    for r in rows:
        pred = r['family_aware_pred']
        if len(pred) >= 2:
            suffix_counter[pred[1:]] += 1
    suffix_consensus = suffix_counter.most_common(1)[0][0] if suffix_counter else ''
    suffix_consensus_count = suffix_counter[suffix_consensus] if suffix_consensus else 0
    suffix_consensus_share = safe_div(suffix_consensus_count, len(rows))

    eligible_rows = []
    for r in rows:
        pred = r['family_aware_pred']
        suffix_match = int(len(pred) >= 2 and pred[1:] == suffix_consensus)
        length_ok = int(len(pred) == len(gt_text)) if gt_text else int(len(pred) > 0)
        occ = float(r.get('app_occ_ratio') or 0.0)
        eligible = int(length_ok and suffix_match and occ >= args.min_occ_ratio)
        r['suffix_match'] = suffix_match
        r['length_ok'] = length_ok
        r['eligible'] = eligible
        if eligible:
            eligible_rows.append(r)

    track_score = {prov: 0.0 for prov in PROVINCE_CHARS}
    frame_winner_counter = Counter()
    frame_score_rows = []
    candidate_texts = [prov + suffix_consensus for prov in PROVINCE_CHARS] if suffix_consensus else []
    for r in eligible_rows:
        score_map = ctc_candidate_scores(r['logits'], candidate_texts, ctc_loss)
        if score_map:
            for prov in PROVINCE_CHARS:
                text = prov + suffix_consensus
                sc = float(score_map.get(text, float('-inf')))
                if math.isfinite(sc):
                    track_score[prov] += sc
            best_text, best_score = max(score_map.items(), key=lambda kv: kv[1])
            best_prov = best_text[0]
            frame_winner_counter[best_prov] += 1
            top_items = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:5]
            frame_score_rows.append({
                'frame_id': r.get('frame_id'),
                'sample_id': r.get('sample_id'),
                'best_province': best_prov,
                'best_text': best_text,
                'best_score': float(best_score),
                'top_candidates': [{'text': txt, 'score': float(sc)} for txt, sc in top_items],
            })
        else:
            frame_score_rows.append({
                'frame_id': r.get('frame_id'),
                'sample_id': r.get('sample_id'),
                'best_province': '',
                'best_text': '',
                'best_score': float('-inf'),
                'top_candidates': [],
            })

    sorted_track_scores = sorted(track_score.items(), key=lambda kv: kv[1], reverse=True)
    best_prov, best_score = sorted_track_scores[0] if sorted_track_scores else ('', float('-inf'))
    second_score = sorted_track_scores[1][1] if len(sorted_track_scores) > 1 else float('-inf')
    second_prov = sorted_track_scores[1][0] if len(sorted_track_scores) > 1 else ''
    score_margin = float(best_score - second_score) if math.isfinite(best_score) and math.isfinite(second_score) else 0.0
    winner_share = safe_div(frame_winner_counter[best_prov], len(eligible_rows)) if best_prov else 0.0

    gates = {
        'track_len_ok': len(rows) >= args.min_track_len,
        'suffix_consensus_ok': suffix_consensus_share >= args.min_suffix_share,
        'eligible_frames_ok': len(eligible_rows) > 0,
        'margin_ok': score_margin >= args.min_score_margin,
        'winner_share_ok': winner_share >= args.min_winner_share,
    }
    activated = gates['track_len_ok'] and gates['suffix_consensus_ok'] and gates['eligible_frames_ok']
    accepted = activated and gates['margin_ok'] and gates['winner_share_ok']

    if accepted and suffix_consensus:
        final_pred = best_prov + suffix_consensus
        decision = 'track_override'
    else:
        final_pred = baseline_pred
        decision = 'abstain_fallback'

    return {
        'track_id': track_id,
        'gt_text': gt_text,
        'track_len': len(rows),
        'baseline_pred_text': baseline_pred,
        'baseline_exact_match': int(baseline_pred == gt_text),
        'baseline_first_char_match': int(bool(gt_text) and bool(baseline_pred) and baseline_pred[0] == gt_text[0]),
        'suffix_consensus': suffix_consensus,
        'suffix_consensus_count': suffix_consensus_count,
        'suffix_consensus_share': suffix_consensus_share,
        'eligible_frame_count': len(eligible_rows),
        'activated': activated,
        'accepted': accepted,
        'decision': decision,
        'best_province': best_prov,
        'second_province': second_prov,
        'track_score_margin': score_margin,
        'winner_share': winner_share,
        'track_pred_text': final_pred,
        'track_exact_match': int(final_pred == gt_text),
        'track_first_char_match': int(bool(gt_text) and bool(final_pred) and final_pred[0] == gt_text[0]),
        'gates': gates,
        'track_scores_top5': [{'province': prov, 'score': float(score)} for prov, score in sorted_track_scores[:5]],
        'frame_winner_counts': dict(frame_winner_counter),
        'frame_score_rows': frame_score_rows,
        'frames': [
            {
                'sample_id': r.get('sample_id'),
                'frame_id': r.get('frame_id'),
                'ts_us': r.get('ts_us'),
                'family_aware_pred': r['family_aware_pred'],
                'board_greedy_pred': r['board_greedy_pred'],
                'app_text': r.get('app_text', ''),
                'app_conf': float(r.get('app_conf') or 0.0),
                'app_blank_top1': float(r.get('app_blank_top1') or 0.0),
                'app_occ_ratio': float(r.get('app_occ_ratio') or 0.0),
                'suffix_match': r['suffix_match'],
                'length_ok': r['length_ok'],
                'eligible': r['eligible'],
                'province_top5': r['province_top5'],
            }
            for r in rows
        ],
    }


def main():
    ap = argparse.ArgumentParser(description='Offline multiframe first-char aggregation MVP on board OCR dump tracks.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--input-csv', nargs='+', required=True, help='One or more track CSVs with gt_text/local_ocrin_path rows')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-csv', default='')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--decode-mode', default='family_aware_beam', choices=['greedy', 'green_ctc_beam', 'family_aware_beam'])
    ap.add_argument('--beam-size', type=int, default=20)
    ap.add_argument('--beam-topk', type=int, default=12)
    ap.add_argument('--min-track-len', type=int, default=8)
    ap.add_argument('--min-suffix-share', type=float, default=0.85)
    ap.add_argument('--min-score-margin', type=float, default=3.0)
    ap.add_argument('--min-winner-share', type=float, default=0.60)
    ap.add_argument('--min-occ-ratio', type=float, default=0.70)
    args = ap.parse_args()

    model_path = Path(args.model)
    input_csvs = [Path(p) for p in args.input_csv]
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv) if args.out_csv else out_json.with_suffix('.csv')
    temp_manifest = out_json.parent / (out_json.stem + '.manifest.csv')

    meta_rows = build_temp_manifest(input_csvs, temp_manifest)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds_full = UnifiedManifestDataset(
        manifest_path=str(temp_manifest),
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
    idx = [i for i, row in enumerate(ds_full.records) if Path(row.get('img_path', '')).exists()]
    ds = Subset(ds_full, idx)
    meta_rows = [meta_rows[i] for i in idx]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    net = load_model(model_path, device)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction='none', zero_infinity=True)

    frame_rows = []
    meta_cursor = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            images = images.to(device)
            sample_families = list(families)
            logits_t = forward_family_logits(net, images, sample_families=sample_families)
            logits_np = logits_t.detach().cpu().numpy()
            province_probs = torch.softmax(logits_t[:, :PROVINCE_COUNT, :4].mean(dim=2), dim=1).detach().cpu().numpy()
            family_decoded = decode_logits(logits_np, args.decode_mode, args.beam_size, args.beam_topk, sample_families=sample_families)
            greedy_decoded = greedy_decode_logits(logits_np)
            for batch_idx in range(len(sample_families)):
                meta = dict(meta_rows[meta_cursor])
                meta_cursor += 1
                meta['family_aware_pred'] = ''.join(CHARS[int(c)] for c in family_decoded[batch_idx])
                meta['board_greedy_pred'] = ''.join(CHARS[int(c)] for c in greedy_decoded[batch_idx])
                meta['province_top5'] = topk_chars(province_probs[batch_idx], k=5)
                meta['logits'] = logits_np[batch_idx]
                frame_rows.append(meta)

    tracks = defaultdict(list)
    for row in frame_rows:
        tracks[row['_track_id']].append(row)

    track_reports = []
    for track_id, rows in sorted(tracks.items()):
        track_reports.append(summarize_track(track_id, rows, args, ctc_loss))

    baseline_exact = sum(t['baseline_exact_match'] for t in track_reports)
    baseline_first = sum(t['baseline_first_char_match'] for t in track_reports)
    track_exact = sum(t['track_exact_match'] for t in track_reports)
    track_first = sum(t['track_first_char_match'] for t in track_reports)
    activated_count = sum(int(t['activated']) for t in track_reports)
    accepted_count = sum(int(t['accepted']) for t in track_reports)
    abstain_count = sum(int(t['decision'] == 'abstain_fallback') for t in track_reports)
    correct_to_wrong = sum(int(t['baseline_exact_match'] == 1 and t['track_exact_match'] == 0) for t in track_reports)
    wrong_to_correct = sum(int(t['baseline_exact_match'] == 0 and t['track_exact_match'] == 1) for t in track_reports)
    first_wrong_to_correct = sum(int(t['baseline_first_char_match'] == 0 and t['track_first_char_match'] == 1) for t in track_reports)
    first_correct_to_wrong = sum(int(t['baseline_first_char_match'] == 1 and t['track_first_char_match'] == 0) for t in track_reports)

    summary = {
        'model': str(model_path),
        'input_csvs': [str(p) for p in input_csvs],
        'decode_mode': args.decode_mode,
        'thresholds': {
            'min_track_len': args.min_track_len,
            'min_suffix_share': args.min_suffix_share,
            'min_score_margin': args.min_score_margin,
            'min_winner_share': args.min_winner_share,
            'min_occ_ratio': args.min_occ_ratio,
        },
        'track_count': len(track_reports),
        'activated_count': activated_count,
        'accepted_count': accepted_count,
        'abstain_count': abstain_count,
        'baseline_track_exact_acc': safe_div(baseline_exact, len(track_reports)),
        'baseline_track_first_char_acc': safe_div(baseline_first, len(track_reports)),
        'mvp_track_exact_acc': safe_div(track_exact, len(track_reports)),
        'mvp_track_first_char_acc': safe_div(track_first, len(track_reports)),
        'wrong_to_correct_count': wrong_to_correct,
        'correct_to_wrong_count': correct_to_wrong,
        'first_wrong_to_correct_count': first_wrong_to_correct,
        'first_correct_to_wrong_count': first_correct_to_wrong,
        'track_reports': track_reports,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write('\n')

    csv_rows = []
    for t in track_reports:
        csv_rows.append({
            'track_id': t['track_id'],
            'gt_text': t['gt_text'],
            'track_len': t['track_len'],
            'suffix_consensus': t['suffix_consensus'],
            'suffix_consensus_share': round(t['suffix_consensus_share'], 6),
            'eligible_frame_count': t['eligible_frame_count'],
            'activated': int(t['activated']),
            'accepted': int(t['accepted']),
            'decision': t['decision'],
            'baseline_pred_text': t['baseline_pred_text'],
            'track_pred_text': t['track_pred_text'],
            'best_province': t['best_province'],
            'second_province': t['second_province'],
            'track_score_margin': round(t['track_score_margin'], 6),
            'winner_share': round(t['winner_share'], 6),
            'baseline_exact_match': t['baseline_exact_match'],
            'baseline_first_char_match': t['baseline_first_char_match'],
            'track_exact_match': t['track_exact_match'],
            'track_first_char_match': t['track_first_char_match'],
        })
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ['track_id'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps({k: v for k, v in summary.items() if k != 'track_reports'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
