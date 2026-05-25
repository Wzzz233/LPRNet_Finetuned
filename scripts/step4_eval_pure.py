#!/usr/bin/env python3
"""Pure evaluation — no training path. Reports exact, char, province, short_pred metrics.
Evaluates old expert vs posquad-trained model on CCPD2019 pose-quad test set
plus regression check on standard blue validation sets."""

import csv, json, sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import CHARS, UnifiedManifestDataset
from LPRNet import build_lprnet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

OCR_PARAMS = dict(
    ocr_crop_mode='obb_warp', ocr_resize_mode='letterbox',
    ocr_resize_kernel='nn', ocr_preproc='none',
    ocr_channel_order='bgr', ocr_quad_pad_ratio=0.0,
)

# Province list (first 31 chars of CHARS = provinces)
PROVINCES = CHARS[:31]

MODELS = {
    'old_expert': ROOT / 'experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth',
    'posquad_trained': ROOT / 'experiments/blue_ccpd2019_tilt_db_challenge_posquad_20260508/best_LPRNet_model.pth',
    'posquad_final': ROOT / 'experiments/blue_ccpd2019_tilt_db_challenge_posquad_20260508/Final_LPRNet_model.pth',
}

MANIFESTS = {
    'test_posquad': ROOT / 'manifests_rebased/blue_ccpd2019_tilt_db_challenge_posquad_20260508/test_posquad.csv',
    # Regression manifests
    'blue_simple': ROOT / 'manifests_rebased/curriculum_gray3/test_blue_simple.csv',
    'blue_ccpd2019': ROOT / 'manifests_rebased/curriculum_gray3/val_ccpd2019_blue.csv',
    'blue_hard': ROOT / 'manifests_rebased/curriculum_gray3/test_blue_hard.csv',
}

OUTPUT_JSON = ROOT / 'experiments/blue_ccpd2019_tilt_db_challenge_posquad_20260508/final_old_vs_new_eval.json'


def greedy_decode_ctc(logits, blank_idx):
    """Greedy CTC decode. logits: [N, C, T]. Returns list of label-id lists."""
    preds = logits.permute(2, 0, 1).cpu().numpy()  # [T, N, C]
    results = []
    for b in range(logits.shape[0]):
        pred_ids = np.argmax(preds[:, b, :], axis=1)  # [T]
        decoded = []
        prev = pred_ids[0]
        if prev != blank_idx:
            decoded.append(prev)
        for c in pred_ids:
            if c == prev or c == blank_idx:
                if c == blank_idx:
                    prev = c
                continue
            decoded.append(c)
            prev = c
        results.append(decoded)
    return results


def ids_to_text(ids):
    return ''.join(CHARS[i] for i in ids if 0 <= i < len(CHARS))


@torch.no_grad()
def evaluate_model(net, manifest_path, name=''):
    """Evaluate model on a manifest file. Returns detailed metrics dict."""
    dataset = UnifiedManifestDataset(
        str(manifest_path), img_size=[94, 24], lpr_max_len=8,
        split_filter='test',
        dataset_root=str(ROOT), **OCR_PARAMS,
    )
    if len(dataset) == 0:
        print(f"  [{name}] 0 samples, skip", flush=True)
        return None

    loader = DataLoader(
        dataset, batch_size=120, shuffle=False,
        num_workers=4,
        collate_fn=lambda b: (
            torch.from_numpy(np.stack([x[0] for x in b])),
            torch.from_numpy(np.concatenate([x[1] for x in b])),
            [x[2] for x in b],
            [x[3] for x in b] if len(b[0]) > 3 else ['normal7'] * len(b),
        ),
    )

    net.eval()
    blank_idx = len(CHARS) - 1

    total = 0
    exact_correct = 0
    char_correct = 0
    char_total = 0
    short_preds = 0
    province_correct = 0
    province_total = 0

    # Per-province tracking
    province_hits = defaultdict(int)
    province_counts = defaultdict(int)

    for images, labels, lengths, families in loader:
        images = images.to(device)
        logits = net(images)  # [N, C, T]

        pred_lists = greedy_decode_ctc(logits, blank_idx)

        # Parse GT from flat label tensor
        label_offset = 0
        for b in range(logits.shape[0]):
            gt_len = lengths[b]
            gt_ids = labels[label_offset:label_offset + gt_len].tolist()
            label_offset += gt_len
            gt_text = ids_to_text(gt_ids)

            pred_text = ids_to_text(pred_lists[b])

            total += 1
            if pred_text == gt_text:
                exact_correct += 1

            # Character accuracy
            for p, g in zip(pred_text, gt_text):
                if p == g:
                    char_correct += 1
            char_total += len(gt_text)

            # Short prediction rate (pred shorter than GT)
            if len(pred_text) < len(gt_text):
                short_preds += 1

            # Province first-char accuracy
            if gt_text and CHARS.index(gt_text[0]) < 31:
                province_total += 1
                province_counts[gt_text[0]] += 1
                if pred_text and pred_text[0] == gt_text[0]:
                    province_correct += 1
                    province_hits[gt_text[0]] += 1

    province_breakdown = {}
    for prov in sorted(province_counts.keys()):
        province_breakdown[prov] = {
            'count': province_counts[prov],
            'pct_of_total': province_counts[prov] / max(total, 1) * 100,
            'first_char_correct': province_hits.get(prov, 0),
            'first_char_acc': province_hits.get(prov, 0) / max(province_counts[prov], 1) * 100,
        }

    # Province macro first-char accuracy
    province_macro = np.mean([v['first_char_acc'] for v in province_breakdown.values()]) if province_breakdown else 0.0

    return {
        'sample_count': total,
        'exact_plate_acc': exact_correct / max(total, 1),
        'exact_correct': exact_correct,
        'char_acc': char_correct / max(char_total, 1),
        'char_correct': char_correct,
        'char_total': char_total,
        'short_pred_rate': short_preds / max(total, 1),
        'short_pred_count': short_preds,
        'province_first_char_acc': province_correct / max(province_total, 1),
        'province_macro_first_char_acc': float(province_macro),
        'province_correct': province_correct,
        'province_total': province_total,
        'province_breakdown': province_breakdown,
    }


def main():
    print(f"Device: {device}", flush=True)

    all_results = {}

    # ── Part 1: Pose-quad test set per-subset evaluation ──────────
    print(f"\n{'='*60}", flush=True)
    print(f"PART 1: CCPD2019 Pose-Quad Test Set (per-subset)", flush=True)
    print(f"{'='*60}", flush=True)

    # Create per-subset manifests from the combined test set
    all_rows = []
    with open(MANIFESTS['test_posquad'], encoding='utf-8') as f:
        for row in csv.DictReader(f):
            all_rows.append(row)

    by_subset = defaultdict(list)
    for row in all_rows:
        subset = row.get('source', '').replace('ccpd2019_', '')
        by_subset[subset].append(row)

    subset_manifest_paths = {}
    for subset, rows in by_subset.items():
        path = Path(f'/tmp/eval_posequad_{subset}.csv')
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            for r in rows:
                r['split'] = 'test'
                w.writerow(r)
        subset_manifest_paths[subset] = path

    # Combined full test
    combined_path = Path(f'/tmp/eval_posequad_all.csv')
    with open(combined_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        for r in all_rows:
            r['split'] = 'test'
            w.writerow(r)
    subset_manifest_paths['all'] = combined_path

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"  SKIP {model_name}: not found", flush=True)
            continue

        print(f"\n  Loading {model_name}...", flush=True)
        net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0.5)
        state = torch.load(str(model_path), map_location='cpu')
        net.load_state_dict(state, strict=False)
        net.to(device)
        net.eval()
        print(f"  Params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)

        all_results[model_name] = {}
        for subset_name, manifest_path in subset_manifest_paths.items():
            metrics = evaluate_model(net, manifest_path, name=f'{model_name}/{subset_name}')
            if metrics is not None:
                all_results[model_name][f'posequad_{subset_name}'] = metrics
                a = metrics['exact_plate_acc']
                c = metrics['char_acc']
                p = metrics['province_first_char_acc']
                s = metrics['short_pred_rate']
                print(f"    {subset_name:20s} exact={a*100:5.1f}%  char={c*100:5.1f}%  "
                      f"prov1st={p*100:5.1f}%  short={s*100:4.1f}%  "
                      f"(n={metrics['sample_count']})", flush=True)

        del net
        torch.cuda.empty_cache()

    # ── Part 2: Regression check on standard blue sets ────────────
    print(f"\n{'='*60}", flush=True)
    print(f"PART 2: Regression Check — Standard Blue Sets", flush=True)
    print(f"{'='*60}", flush=True)

    for set_name in ['blue_simple', 'blue_ccpd2019', 'blue_hard']:
        manifest_path = MANIFESTS.get(set_name)
        if not manifest_path or not manifest_path.exists():
            print(f"\n  [{set_name}] manifest not found at {manifest_path}", flush=True)
            continue

        print(f"\n  Manifest: {set_name} ({manifest_path.name})", flush=True)
        for model_name, model_path in MODELS.items():
            if model_name not in all_results:
                continue
            net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0.5)
            state = torch.load(str(model_path), map_location='cpu')
            net.load_state_dict(state, strict=False)
            net.to(device)
            net.eval()

            metrics = evaluate_model(net, manifest_path, name=f'{model_name}/{set_name}')
            if metrics is not None:
                all_results[model_name][set_name] = metrics
                a = metrics['exact_plate_acc']
                c = metrics['char_acc']
                p = metrics['province_first_char_acc']
                print(f"    {model_name:20s} exact={a*100:5.1f}%  char={c*100:5.1f}%  "
                      f"prov1st={p*100:5.1f}%  (n={metrics['sample_count']})", flush=True)

            del net
            torch.cuda.empty_cache()

    # ── Part 3: Summary comparison tables ─────────────────────────
    old = 'old_expert'
    new = 'posquad_trained'

    print(f"\n\n{'='*70}", flush=True)
    print(f"COMPARISON: Old Expert vs Posquad-Trained", flush=True)
    print(f"{'='*70}", flush=True)

    if old in all_results and new in all_results:
        # Table 1: Pose-quad test sets
        print(f"\n  --- Pose-Quad Test Sets ---", flush=True)
        print(f"  {'Set':22s} {'Old':>9s} {'New':>9s} {'ΔExact':>9s}  {'OldProv1st':>11s} {'NewProv1st':>11s}", flush=True)
        print(f"  {'-'*65}", flush=True)
        for subset in ['all', 'ccpd_tilt', 'ccpd_db', 'ccpd_challenge']:
            key = f'posequad_{subset}'
            if key not in all_results[old] or key not in all_results[new]:
                continue
            o = all_results[old][key]
            n = all_results[new][key]
            d = n['exact_plate_acc'] - o['exact_plate_acc']
            print(f"  {subset:22s} {o['exact_plate_acc']*100:8.1f}% {n['exact_plate_acc']*100:8.1f}% "
                  f"{d*100:+8.1f}pp  {o['province_first_char_acc']*100:10.1f}% {n['province_first_char_acc']*100:10.1f}%", flush=True)

        # Table 2: Regression check
        print(f"\n  --- Regression Check ---", flush=True)
        print(f"  {'Set':22s} {'Old':>9s} {'New':>9s} {'ΔExact':>9s}  {'ΔProv1st':>11s}", flush=True)
        print(f"  {'-'*60}", flush=True)
        for set_name in ['blue_simple', 'blue_ccpd2019', 'blue_hard']:
            if set_name not in all_results[old] or set_name not in all_results[new]:
                # fallback: try 'posquad_final' for the new model
                if set_name not in all_results.get('posquad_final', {}):
                    continue
            old_k = all_results[old].get(set_name, all_results[old].get(set_name))
            new_k = all_results[new].get(set_name, all_results.get('posquad_final', {}).get(set_name))
            if old_k is None or new_k is None:
                continue
            d = new_k['exact_plate_acc'] - old_k['exact_plate_acc']
            dp = new_k['province_first_char_acc'] - old_k['province_first_char_acc']
            print(f"  {set_name:22s} {old_k['exact_plate_acc']*100:8.1f}% {new_k['exact_plate_acc']*100:8.1f}% "
                  f"{d*100:+8.1f}pp  {dp*100:+8.1f}pp", flush=True)

        # Table 3: Per-province breakdown for the combined test set
        print(f"\n  --- Per-Province Breakdown (All) ---", flush=True)
        old_provs = all_results[old].get('posequad_all', {}).get('province_breakdown', {})
        new_provs = all_results[new].get('posequad_all', {}).get('province_breakdown', {})
        if old_provs and new_provs:
            all_provs = sorted(set(list(old_provs.keys()) + list(new_provs.keys())))
            print(f"  {'Prov':6s} {'Count':>6s} {'%Total':>7s} {'Old1st%':>8s} {'New1st%':>8s}", flush=True)
            total_all = all_results[old].get('posequad_all', {}).get('sample_count', 1)
            for p in all_provs:
                if p not in old_provs:
                    continue
                c = old_provs[p]['count']
                pct = c / max(total_all, 1) * 100
                oa = old_provs[p]['first_char_acc']
                na = new_provs.get(p, {}).get('first_char_acc', 0)
                marker = ' ***' if pct > 30 else ''
                print(f"  {p:6s} {c:6d} {pct:6.1f}%{marker} {oa:7.1f}% {na:7.1f}%", flush=True)

    # ── Save final JSON ───────────────────────────────────────────
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(OUTPUT_JSON, 'w'), ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {OUTPUT_JSON}", flush=True)


if __name__ == '__main__':
    main()
