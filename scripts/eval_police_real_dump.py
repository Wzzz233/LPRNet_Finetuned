#!/usr/bin/env python3
"""
Evaluate police LPRNet on real board dump ocrin images.

Usage:
  python3 scripts/eval_police_real_dump.py \
    --image-dir /path/to/police_dump \
    --checkpoint experiments/police_v2_fullft_officialwarm_20260601/best_LPRNet_model.pth \
    --keys-file keys/police_keys.txt \
    --output-dir experiments/police_real_dump_eval_20260603 \
    --device cuda:0

Output:
  - inventory.json        image count, types, dimensions
  - predictions.csv       per-file pred, GT (if available), legal checks
  - metrics.json          aggregated accuracy and distribution stats
  - bad_cases.csv         samples failing legal-format checks
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

# Path setup to import src modules
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent / 'src'
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'utils'), str(_SRC_DIR / 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from LPRNet import build_lprnet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_keys(path: str) -> list[str]:
    keys = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                keys.append(line)
    return keys


def greedy_decode_logits(prebs: np.ndarray, blank_idx: int) -> list[list[int]]:
    """Greedy CTC decode with configurable blank index."""
    preb_labels = []
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]                       # (C, T)
        argmax = np.argmax(preb, axis=0)             # (T,)
        no_repeat = []
        prev = argmax[0]
        if prev != blank_idx:
            no_repeat.append(int(prev))
        for c in argmax:
            if c == prev or c == blank_idx:
                if c == blank_idx:
                    prev = int(c)
                continue
            no_repeat.append(int(c))
            prev = int(c)
        preb_labels.append(no_repeat)
    return preb_labels


def preprocess(img_bgr: np.ndarray, img_size=(94, 24)) -> np.ndarray:
    """Match police LPRNet training preprocessing:
    - BGR input (OpenCV default)
    - INTER_LINEAR resize to (94, 24)
    - (pixel - 127.5) * 0.0078125
    - CHW
    """
    h, w = img_bgr.shape[:2]
    if h != img_size[1] or w != img_size[0]:
        img_bgr = cv2.resize(img_bgr, img_size, interpolation=cv2.INTER_LINEAR)
    img = img_bgr.astype('float32')
    img = (img - 127.5) * 0.0078125
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img


def indices_to_text(indices: list[int], keys: list[str]) -> str:
    return ''.join(keys[i] for i in indices if i < len(keys))


def legal_format_check(pred: str, keys: list[str]):
    """Police plate legal format:
    pos0 = province (first 31 keys)
    pos1 = letter (alpha ASCII)
    len = 7
    pos6 = 警 (last key)
    """
    prov_set = set(keys[:31])
    jing = keys[-1]  # 警
    ok_len = len(pred) == 7
    ok_pos0 = len(pred) > 0 and pred[0] in prov_set
    ok_pos1 = len(pred) > 1 and pred[1].isascii() and pred[1].isalpha()
    ok_tail = ok_len and pred[6] == jing
    legal = ok_len and ok_pos0 and ok_pos1 and ok_tail
    return legal, ok_len, ok_pos0, ok_pos1, ok_tail


def build_inventory(image_dir: Path) -> dict:
    """Count images by suffix and measure sizes."""
    inv = {'total_files': 0, 'by_ext': Counter(), 'size_distribution': {},
           'ocrin_count': 0, 'fc224_count': 0, 'crop_count': 0, 'coarse_count': 0}
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.ppm')
    sizes = Counter()
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() in exts:
            inv['total_files'] += 1
            inv['by_ext'][p.suffix.lower()] += 1
            if p.name.startswith('ocrin_'):
                inv['ocrin_count'] += 1
            elif p.name.startswith('fc224_'):
                inv['fc224_count'] += 1
            elif p.name.startswith('crop_'):
                inv['crop_count'] += 1
            elif p.name.startswith('coarse_'):
                inv['coarse_count'] += 1
            if p.name.startswith('ocrin_') or p.name.startswith('crop_') or p.name.startswith('fc224_'):
                try:
                    im = cv2.imread(str(p))
                    if im is not None:
                        sizes[f'{im.shape[1]}x{im.shape[0]}'] += 1
                except Exception:
                    pass
    inv['by_ext'] = dict(inv['by_ext'])
    inv['size_distribution'] = dict(sizes)

    # Check auxiliary files
    aux = list(image_dir.glob('*.csv')) + list(image_dir.glob('*.txt')) + list(image_dir.glob('*.json'))
    inv['auxiliary_files'] = [p.name for p in sorted(aux)]
    inv['quad_txt_count'] = len(list(image_dir.glob('*.quad.txt')))

    return inv


def load_index_csv(image_dir: Path) -> dict[str, str] | None:
    """Parse index.csv for app_text as reference (not verified GT)."""
    csv_path = image_dir / 'index.csv'
    if not csv_path.exists():
        return None
    ref = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Sample id mapping: row['ocr_input_path'] has full path
            # We map by frame_id which corresponds to the filename
            frame_key = f'ocrin_{row["sample_id"]}_f{row["frame_id"]}.ppm'
            ref[frame_key] = row
    return ref


def write_summary(metrics: dict, output_dir: Path, keys: list[str]):
    """Write human-readable summary.md."""
    prov_set = set(keys[:31])
    jing_char = keys[-1]

    lines = []
    lines.append("# Police Real Dump Evaluation Summary")
    lines.append("")
    lines.append(f"- **Model**: `{metrics['model']}`")
    lines.append(f"- **Keys file**: `{metrics['keys_file']}` ({len(keys)} keys, blank idx={len(keys)})")
    lines.append(f"- **Samples**: {metrics['total_samples']}")
    lines.append(f"- **Device**: {metrics['device']}")
    lines.append("")

    lines.append("## Legal Format Compliance")
    lines.append("")
    lines.append(f"| Metric | Count | Rate |")
    lines.append(f"|--------|------:|:---:|")
    lines.append(f"| Legal format (7 chars, province+letter+4+警) | {metrics.get('legal_count',0)}/{metrics['total_samples']} | {metrics['legal_format_rate']*100:.2f}% |")
    lines.append(f"| Correct length (7) | {metrics.get('len_ok_count',0)}/{metrics['total_samples']} | {metrics['len_ok_rate']*100:.2f}% |")
    lines.append(f"| Province char (pos0) | {metrics.get('pos0_ok_count',0)}/{metrics['total_samples']} | {metrics['pos0_ok_rate']*100:.2f}% |")
    lines.append(f"| Letter (pos1) | {metrics.get('pos1_ok_count',0)}/{metrics['total_samples']} | {metrics['pos1_ok_rate']*100:.2f}% |")
    lines.append(f"| Tail 警 (pos6) | {metrics.get('tail_ok_count',0)}/{metrics['total_samples']} | {metrics['tail_jing_ok_rate']*100:.2f}% |")
    lines.append("")

    lines.append("## Length Distribution")
    lines.append("")
    for ln, cnt in sorted(metrics.get('len_distribution', {}).items(), key=lambda x: int(x[0])):
        lines.append(f"- Length {ln}: {cnt} samples ({cnt/metrics['total_samples']*100:.1f}%)")
    lines.append("")

    lines.append("## Province (pos0) Distribution")
    lines.append("")
    for prov, cnt in metrics.get('province_distribution', {}).items():
        mark = " ✅" if prov in prov_set else " ❌"
        lines.append(f"- {prov}: {cnt}{mark}")
    lines.append("")

    lines.append("## Second Char (pos1) Distribution")
    lines.append("")
    for ch, cnt in metrics.get('pos1_distribution', {}).items():
        is_letter = ch.isascii() and ch.isalpha()
        mark = " ✅" if is_letter else " ❌"
        lines.append(f"- {ch}: {cnt}{mark}")
    lines.append("")

    lines.append("## Tail Char (last pos) Distribution")
    lines.append("")
    for ch, cnt in metrics.get('tail_distribution', {}).items():
        mark = f" ✅ (警)" if ch == jing_char else f" ❌ (expected {jing_char})"
        lines.append(f"- {ch}: {cnt}{mark}")
    lines.append("")

    lines.append("## Bad Cases")
    lines.append("")
    bc = metrics.get('bad_cases_count', 0)
    lines.append(f"Total non-legal samples: {bc}")
    if bc > 0:
        lines.append("")
        lines.append("| File | Pred | Legal | Len | Pos0 | Pos1 | Tail |")
        lines.append("|------|------|:----:|:---:|:----:|:----:|:----:|")
        for r in metrics.get('bad_cases', []):
            lines.append(f"| {r['file']} | {r['pred']} | {r['legal']} | {r['len_ok']} | {r['pos0_ok']} | {r['pos1_ok']} | {r['tail_jing_ok']} |")

    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    loss_breakdown = []
    if metrics['pos0_ok_rate'] < 0.95:
        loss_breakdown.append(f"- **Province (pos0)**: {metrics['pos0_ok_rate']*100:.1f}% — model struggles with first character (province)")
    if metrics['pos1_ok_rate'] < 0.95:
        loss_breakdown.append(f"- **Second char (pos1)**: {metrics['pos1_ok_rate']*100:.1f}% — letter/character at position 1 is unstable")
    if metrics['len_ok_rate'] < 0.95:
        loss_breakdown.append(f"- **Length error**: {metrics['len_ok_rate']*100:.1f}% — model outputs wrong sequence length (collapse/extra)")
    if metrics['tail_jing_ok_rate'] < 0.95:
        loss_breakdown.append(f"- **Tail 警**: {metrics['tail_jing_ok_rate']*100:.1f}% — 警 character at end is wrong/missing")

    if not loss_breakdown:
        lines.append("All metrics above 95% — model performs well on this dump's legal format.")
    else:
        lines.extend(loss_breakdown)

    with open(output_dir / 'summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[Info] summary written to {output_dir / 'summary.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Eval police LPRNet on real board dump')
    parser.add_argument('--image-dir', required=True, help='path to police_dump directory')
    parser.add_argument('--checkpoint', required=True, help='LPRNet .pth checkpoint')
    parser.add_argument('--keys-file', required=True, help='police keys file (one char per line)')
    parser.add_argument('--output-dir', required=True, help='output directory for eval results')
    parser.add_argument('--device', default='cuda:0', help='torch device (auto falls back to cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[Info] Device: {device}')

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Inventory
    # -----------------------------------------------------------------------
    print(f'[Step 1] Building inventory from {image_dir}')
    inventory = build_inventory(image_dir)
    with open(output_dir / 'inventory.json', 'w', encoding='utf-8') as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)
    print(f'[Info] inventory.json: {inventory["total_files"]} files, {inventory["ocrin_count"]} ocrin images')

    # Load index.csv reference if available
    index_ref = load_index_csv(image_dir)
    if index_ref:
        print(f'[Info] index.csv found with {len(index_ref)} entries (board OCR output, not verified GT)')

    # -----------------------------------------------------------------------
    # 2. Collect ocrin images
    # -----------------------------------------------------------------------
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.ppm')
    ocrin_files = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in exts and p.name.startswith('ocrin_')
    ])
    if not ocrin_files:
        print(f'[Error] No ocrin_*.ppm files found in {image_dir}')
        sys.exit(1)
    print(f'[Step 2] Found {len(ocrin_files)} ocrin images')

    # Check which frame IDs exist (to cross-ref with index.csv)
    frame_ids = set()
    for p in ocrin_files:
        # Parse sample_id and frame_id from filename
        # ocrin_SSSS_fFFFFFFFF.ppm
        parts = p.stem.split('_')
        if len(parts) >= 3:
            frame_ids.add(parts[-1].lstrip('f'))

    # -----------------------------------------------------------------------
    # 3. Load model and keys
    # -----------------------------------------------------------------------
    print(f'[Step 3] Loading keys from {args.keys_file}')
    keys = load_keys(args.keys_file)
    blank_idx = len(keys)
    class_num = len(keys) + 1
    print(f'[Info] Keys: {len(keys)}, class_num={class_num}, blank_idx={blank_idx}')

    print(f'[Info] Loading checkpoint {args.checkpoint}')
    net = build_lprnet(lpr_max_len=8, phase=False, class_num=class_num, dropout_rate=0)
    state = torch.load(args.checkpoint, map_location='cpu')
    # Strip 'module.' prefix from DDP-wrapped checkpoints
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    print('[Info] Model loaded successfully')

    # -----------------------------------------------------------------------
    # 4. Inference
    # -----------------------------------------------------------------------
    print(f'[Step 4] Running inference on {len(ocrin_files)} images')
    results = []
    for img_path in ocrin_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'  WARN: failed to read {img_path}', file=sys.stderr)
            continue

        x = preprocess(img)
        x_t = torch.from_numpy(x[None, ...]).to(device)

        with torch.no_grad():
            logits = net(x_t)
        logits_np = logits.cpu().numpy()

        pred_indices = greedy_decode_logits(logits_np, blank_idx)[0]
        pred_text = indices_to_text(pred_indices, keys)

        # Legal format check
        legal, len_ok, pos0_ok, pos1_ok, tail_jing_ok = legal_format_check(pred_text, keys)

        # Look up reference text from index.csv
        ref_text = None
        ref_conf = None
        fname = img_path.name
        if fname in index_ref:
            ref_text = index_ref[fname].get('app_text', '')
            ref_conf = index_ref[fname].get('app_conf', '')

        results.append({
            'file': fname,
            'pred': pred_text,
            'ref': ref_text or '',
            'ref_conf': ref_conf or '',
            'legal': legal,
            'len_ok': len_ok,
            'pos0_ok': pos0_ok,
            'pos1_ok': pos1_ok,
            'tail_jing_ok': tail_jing_ok,
        })

    n = len(results)
    print(f'[Info] Inference complete: {n} results')

    # -----------------------------------------------------------------------
    # 5. Write predictions.csv
    # -----------------------------------------------------------------------
    fieldnames = ['file', 'pred', 'ref', 'ref_conf', 'legal', 'len_ok', 'pos0_ok', 'pos1_ok', 'tail_jing_ok']
    csv_path = output_dir / 'predictions.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f'[Info] predictions.csv written to {csv_path}')

    # -----------------------------------------------------------------------
    # 6. Aggregate metrics
    # -----------------------------------------------------------------------
    n_legal = sum(r['legal'] for r in results)
    n_len_ok = sum(r['len_ok'] for r in results)
    n_pos0_ok = sum(r['pos0_ok'] for r in results)
    n_pos1_ok = sum(r['pos1_ok'] for r in results)
    n_tail_ok = sum(r['tail_jing_ok'] for r in results)

    len_dist = Counter(len(r['pred']) for r in results)
    province_dist = Counter(r['pred'][0] for r in results if len(r['pred']) > 0)
    pos1_dist = Counter(r['pred'][1] for r in results if len(r['pred']) > 1)
    tail_dist = Counter(r['pred'][-1] for r in results if len(r['pred']) > 0)

    bad_cases = [r for r in results if not r['legal']]

    metrics = {
        'model': args.checkpoint,
        'keys_file': args.keys_file,
        'device': str(device),
        'total_samples': n,
        'legal_count': n_legal,
        'legal_format_rate': n_legal / n if n else 0,
        'len_ok_count': n_len_ok,
        'len_ok_rate': n_len_ok / n if n else 0,
        'pos0_ok_count': n_pos0_ok,
        'pos0_ok_rate': n_pos0_ok / n if n else 0,
        'pos1_ok_count': n_pos1_ok,
        'pos1_ok_rate': n_pos1_ok / n if n else 0,
        'tail_ok_count': n_tail_ok,
        'tail_jing_ok_rate': n_tail_ok / n if n else 0,
        'len_distribution': {str(k): v for k, v in sorted(len_dist.items())},
        'province_distribution': dict(province_dist.most_common()),
        'pos1_distribution': dict(pos1_dist.most_common()),
        'tail_distribution': dict(tail_dist.most_common()),
        'bad_cases_count': len(bad_cases),
        'bad_cases': [{'file': r['file'], 'pred': r['pred'], 'legal': r['legal'],
                        'len_ok': r['len_ok'], 'pos0_ok': r['pos0_ok'],
                        'pos1_ok': r['pos1_ok'], 'tail_jing_ok': r['tail_jing_ok']}
                      for r in bad_cases],
    }

    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f'[Info] metrics.json written')

    # Bad cases CSV
    if bad_cases:
        bc_path = output_dir / 'bad_cases.csv'
        with open(bc_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(bad_cases)
        print(f'[Info] bad_cases.csv: {len(bad_cases)} samples')
    else:
        print('[Info] No bad cases')

    # -----------------------------------------------------------------------
    # 7. Summary
    # -----------------------------------------------------------------------
    write_summary(metrics, output_dir, keys)

    # Print headline
    print()
    print('=' * 60)
    print('HEADLINE RESULTS')
    print('=' * 60)
    print(f'Total samples:        {n}')
    print(f'Legal format:         {n_legal}/{n} = {n_legal/n*100:.2f}%')
    print(f'  Length (7 chars):   {n_len_ok}/{n} = {n_len_ok/n*100:.2f}%')
    print(f'  Province (pos0):    {n_pos0_ok}/{n} = {n_pos0_ok/n*100:.2f}%')
    print(f'  Letter (pos1):      {n_pos1_ok}/{n} = {n_pos1_ok/n*100:.2f}%')
    print(f'  Tail 警 (pos6):     {n_tail_ok}/{n} = {n_tail_ok/n*100:.2f}%')
    print(f'Bad cases:            {len(bad_cases)}')
    print('=' * 60)
    if bad_cases:
        print('First 10 bad cases:')
        for r in bad_cases[:10]:
            issues = []
            if not r['len_ok']: issues.append('len')
            if not r['pos0_ok']: issues.append('pos0')
            if not r['pos1_ok']: issues.append('pos1')
            if not r['tail_jing_ok']: issues.append('tail')
            print(f'  {r["file"]}: pred="{r["pred"]}" [{"|".join(issues)}]')


if __name__ == '__main__':
    main()
