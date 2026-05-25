#!/usr/bin/env python3
"""
Board-side gray3 OCR dump evaluation — CORRECTED ground truth.

Uses cluster*_wsl.csv files where gt_text is the verified ground truth
(unlike app_text in index.csv which is the board's OCR prediction).

Pipeline: ocrin PPM → (img-127.5)*0.0078125 → PyTorch multihead → CTC decode → compare to gt_text.
"""
import argparse, csv, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'training'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from load_data import CHARS
from training.train_LPRNet import forward_family_logits
from evaluation.test_LPRNet import greedy_decode_logits


def load_ocrin_ppm(path: Path) -> np.ndarray:
    """Load PPM P6 → uint8 BGR (board-side gray3 output, already gray)."""
    blob = path.read_bytes()
    assert blob.startswith(b'P6'), f'not PPM P6: {path}'
    i = 2
    tokens = []
    n = len(blob)
    while len(tokens) < 3:
        while i < n and blob[i] in b' \t\r\n':
            i += 1
        if i < n and blob[i] == ord('#'):
            while i < n and blob[i] not in b'\r\n':
                i += 1
            continue
        j = i
        while j < n and blob[j] not in b' \t\r\n':
            j += 1
        tokens.append(blob[i:j].decode('ascii'))
        i = j
    w, h, maxv = map(int, tokens)
    assert maxv == 255, f'unsupported max value: {maxv}'
    while i < n and blob[i] in b' \t\r\n':
        i += 1
    payload = np.frombuffer(blob[i:], dtype=np.uint8)
    assert payload.size == w * h * 3
    return payload.reshape(h, w, 3).copy()


def normalize_board_input(img: np.ndarray) -> torch.Tensor:
    """(img - 127.5) * 0.0078125 → CHW → batch dim."""
    x = img.astype(np.float32)
    x -= 127.5
    x *= 0.0078125
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def decode_ctc(logits: torch.Tensor) -> str:
    """Greedy CTC decode from family head logits."""
    prebs = logits.detach().cpu().numpy()
    decoded = greedy_decode_logits(prebs)
    return ''.join(CHARS[int(c)] for c in decoded[0])


def load_model(weights_path: str, device: torch.device):
    state = torch.load(weights_path, map_location=device)
    net, _cfg = build_lprnet_multihead_from_state_dict(
        state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0,
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser(description='Evaluate on board-side gray3 dump with gt_text labels')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--csv', default='/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster_assignments_wsl.csv',
                    help='CSV with gt_text column')
    ap.add_argument('--family', default='green8')
    ap.add_argument('--out_json', default='')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    assert csv_path.exists(), f'CSV not found: {csv_path}'

    # Parse CSV with BOM support
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            gt = (row.get('gt_text') or '').strip()
            ocrin = (row.get('local_ocrin_path') or row.get('ocr_input_path') or '')
            if not gt or not ocrin:
                continue
            records.append({'gt': gt, 'ocrin_path': Path(ocrin)})

    print(f'Loaded {len(records)} records with GT from {csv_path.name}')

    # Also try the gray3_ocr_demp directory if it has samples not in the CSV
    # (user says only one GT there: 苏BF01111)
    gray3_dir = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/gray3_ocr_demp')
    if gray3_dir.exists() and 'gray3' not in str(csv_path):
        # Only add if not already covered
        print(f'  (gray3_ocr_demp not used directly — use cluster CSV which has verified gt_text)')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    net = load_model(args.weights, device)
    print(f'Weights: {args.weights}')

    results = []
    exact_ok = first_ok = 0
    total = 0

    for r in records:
        ocrin_path = r['ocrin_path']
        gt = r['gt']
        if not ocrin_path.exists():
            print(f'  WARNING: ocrin not found: {ocrin_path}')
            continue

        img = load_ocrin_ppm(ocrin_path)
        x = normalize_board_input(img).to(device)

        with torch.no_grad():
            logits_dict = forward_family_logits(net, x, return_all=True)
        logits = logits_dict.get(args.family, logits_dict.get('normal7'))
        pred = decode_ctc(logits)

        exact = (pred == gt)
        first = (len(pred) > 0 and len(gt) > 0 and pred[0] == gt[0])

        if exact:
            exact_ok += 1
        if first:
            first_ok += 1
        total += 1

        results.append({
            'ocrin': str(ocrin_path.name),
            'gt': gt,
            'pred': pred,
            'exact': exact,
            'first_char': first,
        })

    exact_rate = exact_ok / total if total else 0.0
    first_rate = first_ok / total if total else 0.0

    print(f'\n{"="*60}')
    print(f'Weights: {args.weights}')
    print(f'Family head: {args.family}')
    print(f'Total with GT: {total}')
    print(f'Exact match: {exact_ok}/{total} = {exact_rate*100:.2f}%')
    print(f'First-char:  {first_ok}/{total} = {first_rate*100:.2f}%')

    # Per-sample
    print(f'\n{"#":>3} {"GT":<12} {"PRED":<12} {"exact":>5} {"first":>5}')
    print('-' * 42)
    for i, r in enumerate(results):
        e = 'Y' if r['exact'] else 'N'
        f = 'Y' if r['first_char'] else 'N'
        print(f'{i:>3} {r["gt"]:<12} {r["pred"]:<12} {e:>5} {f:>5}')

    summary = {
        'weights': str(args.weights),
        'family': args.family,
        'total': total,
        'exact_ok': exact_ok,
        'exact_rate': exact_rate,
        'first_char_ok': first_ok,
        'first_char_rate': first_rate,
        'results': results,
    }

    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f'\nSaved: {args.out_json}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
