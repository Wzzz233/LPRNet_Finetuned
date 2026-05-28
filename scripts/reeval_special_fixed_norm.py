"""
Re-evaluate embassy and police checkpoints with FIXED (per-sample) normalization.
Outputs:
  experiments/embassy_formal_20260526/acceptance_report_fixed_norm.txt
  experiments/police_probe_20260526_3k/acceptance_report_fixed_norm.txt
"""

import torch, numpy as np, sys, os
from datetime import datetime

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)

from LPRNet import build_lprnet
import load_data as ld
from load_data import UnifiedManifestDataset

EVALS = [
    {
        'name': 'embassy',
        'ckpt': 'experiments/embassy_formal_20260526/best_LPRNet_model.pth',
        'keys': 'keys/embassy_keys.txt',
        'manifest': 'manifests_rebased/special_split_20260526/val_embassy_only.csv',
        'report': 'experiments/embassy_formal_20260526/acceptance_report_fixed_norm.txt',
        'root': '.',
    },
    {
        'name': 'police',
        'ckpt': 'experiments/police_probe_20260526_3k/best_LPRNet_model.pth',
        'keys': 'keys/police_keys.txt',
        'manifest': 'manifests_rebased/special_split_20260526/val_police_only.csv',
        'report': 'experiments/police_probe_20260526_3k/acceptance_report_fixed_norm.txt',
        'root': '.',
    },
]


def load_keys(keys_path):
    chars_raw = []
    with open(keys_path) as f:
        for line in f:
            if c := line.strip():
                chars_raw.append(c)
    chars = chars_raw + ['-']
    return chars, len(chars)


def decode(prebs, chars, blank_idx):
    mg, pv = [], -1
    for t in range(prebs.shape[1]):
        idx = int(np.argmax(prebs[:, t]))
        if idx != pv and idx != blank_idx:
            mg.append(chars[idx])
        pv = idx
    return ''.join(mg)


def eval_checkpoint(cfg):
    name = cfg['name']
    ckpt_path = cfg['ckpt']
    keys_path = cfg['keys']
    manifest_path = cfg['manifest']
    report_path = cfg['report']
    root = cfg['root']

    chars, class_num = load_keys(keys_path)
    blank_idx = class_num - 1  # '-' appended as blank

    # Setup LD globals
    ld.CHARS.clear()
    ld.CHARS.extend(chars)
    ld.CHARS_DICT.clear()
    ld.CHARS_DICT.update({c: i for i, c in enumerate(chars)})

    # Load dataset
    ds = UnifiedManifestDataset(
        manifest_path=manifest_path, img_size=(94, 24), lpr_max_len=8,
        split_filter='val', dataset_root=root,
        ocr_channel_order='bgr', ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox', ocr_resize_kernel='nn', ocr_preproc='none',
        ocr_min_occ_ratio=0.0, ocr_quad_pad_ratio=0.0, gray3_prob=0.0,
    )
    paths = [ds.img_paths[i] for i in range(len(ds))]
    gts = [ds.records[i]['text'] for i in range(len(ds))]

    # Load model
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = ckpt.get('state_dict', ckpt)
    model = build_lprnet(lpr_max_len=8, phase='test', class_num=class_num, dropout_rate=0.0)
    model.load_state_dict(sd)
    model.eval()

    # Evaluate (single-image, fixed code)
    preds = []
    for idx in range(len(ds)):
        img, _, _, _ = ds[idx]
        img_t = torch.from_numpy(img).unsqueeze(0)  # [1, 3, 24, 94]
        with torch.no_grad():
            prebs = model(img_t).detach().cpu().numpy()[0]
        preds.append(decode(prebs, chars, blank_idx))

    # Metrics
    n = len(ds)
    exact_correct = sum(1 for i in range(n) if preds[i] == gts[i])
    len_errors = sum(1 for i in range(n) if preds[i] != gts[i] and len(preds[i]) != len(gts[i]))

    # Illegal chars
    legal_set = set(chars[:-1])  # exclude blank
    illegal_count = 0
    illegal_samples = []
    for i in range(n):
        illegal = [c for c in preds[i] if c not in legal_set]
        if illegal:
            illegal_count += 1
            if len(illegal_samples) < 10:
                illegal_samples.append(f'  idx={i}: pred={preds[i]} illegal={illegal}')

    # Error analysis: first 100 errors
    errors = [(i, gts[i], preds[i]) for i in range(n) if preds[i] != gts[i]]

    # Write report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write(f'{name.upper()} Re-Evaluation Report (FIXED normalization)\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 70 + '\n\n')

        f.write(f'Checkpoint: {ckpt_path}\n')
        f.write(f'Keys file:  {keys_path} ({class_num - 1} chars + blank)\n')
        f.write(f'Manifest:   {manifest_path}\n')
        f.write(f'Samples:    {n}\n')
        f.write(f'Code:       FIXED per-sample normalization (LPRNet.py batch-invariant)\n')
        f.write('\n')

        f.write('--- Accuracy ---\n')
        f.write(f'Exact match:     {exact_correct}/{n} = {exact_correct / n * 100:.2f}%\n')
        f.write(f'Errors:          {n - exact_correct}/{n}\n')
        f.write(f'Length errors:   {len_errors}/{n - exact_correct} (among errors)\n')
        f.write(f'Illegal chars:   {illegal_count} predictions\n')
        f.write('\n')

        if illegal_samples:
            f.write('--- Illegal Character Samples (first 10) ---\n')
            for s in illegal_samples:
                f.write(s + '\n')
        else:
            f.write('--- Illegal Characters ---\n')
            f.write('None detected.\n')
        f.write('\n')

        f.write('--- Top-100 GT vs Pred ---\n')
        f.write(f'{"idx":>4s} {"GT":>12s} {"pred":>12s} {"ok":>3s} {"len_err":>7s}\n')
        f.write('-' * 45 + '\n')
        for i in range(min(100, n)):
            ok = 'Y' if preds[i] == gts[i] else 'N'
            le = 'Y' if (preds[i] != gts[i] and len(preds[i]) != len(gts[i])) else ''
            f.write(f'{i:4d} {gts[i]:>12s} {preds[i]:>12s} {ok:>3s} {le:>7s}\n')
        if n > 100:
            f.write(f'... (showing first 100 of {n})\n')
        f.write('\n')

        if errors:
            f.write(f'--- All Errors ({len(errors)}) ---\n')
            f.write(f'{"idx":>4s} {"GT":>12s} {"pred":>12s}\n')
            f.write('-' * 35 + '\n')
            for idx, gt, pred in errors:
                f.write(f'{idx:4d} {gt:>12s} {pred:>12s}\n')
        else:
            f.write('--- Errors ---\n')
            f.write('None!\n')

    print(f'[{name}] {exact_correct}/{n} = {exact_correct/n*100:.2f}%  len_err={len_errors}  illegal={illegal_count}')
    print(f'        report: {report_path}')

    return exact_correct, n, len_errors, illegal_count


if __name__ == '__main__':
    print('=== LPRNet Special Checkpoint Re-Evaluation (FIXED normalization) ===')
    print()
    for cfg in EVALS:
        eval_checkpoint(cfg)
    print()
    print('Done.')
