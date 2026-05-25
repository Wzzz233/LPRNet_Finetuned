#!/usr/bin/env python3
"""
Replay StageA checkpoints on redesigned foundation proxies.
Uses evaluators that actually emit JSON files and includes iteration checkpoints.
"""

import json
import subprocess
from pathlib import Path

BASE = Path('/home/wzzz/LPRNet')
PY = 'python3'


def run_json(cmd, workdir):
    subprocess.run(cmd, cwd=workdir, check=True, capture_output=True, text=True)
    out_json = Path(cmd[cmd.index('--out_json') + 1])
    if not out_json.is_absolute():
        out_json = Path(workdir) / out_json
    with open(out_json, encoding='utf-8') as f:
        return json.load(f)


def summarize_blue_family_aware(fa):
    fam = fa.get('families', {}).get('normal7', {})
    return {
        'exact_plate_acc': fam.get('exact_plate_acc', 0.0),
        'first_char_acc': fam.get('first_char_acc', 0.0),
    }


def score(blue, green, mixed):
    return (
        0.40 * blue.get('exact_plate_acc', 0.0)
        + 0.30 * green.get('exact_plate_acc', 0.0)
        + 0.15 * mixed.get('families', {}).get('normal7', {}).get('exact_plate_acc', 0.0)
        + 0.15 * mixed.get('families', {}).get('green8', {}).get('exact_plate_acc', 0.0)
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True)
    ap.add_argument('--manifest_dir', required=True)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    manifest_dir = Path(args.manifest_dir)
    replay_dir = exp_dir / 'stageA_replay_foundation'
    replay_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(exp_dir.glob('*.pth'))
    iter_ckpts = sorted(Path(exp_dir.parent).glob(f'{exp_dir.name}LPRNet__iteration_*.pth'))
    ckpts.extend(iter_ckpts)
    # de-dup while preserving order
    seen = set()
    unique_ckpts = []
    for p in ckpts:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        unique_ckpts.append(p)
    ckpts = unique_ckpts
    if not ckpts:
        raise SystemExit('no checkpoints found')

    rows = []
    for ckpt in ckpts:
        stem = ckpt.stem.replace('/', '_')
        blue_json = (replay_dir / f'{stem}_blue.json').resolve()
        green_json = (replay_dir / f'{stem}_green.json').resolve()
        mixed_json = (replay_dir / f'{stem}_mixed.json').resolve()

        blue_fa = run_json([
            PY, 'src/evaluation/eval_family_aware_blue_green_by_province.py',
            '--model', str(ckpt),
            '--manifest', str(manifest_dir / 'proxy_stageA_blue_simple.csv'),
            '--out_json', str(blue_json),
            '--batch_size', '300',
            '--num_workers', '4',
        ], BASE)
        blue = summarize_blue_family_aware(blue_fa)

        green = run_json([
            PY, 'src/evaluation/eval_green8_metrics_only.py',
            '--model', str(ckpt),
            '--manifest', str(manifest_dir / 'proxy_stageA_green_simple.csv'),
            '--out_json', str(green_json),
            '--batch_size', '300',
            '--num_workers', '4',
            '--ocr_preproc', 'gray3',
        ], BASE)

        mixed = run_json([
            PY, 'src/evaluation/eval_family_aware_blue_green_by_province.py',
            '--model', str(ckpt),
            '--manifest', str(manifest_dir / 'proxy_stageA_mixed_foundation.csv'),
            '--out_json', str(mixed_json),
            '--batch_size', '300',
            '--num_workers', '4',
        ], BASE)

        s = score(blue, green, mixed)
        rows.append({
            'checkpoint': str(ckpt),
            'score': s,
            'blue_exact': blue.get('exact_plate_acc', 0.0),
            'green_exact': green.get('exact_plate_acc', 0.0),
            'mixed_normal7_exact': mixed.get('families', {}).get('normal7', {}).get('exact_plate_acc', 0.0),
            'mixed_green8_exact': mixed.get('families', {}).get('green8', {}).get('exact_plate_acc', 0.0),
        })

    rows.sort(key=lambda x: x['score'], reverse=True)
    with open(replay_dir / 'ranking.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    md = ['# StageA Replay Ranking', '']
    for i, r in enumerate(rows, 1):
        md.append(f"{i}. {Path(r['checkpoint']).name} | score={r['score']:.4f} | blue={r['blue_exact']:.4f} | green={r['green_exact']:.4f} | normal7={r['mixed_normal7_exact']:.4f} | green8={r['mixed_green8_exact']:.4f}")
    (replay_dir / 'ranking.md').write_text('\n'.join(md), encoding='utf-8')
    print(replay_dir / 'ranking.json')
    print(replay_dir / 'ranking.md')


if __name__ == '__main__':
    main()
