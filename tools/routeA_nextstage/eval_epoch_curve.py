#!/usr/bin/env python3
"""Batch eval all intermediate epoch checkpoints + best/last on board dumps.

Usage:
  python tools/routeA_nextstage/eval_epoch_curve.py \
    --exp_dir experiments/routeA_epochcurve_20260512/G0_refit \
    --out_dir experiments/routeA_epochcurve_20260512/G0_refit/curve_evals
"""
import argparse, json, subprocess, sys, re
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
EVAL_SCRIPT = ROOT / 'tools/routeA_prime/eval_largecrop_and_fuse.py'
EVAL_BASE = ROOT / 'experiments/routeA_prime_quadwarp_20260512'


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', str(s))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--input_size', type=int, nargs=2, default=[224,72])
    ap.add_argument('--ocr_preproc', default='gray3')
    ap.add_argument('--in_channels', type=int, default=1)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all checkpoints
    ckpts = []

    # Intermediate checkpoints
    ckpt_dir = exp_dir / 'checkpoints'
    if ckpt_dir.exists():
        for f in sorted(ckpt_dir.glob('epoch_*.pt'), key=natural_sort_key):
            ckpts.append(('ckpt_' + f.stem, str(f)))

    # best.pt and last.pt
    for name in ['best', 'last']:
        p = exp_dir / f'{name}.pt'
        if p.exists():
            ckpts.append((f'{name}', str(p)))

    print(f'Found {len(ckpts)} checkpoints in {exp_dir}')
    for name, path in ckpts:
        sz = Path(path).stat().st_size
        print(f'  {name}: {path} ({sz//1024//1024}MB)')

    # Evaluate each
    results = {}
    for ckpt_name, ckpt_path in ckpts:
        eval_name = f'epochcurve_{ckpt_name}'
        print(f'\n--- {ckpt_name} ---')

        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            '--model', ckpt_path,
            '--experiment_name', eval_name,
            '--input_size', str(args.input_size[0]), str(args.input_size[1]),
            '--ocr_preproc', args.ocr_preproc,
            '--in_channels', str(args.in_channels),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        if result.returncode != 0:
            print(f'  ERROR: {result.stderr[:200]}')
            results[ckpt_name] = {'error': result.stderr[:200]}
            continue

        # Extract results from stdout
        for line in result.stdout.split('\n'):
            if 'fc_acc=' in line or 'Board' in line or 'province stress' in line.lower() or 'Fusion' in line:
                print(f'  {line.strip()}')

        # Copy eval JSONs to out dir
        src_dir = EVAL_BASE / eval_name
        if src_dir.exists():
            for jf in src_dir.glob('eval_*.json'):
                import shutil
                shutil.copy2(jf, out_dir / f'{ckpt_name}_{jf.name}')
            # Remove source
            import shutil
            shutil.rmtree(src_dir)

    # Compile summary table
    print('\n\n=== EPOCH CURVE SUMMARY ===')
    print(f'{"Checkpoint":20s} {"dump2_fc":>8s} {"dump_fc":>8s} {"stress_macro":>12s} {"fusion_exact":>12s}')
    print('-' * 60)

    curve_rows = []
    for ckpt_name, _ in ckpts:
        d2_path = out_dir / f'{ckpt_name}_eval_board_dump2.json'
        d1_path = out_dir / f'{ckpt_name}_eval_board_dump.json'
        stress_path = out_dir / f'{ckpt_name}_eval_province_stress.json'
        fusion_path = out_dir / f'{ckpt_name}_eval_fused_with_r50.json'

        d2 = json.loads(d2_path.read_text()) if d2_path.exists() else {}
        d1 = json.loads(d1_path.read_text()) if d1_path.exists() else {}
        s = json.loads(stress_path.read_text()) if stress_path.exists() else {}
        f = json.loads(fusion_path.read_text()) if fusion_path.exists() else {}

        d2_fc = d2.get('first_char_acc', '?')
        d1_fc = d1.get('first_char_acc', '?')
        stress_m = s.get('macro_first_char_acc', '?')
        fusion_ex = f.get('dump2',{}).get('fusion',{}).get('always_replace',{}).get('exact_acc','?')

        print(f'{ckpt_name:20s} {str(d2_fc):>8s} {str(d1_fc):>8s} {str(stress_m):>12s} {str(fusion_ex):>12s}')

        # Store for JSON
        curve_rows.append({
            'checkpoint': ckpt_name,
            'dump2_fc': d2_fc,
            'dump_fc': d1_fc,
            'stress_macro': stress_m,
            'fusion_exact': fusion_ex,
            'dump2_preds': d2.get('province_prediction_distribution', {}),
        })

    # Write summary JSON
    summary = {
        'exp_dir': str(exp_dir),
        'input_size': list(args.input_size),
        'ocr_preproc': args.ocr_preproc,
        'curve': curve_rows,
    }
    (out_dir / 'curve_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'\nSummary: {out_dir / "curve_summary.json"}')


if __name__ == '__main__':
    main()
