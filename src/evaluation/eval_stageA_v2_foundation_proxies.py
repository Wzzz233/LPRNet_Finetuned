#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')

PROXIES = [
    {
        'name': 'blue_real_foundation',
        'script': 'src/evaluation/eval_family_aware_blue_green_by_province.py',
        'manifest': 'manifests/curriculum_gray3_stagea_v2_foundation/proxy_blue_real_foundation.csv',
        'family': 'normal7',
    },
    {
        'name': 'green_real_foundation',
        'script': 'src/evaluation/eval_green8_metrics_only.py',
        'manifest': 'manifests/curriculum_gray3_stagea_v2_foundation/proxy_green_real_foundation.csv',
        'family': 'green8',
    },
    {
        'name': 'green_bridge',
        'script': 'src/evaluation/eval_green8_metrics_only.py',
        'manifest': 'manifests/curriculum_gray3_stagea_v2_foundation/proxy_green_bridge.csv',
        'family': 'green8',
    },
    {
        'name': 'support',
        'script': 'src/evaluation/eval_family_aware_blue_green_by_province.py',
        'manifest': 'manifests/curriculum_gray3_stagea_v2_foundation/proxy_support.csv',
        'family': None,
    },
]


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f'command failed: {cmd}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}')
    return p.stdout


def metric_from_json(data, family=None):
    if family is None:
        fams = data.get('families', {})
        return {
            fam: {
                'exact_plate_acc': fams.get(fam, {}).get('exact_plate_acc', 0.0),
                'first_char_acc': fams.get(fam, {}).get('first_char_acc', 0.0),
                'sample_count': fams.get(fam, {}).get('sample_count', 0),
            }
            for fam in sorted(fams.keys())
        }
    if 'families' in data:
        fam = data['families'].get(family, {})
        return {
            'exact_plate_acc': fam.get('exact_plate_acc', 0.0),
            'first_char_acc': fam.get('first_char_acc', 0.0),
            'sample_count': fam.get('sample_count', 0),
        }
    return {
        'exact_plate_acc': data.get('exact_plate_acc', 0.0),
        'first_char_acc': data.get('first_char_acc', 0.0),
        'sample_count': data.get('sample_count', 0),
        'province_macro_exact_acc': data.get('province_macro_exact_acc', 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'model': args.model,
        'proxies': {},
    }

    for proxy in PROXIES:
        out_json = out_dir / f"{proxy['name']}.json"
        cmd = [
            'python3', proxy['script'],
            '--model', args.model,
            '--manifest', proxy['manifest'],
            '--out_json', str(out_json),
        ]
        run(cmd, cwd=str(ROOT))
        data = json.loads(out_json.read_text(encoding='utf-8'))
        summary['proxies'][proxy['name']] = {
            'manifest': proxy['manifest'],
            'metrics': metric_from_json(data, proxy['family']),
        }

    with open(out_dir / 'graduation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = []
    md.append(f"model: {args.model}")
    for name, payload in summary['proxies'].items():
        md.append(f"\n[{name}]")
        metrics = payload['metrics']
        if isinstance(metrics, dict) and 'exact_plate_acc' in metrics:
            for k, v in metrics.items():
                md.append(f"{k}: {v}")
        else:
            for fam, fam_metrics in metrics.items():
                md.append(f"{fam}: {fam_metrics}")
    (out_dir / 'graduation_summary.txt').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
