#!/usr/bin/env python3
"""
StageA graduation evaluation
- aggregate foundation metrics from blue/green/mixed proxy eval jsons
- handle both flat schema and family-aware schema
- emit concise markdown report and machine-readable json
"""

import json
from pathlib import Path


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def pct(x):
    return f"{100.0 * float(x):.2f}%"


def fam_metric(blob, family, key):
    fams = blob.get('families') or {}
    if family in fams:
        return fams[family].get(key, 0.0)
    return blob.get(key, 0.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True)
    ap.add_argument('--blue_json', required=True)
    ap.add_argument('--green_json', required=True)
    ap.add_argument('--mixed_json', required=True)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    blue = load_json(args.blue_json)
    green = load_json(args.green_json)
    mixed = load_json(args.mixed_json)

    metrics = {
        'blue_exact_plate_acc': fam_metric(blue, 'normal7', 'exact_plate_acc'),
        'blue_first_char_acc': fam_metric(blue, 'normal7', 'first_char_acc'),
        'green_exact_plate_acc': fam_metric(green, 'green8', 'exact_plate_acc'),
        'green_first_char_acc': fam_metric(green, 'green8', 'first_char_acc'),
        'green_province_macro_exact_acc': green.get('province_macro_exact_acc', 0.0),
        'mixed_exact_plate_acc': mixed.get('exact_plate_acc', 0.0),
        'mixed_normal7_exact_acc': fam_metric(mixed, 'normal7', 'exact_plate_acc'),
        'mixed_green8_exact_acc': fam_metric(mixed, 'green8', 'exact_plate_acc'),
        'pass_gate': False,
    }

    if metrics['mixed_exact_plate_acc'] == 0.0:
        n7 = mixed.get('families', {}).get('normal7', {}).get('sample_count', 0)
        g8 = mixed.get('families', {}).get('green8', {}).get('sample_count', 0)
        total = n7 + g8
        if total > 0:
            metrics['mixed_exact_plate_acc'] = (
                metrics['mixed_normal7_exact_acc'] * n7 + metrics['mixed_green8_exact_acc'] * g8
            ) / total

    metrics['pass_gate'] = (
        metrics['blue_exact_plate_acc'] >= 0.35 and
        metrics['green_exact_plate_acc'] >= 0.70 and
        metrics['mixed_normal7_exact_acc'] >= 0.30 and
        metrics['mixed_green8_exact_acc'] >= 0.70
    )

    out_json = exp_dir / 'stageA_graduation_metrics.json'
    out_md = exp_dir / 'STAGEA_GRADUATION_REPORT.md'
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = []
    lines.append('# STAGEA Graduation Report')
    lines.append('')
    lines.append('## Core metrics')
    lines.append('')
    lines.append(f"- blue foundation exact: {pct(metrics['blue_exact_plate_acc'])}")
    lines.append(f"- blue foundation first-char: {pct(metrics['blue_first_char_acc'])}")
    lines.append(f"- green foundation exact: {pct(metrics['green_exact_plate_acc'])}")
    lines.append(f"- green foundation first-char: {pct(metrics['green_first_char_acc'])}")
    lines.append(f"- green province-macro exact: {pct(metrics['green_province_macro_exact_acc'])}")
    lines.append(f"- mixed normal7 exact: {pct(metrics['mixed_normal7_exact_acc'])}")
    lines.append(f"- mixed green8 exact: {pct(metrics['mixed_green8_exact_acc'])}")
    lines.append(f"- mixed overall exact: {pct(metrics['mixed_exact_plate_acc'])}")
    lines.append('')
    lines.append('## Graduation decision')
    lines.append('')
    lines.append(f"- PASS_GATE: {'PASS' if metrics['pass_gate'] else 'FAIL'}")
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    if metrics['pass_gate']:
        lines.append('- This checkpoint is good enough to be considered a StageA shared base candidate for StageB soft-freeze adaptation.')
    else:
        lines.append('- This checkpoint is NOT yet a qualified shared base. Continue StageA redesign before entering StageB.')
    out_md.write_text('\n'.join(lines), encoding='utf-8')
    print(out_json)
    print(out_md)


if __name__ == '__main__':
    main()
