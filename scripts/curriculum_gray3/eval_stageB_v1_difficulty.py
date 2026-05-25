#!/usr/bin/env python3
"""Evaluate StageB difficulty-finetuning checkpoints with A1D guardrails plus hard/extreme probes."""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
PROXIES = [
    ('blue_ccpd2019_real', 'src/evaluation/eval_family_aware_blue_green_by_province.py', 'normal7'),
    ('blue_crpd_real', 'src/evaluation/eval_family_aware_blue_green_by_province.py', 'normal7'),
    ('green_ccpd2020_real', 'src/evaluation/eval_green8_metrics_only.py', 'green8'),
    ('green_nonanhui_template_synth', 'src/evaluation/eval_green8_metrics_only.py', 'green8'),
    ('green_bridge_exactquad', 'src/evaluation/eval_green8_metrics_only.py', 'green8'),
    ('green_edgefit_hard', 'src/evaluation/eval_green8_metrics_only.py', 'green8'),
    ('green_edgefit_extreme', 'src/evaluation/eval_green8_metrics_only.py', 'green8'),
    ('support_cblprd', 'src/evaluation/eval_family_aware_blue_green_by_province.py', None),
]

GATES = {
    # A1D foundation guardrails.  Do not let difficulty gains destroy real foundation.
    'real_avg_min': 0.6944,
    'family_gap_max': 0.0300,
    'green_nonanhui_exact_min': 0.7653333333333333 - 0.03,
    'green_bridge_exact_min': 0.68375 - 0.03,
    # StageB new objective: must improve these over A1D baseline measured by this evaluator.
    # Final pass should use relative comparisons to a fresh A1D replay; these absolute floors prevent nonsense passes.
    'hard_exact_floor': 0.10,
    'extreme_exact_floor': 0.02,
}


def run(cmd):
    p = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError('command failed: {}\nstdout:\n{}\nstderr:\n{}'.format(' '.join(cmd), p.stdout, p.stderr))


def metric_from_json(data, family):
    if family is None:
        fams = data.get('families', {})
        return {fam: {
            'sample_count': fams.get(fam, {}).get('sample_count', 0),
            'exact_plate_acc': fams.get(fam, {}).get('exact_plate_acc', 0.0),
            'first_char_acc': fams.get(fam, {}).get('first_char_acc', 0.0),
        } for fam in sorted(fams)}
    if 'families' in data:
        d = data.get('families', {}).get(family, {})
    else:
        d = data
    return {
        'sample_count': d.get('sample_count', 0),
        'exact_plate_acc': d.get('exact_plate_acc', 0.0),
        'first_char_acc': d.get('first_char_acc', 0.0),
        'province_macro_exact_acc': d.get('province_macro_exact_acc', 0.0),
        'province_macro_first_char_acc': d.get('province_macro_first_char_acc', 0.0),
        'non_major_province_exact_acc': d.get('non_major_province_exact_acc', 0.0),
        'major_province_exact_acc': d.get('major_province_exact_acc', 0.0),
    }


def discover_checkpoints(exp_dir: Path):
    ckpts = []
    for name in ['Final_LPRNet_model.pth', 'best_LPRNet_model.pth', 'last_LPRNet_model.pth']:
        p = exp_dir / name
        if p.exists():
            ckpts.append(p)
    ckpts.extend(sorted(exp_dir.glob('LPRNet__iteration_*.pth')))
    ckpts.extend(sorted(exp_dir.parent.glob(f'{exp_dir.name}LPRNet__iteration_*.pth')))
    out=[]; seen=set()
    for p in ckpts:
        rp=str(p.resolve())
        if rp not in seen:
            seen.add(rp); out.append(p)
    return out


def checkpoint_label(p: Path):
    m = re.search(r'iteration_(\d+)', p.name)
    if m:
        return f'iter_{int(m.group(1)):06d}'
    return p.name.replace('.pth','')


def core_stats(metrics):
    b1 = metrics['blue_ccpd2019_real']['exact_plate_acc']
    b2 = metrics['blue_crpd_real']['exact_plate_acc']
    g = metrics['green_ccpd2020_real']['exact_plate_acc']
    real_avg = (b1 + b2 + g) / 3.0
    family_gap = abs(((b1 + b2) / 2.0) - g)
    return real_avg, family_gap


def score(metrics):
    real_avg, family_gap = core_stats(metrics)
    non = metrics['green_nonanhui_template_synth']['exact_plate_acc']
    bridge = metrics['green_bridge_exactquad']['exact_plate_acc']
    hard = metrics['green_edgefit_hard']['exact_plate_acc']
    extreme = metrics['green_edgefit_extreme']['exact_plate_acc']
    # hard/extreme matter, but foundation penalties dominate.
    return real_avg - 0.25 * family_gap + 0.05 * non + 0.05 * bridge + 0.08 * hard + 0.04 * extreme


def pass_abs_gate(metrics):
    real_avg, family_gap = core_stats(metrics)
    return (
        real_avg >= GATES['real_avg_min'] and
        family_gap <= GATES['family_gap_max'] and
        metrics['green_nonanhui_template_synth']['exact_plate_acc'] >= GATES['green_nonanhui_exact_min'] and
        metrics['green_bridge_exactquad']['exact_plate_acc'] >= GATES['green_bridge_exact_min'] and
        metrics['green_edgefit_hard']['exact_plate_acc'] >= GATES['hard_exact_floor'] and
        metrics['green_edgefit_extreme']['exact_plate_acc'] >= GATES['extreme_exact_floor']
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', default='')
    ap.add_argument('--manifest_dir', default='manifests/curriculum_gray3_stageb_v1_difficulty')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model', default='')
    args = ap.parse_args()
    exp_dir = ROOT / args.exp_dir if not args.exp_dir.startswith('/') else Path(args.exp_dir)
    out_dir = ROOT / args.out_dir if not args.out_dir.startswith('/') else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = [ROOT / args.model if args.model and not args.model.startswith('/') else Path(args.model)] if args.model else discover_checkpoints(exp_dir)
    if not ckpts:
        raise RuntimeError(f'no checkpoints found in {exp_dir}')
    ranking=[]
    for ckpt in ckpts:
        label=checkpoint_label(ckpt)
        ckpt_dir=out_dir / label
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics={}
        for name, script, family in PROXIES:
            out_json=ckpt_dir / f'{name}.json'
            manifest=f'{args.manifest_dir}/proxy_{name}.csv'
            cmd=['python3', script, '--model', str(ckpt), '--manifest', manifest, '--out_json', str(out_json),
                 '--ocr_preproc', 'gray3']
            run(cmd)
            data=json.loads(out_json.read_text(encoding='utf-8'))
            metrics[name]=metric_from_json(data, family)
        real_avg, family_gap = core_stats(metrics)
        ranking.append({
            'checkpoint': str(ckpt),
            'label': label,
            'metrics': metrics,
            'real_avg': real_avg,
            'family_gap': family_gap,
            'score': score(metrics),
            'pass_abs_gate': pass_abs_gate(metrics),
        })
    ranking.sort(key=lambda x: (x['pass_abs_gate'], x['score']), reverse=True)
    (out_dir/'ranking.json').write_text(json.dumps(ranking, ensure_ascii=False, indent=2), encoding='utf-8')
    best=ranking[0]
    (out_dir/'recommended_checkpoint.txt').write_text(best['checkpoint'] + '\n', encoding='utf-8')
    lines=['# StageB v1 Difficulty Ranking','',f'best: `{best["checkpoint"]}`','']
    lines.append('| rank | label | pass_abs | score | real_avg | family_gap | blue_ccpd | blue_crpd | green_ccpd | nonanhui | bridge | hard | extreme |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for i,r in enumerate(ranking,1):
        m=r['metrics']
        lines.append('| {} | {} | {} | {:.4f} | {:.2f}% | {:.2f}pp | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% |'.format(
            i, r['label'], 'Y' if r['pass_abs_gate'] else 'N', r['score'], r['real_avg']*100, r['family_gap']*100,
            m['blue_ccpd2019_real']['exact_plate_acc']*100,
            m['blue_crpd_real']['exact_plate_acc']*100,
            m['green_ccpd2020_real']['exact_plate_acc']*100,
            m['green_nonanhui_template_synth']['exact_plate_acc']*100,
            m['green_bridge_exactquad']['exact_plate_acc']*100,
            m['green_edgefit_hard']['exact_plate_acc']*100,
            m['green_edgefit_extreme']['exact_plate_acc']*100,
        ))
    (out_dir/'ranking.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
