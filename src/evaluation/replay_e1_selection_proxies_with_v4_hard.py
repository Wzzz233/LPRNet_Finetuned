#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

KEY_PROVS = ['苏', '沪', '浙', '粤', '赣', '豫']


def run_json(cmd, workdir):
    res = subprocess.run(cmd, cwd=workdir, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"cmd failed: {cmd}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    text = res.stdout.strip()
    start = text.find('{')
    if start < 0:
        raise RuntimeError(f'no json found in stdout: {text[:500]}')
    return json.loads(text[start:])


def safe(v):
    return float(v) if v is not None else 0.0


def score_from_metrics(main_real, keyprov, hard, extreme):
    return (
        0.35 * safe(main_real.get('exact_plate_acc'))
        + 0.25 * safe(main_real.get('non_major_province_exact_acc'))
        + 0.20 * safe(keyprov.get('province_macro_exact_acc'))
        + 0.10 * safe(main_real.get('major_province_exact_acc'))
        + 0.10 * safe(hard.get('exact_plate_acc'))
    )


def keyprov_tiebreak(keyprov, hard, extreme):
    pb = keyprov.get('province_breakdown', {})
    su = pb.get('苏', {})
    hu = pb.get('沪', {})
    return (
        safe(su.get('first_char_acc')),
        safe(hu.get('first_char_acc')),
        safe(keyprov.get('province_macro_first_char_acc')),
        safe(extreme.get('exact_plate_acc')),
        safe(hard.get('first_char_acc')),
    )


def eval_green8_metrics(py, workdir, model, manifest, out_json, batch_size=300, num_workers=4):
    return run_json([
        py, 'src/evaluation/eval_green8_metrics_only.py',
        '--model', str(model),
        '--manifest', str(manifest),
        '--out_json', str(out_json),
        '--batch_size', str(batch_size),
        '--num_workers', str(num_workers),
    ], workdir)


def eval_green8_by_source(py, workdir, model, manifest, out_json, batch_size=300, num_workers=4):
    return run_json([
        py, 'src/evaluation/eval_green8_metrics_by_source.py',
        '--model', str(model),
        '--manifest', str(manifest),
        '--out_json', str(out_json),
        '--batch_size', str(batch_size),
        '--num_workers', str(num_workers),
    ], workdir)


def eval_family_aware(py, workdir, model, manifest, out_json, batch_size=300, num_workers=4):
    # 这里用 eval_green8_metrics_only 代替通用 detailed eval，原因是 H36B 含 pos0 head，
    # 而当前 detailed eval 在未显式补 pos0 参数时会加载失败。metrics_only 已内置对
    # family-aware beam + pos0/adapter 权重的自适应构建，能稳定复评 green8-only guardrail。
    return run_json([
        py, 'src/evaluation/eval_green8_metrics_only.py',
        '--model', str(model),
        '--manifest', str(manifest),
        '--out_json', str(out_json),
        '--batch_size', str(batch_size),
        '--num_workers', str(num_workers),
    ], workdir)


def hard_guard(hard, extreme):
    return {
        'hard_exact_plate_acc': safe(hard.get('exact_plate_acc')),
        'hard_first_char_acc': safe(hard.get('first_char_acc')),
        'extreme_exact_plate_acc': safe(extreme.get('exact_plate_acc')),
        'extreme_first_char_acc': safe(extreme.get('first_char_acc')),
    }


def collect_checkpoints(exp_dir):
    exp_dir = Path(exp_dir)
    checkpoints = sorted(exp_dir.glob('*LPRNet__iteration_*.pth'))
    final_candidates = [
        exp_dir / 'Final_LPRNet_model.pth',
        exp_dir.parent / f'{exp_dir.name}Final_LPRNet_model.pth',
    ]
    for cand in final_candidates:
        if cand.exists() and cand not in checkpoints:
            checkpoints.append(cand)
    return checkpoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True)
    ap.add_argument('--proxy_dir', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--workdir', default='/home/wzzz/LPRNet')
    ap.add_argument('--batch_size', type=int, default=300)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    workdir = args.workdir
    py = '/home/wzzz/LPRNet/.conda/bin/python'
    checkpoints = collect_checkpoints(exp_dir)

    rows = []
    for ckpt in checkpoints:
        ckpt_name = ckpt.name
        main_real = eval_green8_metrics(py, workdir, ckpt, Path(args.proxy_dir) / 'green8_balanced_proxy.csv', exp_dir / f'{ckpt_name}.tmp.main_real.json', args.batch_size, args.num_workers)
        keyprov = eval_green8_metrics(py, workdir, ckpt, Path(args.proxy_dir) / 'green8_keyprov_proxy.csv', exp_dir / f'{ckpt_name}.tmp.keyprov.json', args.batch_size, args.num_workers)
        synth_aux = eval_green8_by_source(py, workdir, ckpt, Path(args.proxy_dir) / 'green8_synth_aux_proxy.csv', exp_dir / f'{ckpt_name}.tmp.synth.json', args.batch_size, args.num_workers)
        mixed_guard = eval_family_aware(py, workdir, ckpt, Path(args.proxy_dir) / 'green8_only_proxy.csv', exp_dir / f'{ckpt_name}.tmp.mixed_guard.json', args.batch_size, args.num_workers)
        v4_hard = eval_green8_metrics(py, workdir, ckpt, Path(args.proxy_dir) / 'green8_v4_hard_proxy.csv', exp_dir / f'{ckpt_name}.tmp.v4_hard.json', args.batch_size, args.num_workers)
        v4_extreme = eval_green8_metrics(py, workdir, ckpt, Path(args.proxy_dir) / 'green8_v4_extreme_proxy.csv', exp_dir / f'{ckpt_name}.tmp.v4_extreme.json', args.batch_size, args.num_workers)
        v4_lowocc = eval_green8_metrics(py, workdir, ckpt, Path(args.proxy_dir) / 'green8_v4_lowocc_proxy.csv', exp_dir / f'{ckpt_name}.tmp.v4_lowocc.json', args.batch_size, args.num_workers)
        row = {
            'checkpoint': ckpt_name,
            'checkpoint_path': str(ckpt),
            'main_real': main_real,
            'keyprov': keyprov,
            'synth_aux': synth_aux,
            'mixed_guard': mixed_guard,
            'v4_hard': v4_hard,
            'v4_extreme': v4_extreme,
            'v4_lowocc': v4_lowocc,
            'hard_guard': hard_guard(v4_hard, v4_extreme),
        }
        row['selection_score'] = score_from_metrics(main_real, keyprov, v4_hard, v4_extreme)
        row['tiebreak'] = keyprov_tiebreak(keyprov, v4_hard, v4_extreme)
        rows.append(row)

    ranked = sorted(rows, key=lambda r: (r['selection_score'],) + r['tiebreak'], reverse=True)
    out = {
        'exp_dir': str(exp_dir),
        'proxy_dir': str(Path(args.proxy_dir)),
        'checkpoints_evaluated': [r['checkpoint'] for r in rows],
        'ranking_policy': {
            'selection_score': '0.35*balanced_exact + 0.25*non_major_exact + 0.20*keyprov_macro_exact + 0.10*major_exact + 0.10*v4_hard_exact',
            'tiebreak': ['苏 first_char', '沪 first_char', 'keyprov macro_first', 'v4_extreme exact', 'v4_hard first_char'],
            'guardrail': 'mixed_guard retained for reference; v4 hard/extreme exposed separately',
        },
        'ranked': ranked,
        'best_checkpoint': ranked[0]['checkpoint'] if ranked else None,
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
