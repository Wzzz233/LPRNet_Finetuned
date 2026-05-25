#!/usr/bin/env python3
import argparse
import csv
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


def score_from_metrics(main_real, keyprov, synth_aux, mixed_guard):
    return (
        0.40 * safe(main_real.get('exact_plate_acc'))
        + 0.30 * safe(main_real.get('non_major_province_exact_acc'))
        + 0.20 * safe(keyprov.get('province_macro_exact_acc'))
        + 0.10 * safe(main_real.get('major_province_exact_acc'))
    )


def keyprov_tiebreak(keyprov):
    pb = keyprov.get('province_breakdown', {})
    su = pb.get('苏', {})
    hu = pb.get('沪', {})
    return (
        safe(su.get('first_char_acc')),
        safe(hu.get('first_char_acc')),
        safe(keyprov.get('province_macro_first_char_acc')),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True)
    ap.add_argument('--proxy_dir', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--workdir', default='/home/wzzz/LPRNet')
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    workdir = args.workdir
    py = '/home/wzzz/LPRNet/.conda/bin/python'
    checkpoints = []
    for p in sorted(exp_dir.glob('LPRNet__iteration_*.pth')):
        checkpoints.append(p)
    final_p = exp_dir / 'Final_LPRNet_model.pth'
    if final_p.exists():
        checkpoints.append(final_p)

    rows = []
    for ckpt in checkpoints:
        ckpt_name = ckpt.name
        # main real proxy: balanced proxy + green8 only metrics
        main_real = run_json([
            py, 'src/evaluation/eval_green8_metrics_only.py',
            '--model', str(ckpt),
            '--manifest', str(Path(args.proxy_dir) / 'green8_balanced_proxy.csv'),
            '--out_json', str(exp_dir / f'{ckpt_name}.tmp.main_real.json'),
            '--batch_size', '300', '--num_workers', '4'
        ], workdir)
        keyprov = run_json([
            py, 'src/evaluation/eval_green8_metrics_only.py',
            '--model', str(ckpt),
            '--manifest', str(Path(args.proxy_dir) / 'green8_keyprov_proxy.csv'),
            '--out_json', str(exp_dir / f'{ckpt_name}.tmp.keyprov.json'),
            '--batch_size', '300', '--num_workers', '4'
        ], workdir)
        synth_aux = run_json([
            py, 'src/evaluation/eval_green8_metrics_by_source.py',
            '--model', str(ckpt),
            '--manifest', str(Path(args.proxy_dir) / 'green8_synth_aux_proxy.csv'),
            '--out_json', str(exp_dir / f'{ckpt_name}.tmp.synth.json'),
            '--batch_size', '300', '--num_workers', '4'
        ], workdir)
        mixed_guard = run_json([
            py, 'src/evaluation/eval_lpr_detailed.py',
            '--cuda', 'true',
            '--data_mode', 'manifest',
            '--test_img_dirs', str(Path(args.proxy_dir) / 'green8_only_proxy.csv'),
            '--txt_file', str(Path(args.proxy_dir) / 'green8_only_proxy.csv'),
            '--ocr_channel_order', 'bgr',
            '--ocr_crop_mode', 'obb_warp',
            '--ocr_resize_mode', 'letterbox',
            '--ocr_resize_kernel', 'nn',
            '--ocr_preproc', 'none',
            '--ocr_min_occ_ratio', '0.90',
            '--ocr_quad_pad_ratio', '0.0',
            '--head_mode', 'multihead',
            '--enhanced_green_head', 'expD',
            '--pretrained_model', str(ckpt),
            '--test_batch_size', '300',
            '--decode_mode', 'family_aware_beam',
            '--beam_size', '20',
            '--beam_topk', '12',
            '--out_json', str(exp_dir / f'{ckpt_name}.tmp.mixed_guard.json')
        ], workdir)
        row = {
            'checkpoint': ckpt_name,
            'main_real': main_real,
            'keyprov': keyprov,
            'synth_aux': synth_aux,
            'mixed_guard': mixed_guard,
        }
        row['selection_score'] = score_from_metrics(main_real, keyprov, synth_aux, mixed_guard)
        row['tiebreak'] = keyprov_tiebreak(keyprov)
        rows.append(row)

    ranked = sorted(rows, key=lambda r: (r['selection_score'],) + r['tiebreak'], reverse=True)
    out = {
        'exp_dir': str(exp_dir),
        'proxy_dir': str(Path(args.proxy_dir)),
        'checkpoints_evaluated': [r['checkpoint'] for r in rows],
        'ranked': ranked,
        'best_checkpoint': ranked[0]['checkpoint'] if ranked else None,
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
