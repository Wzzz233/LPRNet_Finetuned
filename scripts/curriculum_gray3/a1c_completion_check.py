#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
EXP = ROOT / 'experiments/curriculum_gray3_stageA_v3_realprimary_A1C_softfreeze_supportmid'
LOG = EXP / 'train.log'
FINAL = EXP / 'Final_LPRNet_model.pth'
OUT = EXP / 'proxy_eval_v3'
REPORT = ROOT / 'reports/GREEN_STAGEA_V3_REALPRIMARY_A1C_SOFTFREEZE_REPORT.md'
STATUS_JSON = ROOT / 'reports/ACTIVE_TRAINING_STATUS.json'
STATUS_MD = ROOT / 'reports/ACTIVE_TRAINING_STATUS.md'
PROCESS_KEY = 'curriculum_gray3_stageA_v3_realprimary_A1C_softfreeze_supportmid'
A0_RANKING = ROOT / 'experiments/curriculum_gray3_stageA_v3_realprimary_A0/proxy_eval_v3/ranking.json'
A1B_RANKING = ROOT / 'experiments/curriculum_gray3_stageA_v3_realprimary_A1B_supportmid_short/proxy_eval_v3/ranking.json'

GATES = {
    'real_avg_min': 0.6944,
    'green_nonanhui_exact_min_exclusive': 0.3547,
    'green_bridge_exact_min': 0.5812,
    'family_gap_max': 0.03,
}

def shell(cmd):
    return subprocess.run(cmd, cwd=str(ROOT), shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def process_running():
    p = shell(f"pgrep -af '{PROCESS_KEY}|train_LPRNet.py.*A1C_softfreeze_supportmid' || true")
    lines = [x for x in p.stdout.splitlines() if 'pgrep -af' not in x and x.strip()]
    return lines

def write_status(state, extra=None):
    extra = extra or {}
    data = {
        'experiment': 'A1C_softfreeze_supportmid',
        'state': state,
        'exp_dir': str(EXP),
        'train_log': str(LOG),
        'final_checkpoint': str(FINAL),
        'eval_out_dir': str(OUT),
        'report': str(REPORT),
        'gates': GATES,
        **extra,
    }
    STATUS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    STATUS_MD.write_text('\n'.join([
        '# ACTIVE TRAINING STATUS',
        '',
        f'- experiment: A1C_softfreeze_supportmid',
        f'- state: {state}',
        f'- exp_dir: {EXP}',
        f'- train_log: {LOG}',
        f'- final_checkpoint: {FINAL}',
        f'- eval_out_dir: {OUT}',
        f'- report: {REPORT}',
        f'- gates: real_avg>=69.44%, nonanhui_exact>35.47%, bridge>=58.12%, family_gap<=3pp',
    ]), encoding='utf-8')

def pct(x):
    if x is None:
        return 'NA'
    return f'{x*100:.2f}%'

def metric(row, name, field='exact_plate_acc'):
    return row['metrics'][name][field]

def build_report(ranking):
    best = ranking[0]
    a0 = json.loads(A0_RANKING.read_text(encoding='utf-8'))[0]
    a1b = json.loads(A1B_RANKING.read_text(encoding='utf-8'))[0]
    final_like = next((r for r in ranking if r['label'] == 'Final_LPRNet_model'), None)
    passed = (
        best['real_avg'] >= GATES['real_avg_min'] and
        metric(best, 'green_nonanhui_template_synth') > GATES['green_nonanhui_exact_min_exclusive'] and
        metric(best, 'green_bridge_exactquad') >= GATES['green_bridge_exact_min'] and
        best['family_gap'] <= GATES['family_gap_max']
    )
    lines = []
    lines.append('# GREEN StageA v3 Real-Primary A1C Soft-Freeze Result Report')
    lines.append('')
    lines.append('## Experiment')
    lines.append('- name: curriculum_gray3_stageA_v3_realprimary_A1C_softfreeze_supportmid')
    lines.append('- init: A0 recommended checkpoint')
    lines.append('- data: reuse A1B train/val manifests')
    lines.append('- variable: freeze_backbone=True, trainable_backbone_prefixes=backbone.18,backbone.19,backbone.20')
    lines.append('- fixed: Gray3=1.0, obb_warp/letterbox/nn/bgr, multihead normal7+green8')
    lines.append('')
    lines.append('## Gates')
    lines.append('- real_avg >= 69.44%')
    lines.append('- green_nonanhui_template_synth exact > 35.47%')
    lines.append('- green_bridge_exactquad exact >= 58.12%')
    lines.append('- family_gap <= 3.00pp')
    lines.append('')
    lines.append('## Best checkpoint by graduation ranking')
    lines.append(f"- label: {best['label']}")
    lines.append(f"- checkpoint: {best['checkpoint']}")
    lines.append(f"- score: {best['score']:.6f}")
    lines.append(f"- real_avg: {pct(best['real_avg'])}")
    lines.append(f"- family_gap: {best['family_gap']*100:.2f}pp")
    for name in ['blue_ccpd2019_real','blue_crpd_real','green_ccpd2020_real','green_nonanhui_template_synth','green_bridge_exactquad']:
        v = metric(best, name)
        b0 = metric(a0, name)
        b1 = metric(a1b, name)
        lines.append(f"- {name}: {pct(v)} (vs A0 {(v-b0)*100:+.2f}pp, vs A1B-best {(v-b1)*100:+.2f}pp)")
    lines.append(f"- green_nonanhui_template_synth first_char: {pct(metric(best, 'green_nonanhui_template_synth', 'first_char_acc'))} (A0 {pct(metric(a0, 'green_nonanhui_template_synth', 'first_char_acc'))})")
    lines.append('')
    if final_like:
        lines.append('## Final checkpoint')
        lines.append(f"- real_avg: {pct(final_like['real_avg'])}")
        lines.append(f"- family_gap: {final_like['family_gap']*100:.2f}pp")
        lines.append(f"- green_nonanhui_template_synth exact: {pct(metric(final_like, 'green_nonanhui_template_synth'))}")
        lines.append(f"- green_bridge_exactquad exact: {pct(metric(final_like, 'green_bridge_exactquad'))}")
        lines.append('')
    lines.append('## Decision')
    if passed:
        lines.append('结论：A1C-softfreeze 通过 gate，可作为 A0 后继 mother，并允许设计后续 StageB difficulty fine-tuning。')
    else:
        lines.append('结论：A1C-softfreeze 未通过 gate，不能进入 StageB；继续保留 A0 作为当前唯一合格 mother。')
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return passed

def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        write_status('not_started', {'reason': 'train.log missing'})
        print('A1C not started: train.log missing')
        return 1
    text = LOG.read_text(errors='ignore')
    running = process_running()
    if (not FINAL.exists()) or ('[Training Done]' not in text) or running:
        tail = '\n'.join(text.splitlines()[-20:]) if text else ''
        write_status('training_or_incomplete', {'running_processes': running, 'log_tail': tail})
        REPORT.write_text('# A1C Training Progress\n\n训练尚未满足最终评估条件。未确认 Final checkpoint + [Training Done] + 进程退出三条件，禁止评估。\n\n```\n' + tail + '\n```\n', encoding='utf-8')
        print('A1C training not complete; status refreshed.')
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    ranking = OUT / 'ranking.json'
    if not ranking.exists():
        cmd = f"python3 scripts/curriculum_gray3/eval_stageA_v3_graduation.py --exp_dir {EXP} --manifest_dir manifests/curriculum_gray3_stagea_v3_realprimary --out_dir {OUT}"
        p = shell(cmd)
        (OUT / 'eval_stageA_v3_graduation.log').write_text(p.stdout, encoding='utf-8')
        if p.returncode != 0:
            write_status('eval_failed', {'eval_returncode': p.returncode, 'eval_log': str(OUT / 'eval_stageA_v3_graduation.log')})
            print(p.stdout)
            return p.returncode
    data = json.loads(ranking.read_text(encoding='utf-8'))
    passed = build_report(data)
    write_status('done_pass' if passed else 'done_fail', {'ranking': str(ranking), 'passed': passed})
    print(REPORT.read_text(encoding='utf-8'))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
