#!/usr/bin/env python3
import json, re, subprocess, sys, time
from pathlib import Path

ROOT=Path('/home/wzzz/LPRNet')
EXP=ROOT/'experiments/curriculum_gray3_stageA_v3_realprimary_A1B_supportmid_short'
LOG=EXP/'train.log'
FINAL=EXP/'Final_LPRNet_model.pth'
OUT=EXP/'proxy_eval_v3'
STATUS_JSON=ROOT/'reports/ACTIVE_TRAINING_STATUS.json'
STATUS_MD=ROOT/'reports/ACTIVE_TRAINING_STATUS.md'
REPORT=ROOT/'reports/GREEN_STAGEA_V3_REALPRIMARY_A1B_RESULT_REPORT.md'
BASELINE=ROOT/'experiments/curriculum_gray3_stageA_v3_realprimary_A0/proxy_eval_v3/ranking.json'
PROCESS_KEY='curriculum_gray3_stageA_v3_realprimary_A1B_supportmid_short'

def sh(cmd, check=True):
    p=subprocess.run(cmd, cwd=str(ROOT), shell=True, text=True, capture_output=True)
    if check and p.returncode!=0:
        raise RuntimeError(f'cmd failed {cmd}\nstdout={p.stdout}\nstderr={p.stderr}')
    return p

def update_status():
    cmd=(f"python3 src/utils/update_training_status.py "
         f"--experiment-dir {EXP} --log {LOG} --process-keyword {PROCESS_KEY} "
         f"--final-weight {FINAL} --eval-json {OUT/'ranking.json'} --baseline-json {BASELINE} "
         f"--out-json {STATUS_JSON} --out-md {STATUS_MD}")
    sh(cmd, check=False)

def running():
    p=sh(f"pgrep -af '{PROCESS_KEY}' | grep -v a1b_completion_check.py | grep -v update_training_status.py || true", check=False)
    return bool(p.stdout.strip())

def metric(r, name, key='exact_plate_acc'):
    return r['metrics'][name][key]

def pct(x): return f'{x*100:.2f}%'

def main():
    update_status()
    text=LOG.read_text(encoding='utf-8', errors='replace') if LOG.exists() else ''
    done='[Training Done]' in text
    if running() or not done or not FINAL.exists():
        # Not done. Refresh status and report progress only.
        tail='\n'.join(text.splitlines()[-30:])
        REPORT.write_text('# A1B Training Progress\n\n训练尚未满足最终评估条件。\n\n'
                          f'- process_running: {running()}\n- training_done_marker: {done}\n- final_weight_exists: {FINAL.exists()}\n\n'
                          '## Log tail\n```\n'+tail+'\n```\n', encoding='utf-8')
        print(REPORT.read_text(encoding='utf-8'))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'ranking.json').exists():
        sh(f"python3 scripts/curriculum_gray3/eval_stageA_v3_graduation.py --exp_dir {EXP} --manifest_dir manifests/curriculum_gray3_stagea_v3_realprimary --out_dir {OUT}")
    update_status()
    rank=json.loads((OUT/'ranking.json').read_text(encoding='utf-8'))
    a0=json.loads(BASELINE.read_text(encoding='utf-8'))
    best=rank[0]; base=a0[0]
    real_avg=best['real_avg']; gap=best['family_gap']
    b1=metric(best,'blue_ccpd2019_real'); b2=metric(best,'blue_crpd_real'); g=metric(best,'green_ccpd2020_real')
    non=metric(best,'green_nonanhui_template_synth'); bridge=metric(best,'green_bridge_exactquad')
    non_fc=metric(best,'green_nonanhui_template_synth','first_char_acc')
    pass_gate = (real_avg >= 0.6944 and non > 0.3547 and bridge >= 0.5812 and gap <= 0.03)
    lines=[]
    lines.append('# GREEN StageA v3 Real-Primary A1B Result Report')
    lines.append('')
    lines.append('## 状态')
    lines.append(f'- training_done: {done}')
    lines.append(f'- final_weight: `{FINAL}`')
    lines.append(f'- graduation_ranking: `{OUT/"ranking.json"}`')
    lines.append(f'- recommended_checkpoint: `{(OUT/"recommended_checkpoint.txt").read_text(encoding="utf-8").strip()}`')
    lines.append('')
    lines.append('## A1B best graduation')
    lines.append(f'- label: {best["label"]}')
    lines.append(f'- score: {best["score"]:.4f}')
    lines.append(f'- real_avg: {pct(real_avg)}')
    lines.append(f'- family_gap: {gap*100:.2f}pp')
    lines.append(f'- blue_ccpd2019_real exact: {pct(b1)}')
    lines.append(f'- blue_crpd_real exact: {pct(b2)}')
    lines.append(f'- green_ccpd2020_real exact: {pct(g)}')
    lines.append(f'- green_nonanhui_template_synth exact: {pct(non)}')
    lines.append(f'- green_nonanhui_template_synth first_char: {pct(non_fc)}')
    lines.append(f'- green_bridge_exactquad exact: {pct(bridge)}')
    lines.append('')
    lines.append('## A0 baseline best')
    lines.append(f'- label: {base["label"]}')
    lines.append(f'- real_avg: {pct(base["real_avg"])}')
    lines.append(f'- family_gap: {base["family_gap"]*100:.2f}pp')
    lines.append(f'- blue_ccpd2019_real exact: {pct(metric(base,"blue_ccpd2019_real"))}')
    lines.append(f'- blue_crpd_real exact: {pct(metric(base,"blue_crpd_real"))}')
    lines.append(f'- green_ccpd2020_real exact: {pct(metric(base,"green_ccpd2020_real"))}')
    lines.append(f'- green_nonanhui_template_synth exact: {pct(metric(base,"green_nonanhui_template_synth"))}')
    lines.append(f'- green_bridge_exactquad exact: {pct(metric(base,"green_bridge_exactquad"))}')
    lines.append('')
    lines.append('## Gate')
    lines.append(f'- PASS: {pass_gate}')
    lines.append('- 判据：real_avg >= 69.44%，non-皖 exact > 35.47%，bridge >= 58.12%，family_gap <= 3pp。')
    lines.append('')
    if pass_gate:
        lines.append('结论：A1B 通过 StageA light-support correction gate，可作为进入后续 StageB 设计的候选 mother。')
    else:
        lines.append('结论：A1B 未通过 gate，不能进入 StageB；继续保留 A0 作为当前唯一合格 mother。')
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(REPORT.read_text(encoding='utf-8'))
    return 0
if __name__=='__main__': sys.exit(main())
