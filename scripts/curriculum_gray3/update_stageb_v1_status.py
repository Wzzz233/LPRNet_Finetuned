#!/usr/bin/env python3
import json, re, subprocess
from datetime import datetime
from pathlib import Path
ROOT=Path('/home/wzzz/LPRNet')
EXP=ROOT/'experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservative'
LOG=EXP/'train.log'
FINAL=EXP/'Final_LPRNet_model.pth'
EVAL=EXP/'proxy_eval_stageB_v1/ranking.json'
OUTJ=ROOT/'reports/ACTIVE_TRAINING_STATUS.json'
OUTM=ROOT/'reports/ACTIVE_TRAINING_STATUS.md'
text=LOG.read_text(encoding='utf-8', errors='ignore') if LOG.exists() else ''
proc=subprocess.run("pgrep -af 'run_stageB_v1_B1A|train_LPRNet.py.*train_B1A|eval_stageB_v1_difficulty' || true",shell=True,text=True,capture_output=True)
procs=[ln for ln in proc.stdout.splitlines() if 'pgrep -af' not in ln and 'update_stageb_v1_status.py' not in ln]
latest=''
for line in text.splitlines():
    if 'Epoch:' in line or '[Training Done]' in line or 'SelectionEval' in line:
        latest=line
state='running' if procs else ('finished' if '[Training Done]' in text and FINAL.exists() else 'unknown')
summary={
 'updated_at': datetime.now().isoformat(timespec='seconds'),
 'experiment_dir': str(EXP), 'log_path': str(LOG), 'state': state,
 'process_running': bool(procs), 'processes': procs[:8],
 'final_weight_exists': FINAL.exists(), 'final_weight_path': str(FINAL),
 'training_done_marker': '[Training Done]' in text,
 'eval_json_exists': EVAL.exists(), 'eval_json_path': str(EVAL),
 'latest_training_line': latest,
 'log_tail': '\n'.join(text.splitlines()[-30:]),
}
if EVAL.exists():
    try:
        data=json.loads(EVAL.read_text(encoding='utf-8'))
        best=data[0] if isinstance(data,list) and data else data.get('ranked',[{}])[0]
        summary['eval_best']=best
    except Exception as e:
        summary['eval_parse_error']=str(e)
OUTJ.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=[f"# Active Training Status",'',f"updated_at: {summary['updated_at']}",f"state: {state}",f"experiment: {EXP}",f"log: {LOG}",f"final_weight_exists: {FINAL.exists()}",f"training_done_marker: {summary['training_done_marker']}",f"eval_json_exists: {EVAL.exists()}",'', '## latest', latest or '(none)', '', '## processes']
lines += procs[:8] or ['(none)']
OUTM.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2)[:2000])
