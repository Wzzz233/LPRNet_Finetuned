#!/usr/bin/env python3
import argparse, json, re, subprocess, datetime
from pathlib import Path

def is_proc_running(keyword: str) -> bool:
    if not keyword:
        return False
    p = subprocess.run(['bash','-lc',f"pgrep -af {keyword!r} | grep -v update_training_status.py | grep -v a1b_completion_check.py || true"], text=True, capture_output=True)
    return bool(p.stdout.strip())

def tail_lines(path: Path, n=30):
    if not path.exists(): return []
    lines=path.read_text(encoding='utf-8', errors='replace').splitlines()
    return lines[-n:]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment-dir', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--process-keyword', required=True)
    ap.add_argument('--final-weight', required=True)
    ap.add_argument('--eval-json', default='')
    ap.add_argument('--baseline-json', default='')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', required=True)
    args=ap.parse_args()
    log=Path(args.log); final=Path(args.final_weight)
    text=log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
    epoch_iter=''
    m=list(re.finditer(r'Epoch:(\d+) \|\| epochiter: (\d+)/(\d+)\|\| Totel iter (\d+)', text))
    if m:
        x=m[-1]; epoch_iter=f"epoch {x.group(1)} iter {x.group(2)}/{x.group(3)} total_iter {x.group(4)}"
    summaries=re.findall(r'\[Epoch Summary\].*', text)
    sels=re.findall(r'\[SelectionProxy\].*|\[SelectionEval\].*', text)
    done='[Training Done]' in text
    running=is_proc_running(args.process_keyword)
    state='running' if running else ('finished' if done and final.exists() else 'stopped_or_finishing')
    data={
      'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
      'experiment_dir': args.experiment_dir,
      'log_path': args.log,
      'state': state,
      'process_running': running,
      'training_done_marker': done,
      'final_weight_exists': final.exists(),
      'final_weight_path': str(final),
      'eval_json_exists': Path(args.eval_json).exists() if args.eval_json else False,
      'eval_json_path': args.eval_json,
      'baseline_json_path': args.baseline_json,
      'latest_epoch_iter': epoch_iter,
      'latest_epoch_summary': summaries[-1] if summaries else '',
      'latest_selection_eval': sels[-1] if sels else '',
      'log_tail': tail_lines(log, 40),
    }
    Path(args.out_json).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    md=[]
    md.append('# Active Training Status')
    md.append('')
    for k in ['updated_at','state','experiment_dir','log_path','latest_epoch_iter','latest_epoch_summary','latest_selection_eval','process_running','training_done_marker','final_weight_exists','final_weight_path','eval_json_exists','eval_json_path']:
        md.append(f'- {k}: {data[k]}')
    md.append('')
    md.append('## Log tail')
    md.append('```')
    md.extend(data['log_tail'])
    md.append('```')
    Path(args.out_md).write_text('\n'.join(md)+'\n',encoding='utf-8')
if __name__=='__main__': main()
