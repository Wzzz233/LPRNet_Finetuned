#!/usr/bin/env bash
set -euo pipefail
cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

OUT_DIR="/home/wzzz/LPRNet/experiments/stageB1A_B_dual_proxy_eval_20260425"
NEW_MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3"
OLD_MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_difficulty"
mkdir -p "$OUT_DIR"

# Ensure B new proxy dir has the full proxy set for repeatable evals.
for f in proxy_blue_ccpd2019_real.csv proxy_blue_crpd_real.csv proxy_green_ccpd2020_real.csv proxy_green_nonanhui_template_synth.csv proxy_green_bridge_exactquad.csv proxy_green_edgefit_hard.csv proxy_support_cblprd.csv; do
  if [[ ! -f "$NEW_MANIFEST_DIR/$f" ]]; then
    cp "$OLD_MANIFEST_DIR/$f" "$NEW_MANIFEST_DIR/$f"
  fi
done

BASE_B1A_BEST="/home/wzzz/LPRNet/experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservativeLPRNet__iteration_2000.pth"
BASE_B1A_FINAL="/home/wzzz/LPRNet/experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservative/Final_LPRNet_model.pth"
JOINT_FINAL="/home/wzzz/LPRNet/experiments/curriculum_gray3_stageB_v1_B1A_extreme_ccpdboard_v4e3/Final_LPRNet_model.pth"

for model in "$BASE_B1A_BEST" "$BASE_B1A_FINAL" "$JOINT_FINAL"; do
  if [[ ! -f "$model" ]]; then echo "[FATAL] missing model $model" >&2; exit 2; fi
done

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --model "$BASE_B1A_BEST" \
  --manifest_dir "$OLD_MANIFEST_DIR" \
  --out_dir "$OUT_DIR/base_B1A_best_on_old_proxy"

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --model "$BASE_B1A_BEST" \
  --manifest_dir "$NEW_MANIFEST_DIR" \
  --out_dir "$OUT_DIR/base_B1A_best_on_new_v4e3_ccpdboard_proxy"

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --model "$BASE_B1A_FINAL" \
  --manifest_dir "$OLD_MANIFEST_DIR" \
  --out_dir "$OUT_DIR/base_B1A_final_on_old_proxy"

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --model "$BASE_B1A_FINAL" \
  --manifest_dir "$NEW_MANIFEST_DIR" \
  --out_dir "$OUT_DIR/base_B1A_final_on_new_v4e3_ccpdboard_proxy"

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --model "$JOINT_FINAL" \
  --manifest_dir "$OLD_MANIFEST_DIR" \
  --out_dir "$OUT_DIR/joint_final_on_old_proxy"

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --model "$JOINT_FINAL" \
  --manifest_dir "$NEW_MANIFEST_DIR" \
  --out_dir "$OUT_DIR/joint_final_on_new_v4e3_ccpdboard_proxy"

python3 - <<'PY'
import json
from pathlib import Path
out=Path('/home/wzzz/LPRNet/experiments/stageB1A_B_dual_proxy_eval_20260425')
rows=[]
for d in sorted(out.iterdir()):
    r=d/'ranking.json'
    if not r.exists(): continue
    data=json.loads(r.read_text(encoding='utf-8'))[0]
    m=data['metrics']
    rows.append({
        'eval':d.name,
        'checkpoint':data['label'],
        'real_avg':data['real_avg'],
        'green_edgefit_extreme_exact':m['green_edgefit_extreme']['exact_plate_acc'],
        'green_edgefit_extreme_first':m['green_edgefit_extreme']['first_char_acc'],
        'hard':m['green_edgefit_hard']['exact_plate_acc'],
        'green_ccpd':m['green_ccpd2020_real']['exact_plate_acc'],
        'blue_ccpd':m['blue_ccpd2019_real']['exact_plate_acc'],
        'blue_crpd':m['blue_crpd_real']['exact_plate_acc'],
    })
(out/'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
lines=['# StageB1A B dual proxy eval','', '| eval | ckpt | real_avg | old/new extreme exact | first | hard | green_ccpd | blue_ccpd | blue_crpd |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x in rows:
    lines.append('| {eval} | {checkpoint} | {real_avg:.4f} | {green_edgefit_extreme_exact:.4f} | {green_edgefit_extreme_first:.4f} | {hard:.4f} | {green_ccpd:.4f} | {blue_ccpd:.4f} | {blue_crpd:.4f} |'.format(**x))
(out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines))
PY
