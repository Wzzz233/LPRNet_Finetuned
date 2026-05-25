#!/usr/bin/env bash
# Step4: Full evaluation with trajectory fusion for board dumps
set -euo pipefail
cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

MODEL=$1
[[ -f "$MODEL" ]] || { echo "Usage: $0 <model.pth>"; exit 1; }
EXP_NAME=$(basename $(dirname $MODEL))
echo "=== Full eval: $EXP_NAME ==="

# 1. Real test
echo -e "\n[1] Real test set"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$MODEL" --manifest /tmp/e12_real_test.csv \
  --out_json "/tmp/${EXP_NAME}_real.json" \
  --batch_size 300 --num_workers 4 --ocr_preproc none 2>&1 | tail -3

# 2. Replacement val
echo -e "\n[2] Replacement val"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$MODEL" --manifest /tmp/e12_replace_val_test.csv \
  --out_json "/tmp/${EXP_NAME}_replace_val.json" \
  --batch_size 300 --num_workers 4 --ocr_preproc none 2>&1 | tail -3

# 3. Stress test
echo -e "\n[3] Province stress test"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$MODEL" \
  --manifest manifests/province_stress_pose_val_v1/province_stress_pose_val_v1.csv \
  --out_json "/tmp/${EXP_NAME}_stress.json" \
  --batch_size 300 --num_workers 4 --ocr_preproc none 2>&1 | tail -3

# 4. Cluster3 trajectory fusion
echo -e "\n[4] Cluster3 traj (pos_ocr_dump, GT=苏BF01111)"
$PY src/utils/trajectory_fusion.py \
  --weights "$MODEL" \
  --dump_dir /mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump \
  --gt 苏BF01111 \
  --out_json "/tmp/${EXP_NAME}_c3_traj.json" 2>&1 | tail -12

# 5. Cluster2 trajectory fusion
echo -e "\n[5] Cluster2 traj (pos_ocr_dump_2, GT=京AD06088)"
$PY src/utils/trajectory_fusion.py \
  --weights "$MODEL" \
  --dump_dir /mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2 \
  --gt 京AD06088 \
  --out_json "/tmp/${EXP_NAME}_c2_traj.json" 2>&1 | tail -12

# 6-7 CSV board dump evals (single-frame, for comparison)
echo -e "\n[6] Cluster2 CSV"
$PY src/utils/eval_gray3_board_dump.py \
  --weights "$MODEL" \
  --csv /home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv \
  --family green8 --out_json "/tmp/${EXP_NAME}_c2_csv.json" 2>&1 | tail -5

echo -e "\n[7] Cluster3 CSV"
$PY src/utils/eval_gray3_board_dump.py \
  --weights "$MODEL" \
  --csv /home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv \
  --family green8 --out_json "/tmp/${EXP_NAME}_c3_csv.json" 2>&1 | tail -5

# 8. Summary
echo -e "\n=== Summary ==="
python3 -c "
import json
for name, key in [('Real test','real'),('Replace val','replace_val'),('Stress test','stress')]:
    d=json.load(open(f'/tmp/${EXP_NAME}_{key}.json'))
    print(f'{name+\":\":20} exact={d[\"exact_plate_acc\"]*100:.2f}% first={d[\"first_char_acc\"]*100:.2f}%')
for name, key in [('C2 traj','c2_traj'),('C3 traj','c3_traj')]:
    d=json.load(open(f'/tmp/${EXP_NAME}_{key}.json'))
    print(f'{name+\":\":20} fused=\"{d[\"fused_text\"]}\" prov={d[\"province_status\"]} GTtop5={d[\"gt_in_top5_ever\"]}')
"
echo -e "\n[Done]"
