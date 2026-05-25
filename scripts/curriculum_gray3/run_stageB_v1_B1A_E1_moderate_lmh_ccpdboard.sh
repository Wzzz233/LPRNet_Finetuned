#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

INIT="/home/wzzz/LPRNet/experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth"
EXPECTED_SHA="d6c011e6572b025c5fae618bc253ec4f5d30e6acd60b46012ee6fdef79dcad2a"
MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original"
NEW_PROXY_MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy"
TRAIN_MANIFEST="$MANIFEST_DIR/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv"
VAL_MANIFEST="$MANIFEST_DIR/val_B1A_E1_original_eval.csv"
SAVE_DIR="experiments/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original"
LOG_PATH="$SAVE_DIR/train.log"
REPORT="reports/GREEN_STAGEB1A_E1_MODERATE_LMH_CCPDBOARD_TRAINING_REPORT.md"

if [[ ! -f "$INIT" ]]; then
  echo "[FATAL] missing exact A1D iter_002000 mother: $INIT" >&2
  exit 2
fi
sha=$(sha256sum "$INIT" | awk '{print $1}')
if [[ "$sha" != "$EXPECTED_SHA" ]]; then
  echo "[FATAL] A1D mother sha mismatch: got=$sha expected=$EXPECTED_SHA" >&2
  exit 2
fi
for p in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$MANIFEST_DIR/proxy_green_edgefit_extreme.csv" "$NEW_PROXY_MANIFEST_DIR/proxy_green_edgefit_extreme.csv"; do
  if [[ ! -f "$p" ]]; then
    echo "[FATAL] missing manifest/proxy: $p" >&2
    exit 2
  fi
done

python3 - <<'PY'
import csv, json
from pathlib import Path
from collections import Counter
root=Path('/home/wzzz/LPRNet')
train=root/'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv'
rows=list(csv.DictReader(train.open(encoding='utf-8')))
ext=[r for r in rows if r.get('source')=='green_edgefit_extreme_E1_moderate_lmh_ccpdboard']
assert len(rows)==65575, len(rows)
assert len(ext)==300, len(ext)
assert Counter(r.get('difficulty_tier') for r in ext)==Counter({'low':120,'mid':120,'high':60}), Counter(r.get('difficulty_tier') for r in ext)
for r in ext:
    assert r.get('preprocess_group')=='ccpd_board'
    assert r.get('has_quad')=='1' and r.get('can_parse_ccpd_geom')=='1' and r.get('can_perspective')=='1'
    assert r.get('ocr_crop_mode')=='obb_warp' and r.get('ocr_resize_mode')=='letterbox'
    assert r.get('ocr_resize_kernel')=='nn' and r.get('ocr_preproc')=='gray3' and r.get('ocr_channel_order')=='bgr'
    assert r.get('ocr_quad_pad_ratio')=='0.0'
    for k in ['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']:
        assert r.get(k,'')!=''
    assert Path(r['img_path']).exists(), r['img_path']
print(json.dumps({'preflight':'ok','rows':len(rows),'e1_extreme':len(ext),'tier_counts':dict(Counter(r.get('difficulty_tier') for r in ext))},ensure_ascii=False))
PY

if [[ -e "$SAVE_DIR" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  mv "$SAVE_DIR" "${SAVE_DIR}.bak_${ts}"
  echo "[INFO] backed up old SAVE_DIR to ${SAVE_DIR}.bak_${ts}"
fi
mkdir -p "$SAVE_DIR"

python3 src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$VAL_MANIFEST" \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --trainable_families normal7,green8 \
  --adapter_target_families green8 \
  --adapter_hidden_channels 128 \
  --province_head_weight 0.20 \
  --province_num_classes 31 \
  --province_target_families green8 \
  --pos0_head_cols 4 \
  --pos0_head_weight 0.10 \
  --pos0_num_classes 31 \
  --pos0_target_families green8 \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_channel_order bgr \
  --ocr_preproc gray3 \
  --gray3_prob 1.0 \
  --main_group_by family \
  --train_batch_size 128 \
  --test_batch_size 120 \
  --max_epoch 6 \
  --learning_rate 0.0002 \
  --lr_schedule 3 5 \
  --freeze_backbone True \
  --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
  --selection_proxy_eval_samples 3000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --save_folder "$SAVE_DIR" \
  --num_workers 4 \
  --seed 47 \
  --cuda True \
  > "$LOG_PATH" 2>&1

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --exp_dir "$SAVE_DIR" \
  --manifest_dir "$MANIFEST_DIR" \
  --out_dir "$SAVE_DIR/proxy_eval_stageB_v1_old_proxy"

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --exp_dir "$SAVE_DIR" \
  --manifest_dir "$NEW_PROXY_MANIFEST_DIR" \
  --out_dir "$SAVE_DIR/proxy_eval_stageB_v1_new_proxy"

python3 - <<'PY'
import json
from pathlib import Path
root=Path('/home/wzzz/LPRNet')
exp=root/'experiments/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original'
report=root/'reports/GREEN_STAGEB1A_E1_MODERATE_LMH_CCPDBOARD_TRAINING_REPORT.md'
old=json.loads((exp/'proxy_eval_stageB_v1_old_proxy/ranking.json').read_text(encoding='utf-8'))[0]
new=json.loads((exp/'proxy_eval_stageB_v1_new_proxy/ranking.json').read_text(encoding='utf-8'))[0]
def row(label,r):
    m=r['metrics']
    return f"| {label} | {r['label']} | {'Y' if r['pass_abs_gate'] else 'N'} | {r['real_avg']*100:.2f}% | {r['family_gap']*100:.2f}pp | {m['green_edgefit_hard']['exact_plate_acc']*100:.2f}% | {m['green_edgefit_extreme']['exact_plate_acc']*100:.2f}% | {m['green_edgefit_extreme']['first_char_acc']*100:.2f}% |"
lines=[]
lines.append('# GREEN StageB1A-E1 Moderate LMH ccpd_board Training Report')
lines.append('')
lines.append('日期：2026-04-26')
lines.append('')
lines.append('## 定义')
lines.append('只替换原 StageB1A train extreme 300 行为 accepted moderate low/mid/high ccpd_board；old proxy 保持不变，new proxy 只作辅助评测。')
lines.append('')
lines.append('## Best checkpoints')
lines.append(f"old proxy best: `{old['checkpoint']}`")
lines.append(f"new proxy best: `{new['checkpoint']}`")
lines.append('')
lines.append('| eval | ckpt | pass | real_avg | family_gap | hard | extreme exact | extreme first |')
lines.append('|---|---|---:|---:|---:|---:|---:|---:|')
lines.append(row('old proxy', old))
lines.append(row('new proxy', new))
lines.append('')
lines.append('## 产物')
lines.append(f'exp_dir: `{exp}`')
lines.append(f'old ranking: `{exp}/proxy_eval_stageB_v1_old_proxy/ranking.json`')
lines.append(f'new ranking: `{exp}/proxy_eval_stageB_v1_new_proxy/ranking.json`')
report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(report)
PY
