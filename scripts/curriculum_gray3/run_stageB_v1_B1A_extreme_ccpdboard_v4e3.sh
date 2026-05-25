#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

INIT="/home/wzzz/LPRNet/experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth"
EXPECTED_SHA="d6c011e6572b025c5fae618bc253ec4f5d30e6acd60b46012ee6fdef79dcad2a"
TRAIN_MANIFEST="manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/train_B1A_extreme_ccpdboard_v4e3.csv"
VAL_MANIFEST="manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/val_B1A_extreme_ccpdboard_v4e3.csv"
MANIFEST_SUMMARY="manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/summary.json"
SAVE_DIR="experiments/curriculum_gray3_stageB_v1_B1A_extreme_ccpdboard_v4e3"
LOG_PATH="$SAVE_DIR/train.log"

if [[ ! -f "$INIT" ]]; then
  echo "[FATAL] missing exact A1D iter_002000 mother: $INIT" >&2
  exit 2
fi
sha=$(sha256sum "$INIT" | awk '{print $1}')
if [[ "$sha" != "$EXPECTED_SHA" ]]; then
  echo "[FATAL] A1D mother sha mismatch: got=$sha expected=$EXPECTED_SHA" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VAL_MANIFEST" || ! -f "$MANIFEST_SUMMARY" ]]; then
  echo "[FATAL] missing extreme ccpd_board replay manifests" >&2
  exit 2
fi

python3 - <<'PY'
import csv, json, sys
from pathlib import Path
train = Path('/home/wzzz/LPRNet/manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/train_B1A_extreme_ccpdboard_v4e3.csv')
summary = json.loads(Path('/home/wzzz/LPRNet/manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/summary.json').read_text(encoding='utf-8'))
need = summary['train_extreme_count']
count = 0
bad = []
with train.open('r', encoding='utf-8', newline='') as f:
    for row in csv.DictReader(f):
        if row.get('source') == 'green_edgefit_extreme_v4e3_ccpdboard':
            count += 1
            checks = {
                'preprocess_group': 'ccpd_board',
                'has_quad': '1',
                'can_parse_ccpd_geom': '1',
                'can_perspective': '1',
                'ocr_crop_mode': 'obb_warp',
                'ocr_resize_mode': 'letterbox',
                'ocr_resize_kernel': 'nn',
                'ocr_preproc': 'gray3',
                'ocr_channel_order': 'bgr',
            }
            for k, v in checks.items():
                if row.get(k) != v:
                    bad.append((row.get('img_path'), k, row.get(k), v))
            q_keys = ['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']
            if not all(row.get(k) for k in q_keys):
                bad.append((row.get('img_path'), 'quad', 'missing', 'present'))
if count != need:
    print(f'[FATAL] extreme count mismatch: train has {count}, summary says {need}', file=sys.stderr)
    sys.exit(2)
if bad:
    print('[FATAL] manifest preflight failed on extreme replay rows:', file=sys.stderr)
    for item in bad[:10]:
        print(item, file=sys.stderr)
    sys.exit(2)
print(json.dumps({'train_extreme_count': count, 'bad_rows': len(bad)}, ensure_ascii=False))
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
  --manifest_dir manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3 \
  --out_dir "$SAVE_DIR/proxy_eval_stageB_v1"
