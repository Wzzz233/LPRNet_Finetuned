#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/training

INIT="/home/wzzz/LPRNet/experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth"
EXPECTED_SHA="d6c011e6572b025c5fae618bc253ec4f5d30e6acd60b46012ee6fdef79dcad2a"
MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original"
TRAIN_MANIFEST="$MANIFEST_DIR/train_B1A_C_train_v4e3_ccpdboard_eval_original.csv"
VAL_MANIFEST="$MANIFEST_DIR/val_B1A_C_original_eval.csv"
SAVE_DIR="experiments/curriculum_gray3_stageB2_S0_softfreeze_smoke"
LOG_PATH="$SAVE_DIR/preflight.log"

if [[ ! -f "$INIT" ]]; then echo "[FATAL] missing INIT $INIT" >&2; exit 2; fi
sha=$(sha256sum "$INIT" | awk '{print $1}')
if [[ "$sha" != "$EXPECTED_SHA" ]]; then echo "[FATAL] INIT sha mismatch got=$sha expected=$EXPECTED_SHA" >&2; exit 2; fi
for f in "$TRAIN_MANIFEST" "$VAL_MANIFEST"; do
  if [[ ! -f "$f" ]]; then echo "[FATAL] missing manifest $f" >&2; exit 2; fi
done

rm -rf "$SAVE_DIR"
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
  --max_epoch 1 \
  --learning_rate 0.0002 \
  --lr_schedule 3 5 \
  --freeze_backbone False \
  --freeze_bn_stats True \
  --backbone_lr_mult 0.1 \
  --head_lr_mult 1.0 \
  --adapter_lr_mult 1.0 \
  --aux_lr_mult 1.0 \
  --selection_proxy_eval_samples 3000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --save_folder "$SAVE_DIR" \
  --num_workers 2 \
  --seed 47 \
  --cuda True \
  --preflight_only True \
  > "$LOG_PATH" 2>&1

python3 - <<'PY'
from pathlib import Path
import re, sys, json
log = Path('/home/wzzz/LPRNet/experiments/curriculum_gray3_stageB2_S0_softfreeze_smoke/preflight.log')
s = log.read_text(encoding='utf-8')
required = [
    'Successful to build network!',
    '[Env] device=cuda:0',
    '[Freeze] backbone BN stats frozen for soft-freeze (params remain trainable)',
    '[PreflightOnly] model/data/optimizer ready; exit before training',
    'pos0=0.10',
    'prov=0.20',
]
missing = [x for x in required if x not in s]
for name, mult, lr in [
    ('backbone', '0.1', '0.00002000'),
    ('adapter', '1', '0.00020000'),
    ('head', '1', '0.00020000'),
]:
    pat = rf'\[OptimGroup\] name={name} .*lr_mult={mult} .*lr={lr}'
    if not re.search(pat, s):
        missing.append(pat)
if missing:
    print('[FATAL] preflight log missing markers:', json.dumps(missing, ensure_ascii=False), file=sys.stderr)
    print(s[-4000:], file=sys.stderr)
    sys.exit(2)
print(json.dumps({'preflight': 'ok', 'log': str(log)}, ensure_ascii=False))
PY
