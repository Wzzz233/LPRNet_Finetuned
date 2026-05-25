#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
GREEN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b.csv
ROOT_EXP=/home/wzzz/LPRNet/experiments/green_post_g2
mkdir -p "$ROOT_EXP"

COMMON_ARGS=(
  --cuda true
  --data_mode manifest
  --ocr_channel_order bgr
  --ocr_crop_mode obb_warp
  --ocr_resize_mode letterbox
  --ocr_resize_kernel nn
  --ocr_preproc none
  --ocr_min_occ_ratio 0.90
  --ocr_quad_pad_ratio 0.0
  --pretrained_model "$INIT"
  --head_mode multihead
  --freeze_backbone true
  --trainable_families green8
  --province_balance_mode inv_sqrt
  --strata_balance_mode none
  --first_char_aux_weight 0.4
  --second_char_aux_weight 0.0
  --ne_type_aux_weight 0.0
  --selection_decode_mode family_aware_beam
  --selection_beam_size 20
  --selection_beam_topk 12
  --main_group_by family
  --main_group_ratios green8=1.0
  --main_group_clip 1.0
  --enhanced_green_head expD
)

run_eval_pair() {
  local save_dir="$1"
  if [[ ! -f "$save_dir/eval_family_aware.json" ]]; then
    $PY eval_lpr_detailed.py \
      --cuda true \
      --data_mode manifest \
      --test_img_dirs "$EVAL_MANIFEST" \
      --txt_file "$EVAL_MANIFEST" \
      --ocr_channel_order bgr \
      --ocr_crop_mode obb_warp \
      --ocr_resize_mode letterbox \
      --ocr_resize_kernel nn \
      --ocr_preproc none \
      --ocr_min_occ_ratio 0.90 \
      --ocr_quad_pad_ratio 0.0 \
      --head_mode multihead \
      --enhanced_green_head expD \
      --pretrained_model "$save_dir/Final_LPRNet_model.pth" \
      --test_batch_size 300 \
      --decode_mode family_aware_beam \
      --beam_size 20 \
      --beam_topk 12 \
      --out_json "$save_dir/eval_family_aware.json" \
      2>&1 | tee "$save_dir/eval_family_aware.log"
  fi
}

run_diff_check() {
  local save_dir="$1"
  $PY - <<'PY' "$INIT" "$save_dir/Final_LPRNet_model.pth" "$save_dir/param_diff_summary.json"
import json, sys, torch
before = torch.load(sys.argv[1], map_location='cpu')
after = torch.load(sys.argv[2], map_location='cpu')
counts = {'backbone_param_changed':0,'backbone_bn_buffer_changed':0,'normal7_param_changed':0,'green8_param_changed':0,'special_param_changed':0}
for k in before:
    if k not in after:
        continue
    same = torch.equal(before[k], after[k])
    if k.startswith('backbone.'):
        if ('running_mean' in k) or ('running_var' in k) or ('num_batches_tracked' in k):
            counts['backbone_bn_buffer_changed'] += int(not same)
        else:
            counts['backbone_param_changed'] += int(not same)
    elif k.startswith('containers.normal7.'):
        counts['normal7_param_changed'] += int(not same)
    elif k.startswith('containers.green8.'):
        counts['green8_param_changed'] += int(not same)
    elif k.startswith('containers.special.'):
        counts['special_param_changed'] += int(not same)
with open(sys.argv[3], 'w', encoding='utf-8') as f:
    json.dump(counts, f, ensure_ascii=False, indent=2)
print(json.dumps(counts, ensure_ascii=False))
PY
}

run_one() {
  local name="$1"
  local max_epoch="$2"
  local batch_size="$3"
  local lr="$4"
  local schedule_csv="$5"
  local selection_proxy_samples="$6"
  shift 6
  local save_dir="$ROOT_EXP/$name"
  rm -rf "$save_dir"
  mkdir -p "$save_dir"
  echo "[RUN] $name (epochs=$max_epoch, batch=$batch_size, lr=$lr, schedule=$schedule_csv)"
  # shellcheck disable=SC2206
  local sched=($schedule_csv)
  $PY train_LPRNet.py \
    --manifest "$GREEN_MANIFEST" \
    --max_epoch "$max_epoch" \
    --train_batch_size "$batch_size" \
    --test_batch_size 120 \
    --learning_rate "$lr" \
    --lr_schedule "${sched[@]}" \
    --selection_proxy_eval_samples "$selection_proxy_samples" \
    --save_folder "$save_dir/" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$save_dir/train.log"
  run_eval_pair "$save_dir"
  run_diff_check "$save_dir"
}

echo "=========================================="
echo "Starting G3: D + 15 epoch + lr_schedule 7 10"
echo "=========================================="
run_one G3_D_15epoch_schedule_7_10 15 64 0.0003 "7 10" 2000

echo "=========================================="
echo "Starting G4: D + 20 epoch + lr_schedule 7 10 15 20"
echo "=========================================="
run_one G4_D_20epoch_schedule_7_10_15_20 20 64 0.0003 "7 10 15 20" 2000

echo "[DONE] G3 and G4 experiments complete"
