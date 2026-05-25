#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
GREEN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_multihead_round3/D_green_head_capacity_e5

rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

echo "[RUN] D_green_head_capacity_e5 with enhanced green8 head"
$PY train_LPRNet.py \
  --cuda true \
  --data_mode manifest \
  --manifest "$GREEN_MANIFEST" \
  --ocr_channel_order bgr \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --enhanced_green_head true \
  --freeze_backbone true \
  --trainable_families green8 \
  --max_epoch 5 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0003 \
  --lr_schedule 3 5 \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --first_char_aux_weight 0.4 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --selection_decode_mode family_aware_beam \
  --selection_beam_size 20 \
  --selection_beam_topk 12 \
  --main_group_by family \
  --main_group_ratios green8=1.0 \
  --main_group_clip 1.0 \
  --save_folder "$SAVE_DIR/" \
  2>&1 | tee "$SAVE_DIR/train.log"

echo "[EVAL] Running family-aware evaluation..."
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
  --enhanced_green_head true \
  --pretrained_model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --test_batch_size 300 \
  --decode_mode family_aware_beam \
  --beam_size 20 \
  --beam_topk 12 \
  --out_json "$SAVE_DIR/eval_family_aware.json" \
  2>&1 | tee "$SAVE_DIR/eval_family_aware.log"

echo "[DIFF] Checking parameter changes..."
$PY - <<'PY' "$INIT" "$SAVE_DIR/Final_LPRNet_model.pth"
import torch, json, sys
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
print(json.dumps(counts, ensure_ascii=False))
PY

echo "[DONE] Experiment complete. Results saved to $SAVE_DIR"
