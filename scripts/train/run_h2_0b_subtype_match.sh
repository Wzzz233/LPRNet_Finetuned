#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/green_balance_round2_subtype_matched.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/h2_0b_subtype_match

mkdir -p "$SAVE_DIR"

# H2-0b: 快速采样对照实验
# 只改 subtype 采样比例，不改模型结构/loss，短程 5 epoch
$PY train_LPRNet.py \
  --cuda true \
  --data_mode manifest \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$EVAL_MANIFEST" \
  --ocr_channel_order bgr \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --enhanced_green_head expD \
  --freeze_backbone true \
  --trainable_families green8 \
  --save_folder "$SAVE_DIR" \
  --train_batch_size 64 \
  --max_epoch 5 \
  --learning_rate 0.0003 \
  --lr_schedule 3 5 \
  --test_interval 5 \
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
  2>&1 | tee "$SAVE_DIR/train.log"

echo "Training complete. Running evaluation..."

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
  --pretrained_model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --test_batch_size 300 \
  --decode_mode family_aware_beam \
  --beam_size 20 \
  --beam_topk 12 \
  --out_json "$SAVE_DIR/eval_family_aware.json" \
  2>&1 | tee "$SAVE_DIR/eval_family_aware.log"

echo "Evaluation complete. Results:"
cat "$SAVE_DIR/eval_family_aware.json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get('family_breakdown',{}).get('green8',{}), ensure_ascii=False, indent=2))"
