#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
ENTRY=/home/wzzz/LPRNet/scripts/run_lpr_python_entry.py
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1_existing_paths.csv
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v2.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_h35/H35D_edgefit_tier3_classbalanced_firstchar
BASELINE_DIR=/home/wzzz/LPRNet/experiments/green_h34/H34F_edgefit_tier3_v2

mkdir -p /home/wzzz/LPRNet/experiments/green_h35
rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

if [ ! -f "$INIT" ]; then
  echo "[FATAL] missing init checkpoint: $INIT"
  exit 2
fi
if [ ! -f "$TRAIN_MANIFEST" ]; then
  echo "[FATAL] missing train manifest: $TRAIN_MANIFEST"
  exit 2
fi
if [ ! -f "$EVAL_MANIFEST" ]; then
  echo "[FATAL] missing eval manifest: $EVAL_MANIFEST"
  exit 2
fi

echo "[RUN] H35D: H34F edgefit tier3 baseline + H32A class-balanced first-char"
echo "[CONFIG] Single variable = first_char_loss_type ce -> class_balanced_ce (beta=0.999)"
echo "[BASE] Keep H34F data, no-leak train manifest, board-aligned OCR chain, freeze_backbone, expD, standard CTC"

time $PY "$ENTRY" /home/wzzz/LPRNet/src/training/train_LPRNet.py \
  --cuda true \
  --data_mode manifest \
  --ocr_channel_order bgr \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$EVAL_MANIFEST" \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --freeze_backbone true \
  --trainable_families green8 \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --main_group_by family \
  --main_group_ratios green8=1.0 \
  --main_group_clip 1.0 \
  --enhanced_green_head expD \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0003 \
  --lr_schedule 7 10 15 20 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_strategy balanced_recovery \
  --ctc_loss_type standard \
  --first_char_loss_type class_balanced_ce \
  --first_char_cb_beta 0.999 \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 15 \
  2>&1 | tee "$SAVE_DIR/train.log"

if [ ! -f "$SAVE_DIR/Final_LPRNet_model.pth" ]; then
  echo "[FATAL] training finished without Final_LPRNet_model.pth"
  exit 3
fi

echo "[EVAL] Running family-aware detailed evaluation first..."
time $PY "$ENTRY" /home/wzzz/LPRNet/src/evaluation/eval_lpr_detailed.py \
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

echo "[EVAL] Running green8 metrics only..."
time $PY "$ENTRY" /home/wzzz/LPRNet/src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

echo "[EVAL] Running green8 by-dataset province breakdown..."
time $PY "$ENTRY" /home/wzzz/LPRNet/src/evaluation/eval_green8_by_dataset_province.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_by_dataset_province.json" \
  2>&1 | tee "$SAVE_DIR/eval_green8_by_dataset_province.log"

echo "[DONE] H35D complete"