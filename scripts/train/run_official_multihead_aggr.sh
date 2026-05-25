#!/bin/bash
cd /home/wzzz/LPRNet
/home/wzzz/LPRNet/.conda/bin/python /home/wzzz/LPRNet/train_LPRNet.py \
  --cuda true \
  --data_mode manifest \
  --manifest /home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv \
  --ocr_channel_order bgr \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --pretrained_model /home/wzzz/LPRNet/weights_official/Final_LPRNet_model.pth \
  --head_mode multihead \
  --freeze_backbone true \
  --max_epoch 6 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0003 \
  --lr_schedule 3 5 \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --first_char_aux_weight 0.4 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --selection_proxy_eval_samples 5000 \
  --selection_decode_mode family_aware_beam \
  --selection_beam_size 20 \
  --selection_beam_topk 12 \
  --main_group_by preprocess_group \
  --main_group_ratios ccpd_board=0.90,plain_plate=0.10 \
  --main_group_clip 2.0 \
  --early_stop_patience 2 \
  --early_stop_regression_patience 2 \
  --early_stop_regression_pp 0.2 \
  --early_stop_start_epoch 3 \
  --save_folder /home/wzzz/LPRNet/experiments/green_multihead_round1/official_multihead_aggr/ \
  2>&1 | tee /home/wzzz/LPRNet/experiments/green_multihead_round1/official_multihead_aggr/train.log
