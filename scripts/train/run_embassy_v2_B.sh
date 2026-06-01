#!/bin/bash
# Embassy B: official weights warm start + full finetune
# pretrained_model: models/weights/weights_official/Final_LPRNet_model.pth

DATE_TAG="20260601"
EXP_NAME="embassy_v2_fullft_officialwarm_${DATE_TAG}"

cd /home/wzzz/LPRNet

python3 src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest manifests_rebased/special_split_v2_20260601/train_embassy.csv \
  --test_split_filter val_clean \
  --test_manifest manifests_rebased/special_split_v2_20260601/val_clean_embassy.csv \
  --keys_file keys/embassy_keys.txt \
  --pretrained_model models/weights/weights_official/Final_LPRNet_model.pth \
  --save_folder experiments/${EXP_NAME}/ \
  --max_epoch 40 \
  --learning_rate 0.001 \
  --backbone_lr_mult 0.1 \
  --head_lr_mult 1.0 \
  --freeze_backbone false \
  --freeze_bn_stats false \
  --first_char_aux_weight 0.0 \
  --lr_schedule 20 30 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --num_workers 0 \
  --cuda true \
  --img_size 94 24 \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_channel_order bgr \
  --train_plate_box_aug_mode none \
  --train_brightness_aug_max 0.0 \
  --momentum 0.9 \
  --weight_decay 2e-5 \
  --selection_proxy_eval_samples 5000 \
  --selection_strategy proxy_exact \
  --selection_decode_mode greedy \
  --early_stop_patience 0 \
  --seed 42 \
  --deterministic true \
  2>&1 | tee /home/wzzz/LPRNet/experiments/${EXP_NAME}/train.log
