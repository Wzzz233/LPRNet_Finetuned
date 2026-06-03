#!/bin/bash
# Formal strict CV training commands (3 stages)
# After smoke QA passes, run these in order.

EXP="experiments/police_bluebase_cvstrict_20260603"
ROOT="/home/wzzz/LPRNet"
MANI="manifests_rebased/police_bluebase_cvstrict_20260603"
KEYS="keys/police_keys.txt"
INIT="${EXP}/init_from_bluebase_police_keys.pth"

# Stage A: frozen backbone, train head (20 epochs)
echo "=== Stage A: frozen backbone ==="
python3 src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest ${MANI}/train.csv \
  --test_split_filter val_clean \
  --test_manifest ${MANI}/val_clean.csv \
  --keys_file ${KEYS} \
  --pretrained_model ${INIT} \
  --save_folder ${EXP}/strictcv_stageA/ \
  --max_epoch 20 \
  --learning_rate 0.001 \
  --backbone_lr_mult 0.0 --head_lr_mult 1.0 \
  --freeze_backbone true --freeze_bn_stats true \
  --first_char_aux_weight 0.10 \
  --lr_schedule 10 15 \
  --train_batch_size 64 --test_batch_size 120 \
  --num_workers 0 --cuda true --dataset_root . \
  --img_size 94 24 \
  --ocr_crop_mode plain --ocr_preproc none --ocr_channel_order bgr \
  --seed 42 --deterministic true \
  > ${EXP}/strictcv_stageA/train.log 2>&1

# Stage B: unfreeze backbone.16-21 (20 epochs)
echo "=== Stage B: unfreeze backbone.16-21 ==="
python3 src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest ${MANI}/train.csv \
  --test_split_filter val_clean \
  --test_manifest ${MANI}/val_clean.csv \
  --keys_file ${KEYS} \
  --pretrained_model ${EXP}/strictcv_stageA/best_LPRNet_model.pth \
  --save_folder ${EXP}/strictcv_stageB/ \
  --max_epoch 20 \
  --learning_rate 0.0005 \
  --backbone_lr_mult 0.1 --head_lr_mult 1.0 \
  --freeze_backbone true --freeze_bn_stats false \
  --trainable_backbone_prefixes backbone.16,backbone.17,backbone.18,backbone.19,backbone.20,backbone.21 \
  --first_char_aux_weight 0.10 \
  --lr_schedule 10 15 \
  --train_batch_size 64 --test_batch_size 120 \
  --num_workers 0 --cuda true --dataset_root . \
  --img_size 94 24 \
  --ocr_crop_mode plain --ocr_preproc none --ocr_channel_order bgr \
  --seed 42 --deterministic true \
  > ${EXP}/strictcv_stageB/train.log 2>&1

# Stage C: full fine-tune (30 epochs)
echo "=== Stage C: full fine-tune ==="
python3 src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest ${MANI}/train.csv \
  --test_split_filter val_clean \
  --test_manifest ${MANI}/val_clean.csv \
  --keys_file ${KEYS} \
  --pretrained_model ${EXP}/strictcv_stageB/best_LPRNet_model.pth \
  --save_folder ${EXP}/strictcv_stageC/ \
  --max_epoch 30 \
  --learning_rate 0.0001 \
  --backbone_lr_mult 0.5 --head_lr_mult 1.0 \
  --freeze_backbone false --freeze_bn_stats false \
  --first_char_aux_weight 0.05 \
  --lr_schedule 15 25 \
  --train_batch_size 64 --test_batch_size 120 \
  --num_workers 0 --cuda true --dataset_root . \
  --img_size 94 24 \
  --ocr_crop_mode plain --ocr_preproc none --ocr_channel_order bgr \
  --seed 42 --deterministic true \
  > ${EXP}/strictcv_stageC/train.log 2>&1

echo "=== All strict CV stages complete ==="
