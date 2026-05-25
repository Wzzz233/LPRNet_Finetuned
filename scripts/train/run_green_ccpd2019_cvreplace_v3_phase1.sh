#!/bin/bash
# Green CCPD2019 CV-replace v3 training: Phase 1 (5 epochs, freeze backbone)
# Then Phase 2 (5 epochs, unfreeze more backbone, LR /10)
# Every epoch saved for Pareto evaluation.

set -e

DATE_TAG="20260508"
EXP_DIR="experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_${DATE_TAG}"
mkdir -p "$EXP_DIR"

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

TRAIN_MANIFEST="manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v3_${DATE_TAG}/train_v3_balanced.csv"
VAL_MANIFEST="manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v3_${DATE_TAG}/val_v3.csv"
PRETRAINED="experiments/green_e12_province_degrade_unfreeze/best_LPRNet_model.pth"

# ── Preflight ──────────────────────────────────────────────────────
echo "=== Preflight ==="
python3 -u src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$VAL_MANIFEST" \
  --dataset_root /home/wzzz/LPRNet \
  --pretrained_model "$PRETRAINED" \
  --save_folder "$EXP_DIR" \
  --head_mode multihead --enhanced_green_head expD \
  --trainable_families green8 \
  --freeze_backbone True --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
  --preflight_only True --train_batch_size 8 --num_workers 2 --cuda True \
  2>&1 | tee "${EXP_DIR}/preflight.log"
echo "Preflight done."

# ── Phase 1: freeze backbone, 5 epochs ────────────────────────────
echo ""
echo "=== Phase 1: freeze backbone, 5 epochs, LR=5e-5 ==="
python3 -u src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$VAL_MANIFEST" \
  --dataset_root /home/wzzz/LPRNet \
  --pretrained_model "$PRETRAINED" \
  --save_folder "$EXP_DIR" \
  --head_mode multihead --enhanced_green_head expD \
  --trainable_families green8 \
  --freeze_backbone True --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
  --max_epoch 5 --train_batch_size 64 --num_workers 4 \
  --learning_rate 0.00005 --lr_schedule 3 4 \
  --lpr_max_len 8 \
  --ocr_crop_mode obb_warp --ocr_resize_mode letterbox --ocr_resize_kernel nn \
  --ocr_preproc none --ocr_channel_order bgr --ocr_quad_pad_ratio 0.0 \
  --province_balance_mode inv_sqrt \
  --train_brightness_aug_max 120.0 --gray3_prob 0.0 \
  --first_char_aux_weight 0.3 \
  --save_interval 2000 --test_interval 2000 \
  --cuda True --phase_train True 2>&1 | tee "${EXP_DIR}/phase1_train.log"

echo ""
echo "=== Phase 1 complete ==="
grep -E 'Epoch Summary|SelectionProxy|Training Done' "${EXP_DIR}/phase1_train.log"

# Save every epoch's checkpoint explicitly (training script does this)
echo "Phase 1 checkpoints:"
ls -la "${EXP_DIR}"/*iteration_*.pth 2>/dev/null || echo "(checkpointed during training)"

# ── Optional Phase 2: unfreeze more backbone ──────────────────────
# Phase 2 decision will be made after Phase 1 evaluation.
echo ""
echo "Phase 2 script ready for manual launch:"
echo "  scripts/train/run_green_ccpd2019_cvreplace_v3_phase2.sh"
echo ""
echo "To launch Phase 2 (unfreeze backbone.16/18/19/20, LR=5e-6, 5 epochs):"
echo "  python3 -u src/training/train_LPRNet.py \\"
echo "    --data_mode manifest \\"
echo "    --train_manifest $TRAIN_MANIFEST \\"
echo "    --test_manifest $VAL_MANIFEST \\"
echo "    --dataset_root /home/wzzz/LPRNet \\"
echo "    --pretrained_model ${EXP_DIR}/best_LPRNet_model.pth \\"
echo "    --save_folder ${EXP_DIR}/phase2 \\"
echo "    --head_mode multihead --enhanced_green_head expD \\"
echo "    --trainable_families green8 \\"
echo "    --freeze_backbone True --trainable_backbone_prefixes backbone.16,backbone.18,backbone.19,backbone.20 \\"
echo "    --max_epoch 5 --train_batch_size 64 --num_workers 4 \\"
echo "    --learning_rate 0.000005 --lr_schedule 3 4 \\"
echo "    ... (rest same as Phase 1)"
