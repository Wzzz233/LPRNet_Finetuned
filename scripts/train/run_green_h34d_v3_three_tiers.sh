#!/bin/bash
# H34D 训练脚本 - 三层 Edgefit
# 实验目的：验证三层透视变形对沪/湘/粤/浙的改善

set -e

cd /home/wzzz/LPRNet

EXPERIMENT_NAME="H34D_edgefit_v3_three_tiers"
EXPERIMENT_DIR="experiments/green_h34/${EXPERIMENT_NAME}"
mkdir -p ${EXPERIMENT_DIR}/{checkpoints,logs,eval}

# 基础配置
BASE_MANIFEST="manifests/unified_manifest_green_h34d_v3_three_tiers.csv"
EVAL_MANIFEST="manifests/unified_manifest_green_balance_aggr_v1.csv"

# 训练参数
BATCH_SIZE=128
EPOCHS=120
LR=0.001
IMG_SIZE="94,24"

# 训练命令
echo "Starting H34D training..."
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Manifest: ${BASE_MANIFEST}"
echo ""

/home/wzzz/LPRNet/.conda/bin/python train.py \
  --train ${BASE_MANIFEST} \
  --val ${BASE_MANIFEST} \
  --eval ${EVAL_MANIFEST} \
  --batch ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --img_size ${IMG_SIZE} \
  --checkpoint_dir ${EXPERIMENT_DIR}/checkpoints \
  --log_dir ${EXPERIMENT_DIR}/logs \
  --experiment_name ${EXPERIMENT_NAME} \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nearest \
  --ocr_channel_order bgr \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --plate_family_split green \
  --family_aware_decoding \
  --family_heads green:66 \
  2>&1 | tee ${EXPERIMENT_DIR}/logs/training.log

echo "Training complete!"
echo "Checkpoint dir: ${EXPERIMENT_DIR}/checkpoints"
