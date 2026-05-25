# LPRNet Model Zoo Index

## Overview

Four expert models are archived. This is a **lightweight index archive**:
metadata (README, lineage.json, keys, manifests) is copied into model_zoo/,
while checkpoints and board artifacts remain referenced in-place.

---

## 1. blue_plate_expert

| Field | Value |
|-------|-------|
| Purpose | Standard blue (蓝牌) license plate OCR |
| Source experiment | `tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319` |
| Training chain | baseline → v2 → v3 → v5 → v6 → v7 |
| Primary checkpoint | `experiments/.../weights_stageC/Final_LPRNet_model.pth` |
| Backup checkpoints | various iteration checkpoints in weights_stageC/ |
| Board ONNX | `experiments/.../weights_stageC/LPRNet_stage3_rk3568_fp16.onnx` |
| Board RKNN | `experiments/.../weights_stageC/LPRNet_stage3_rk3568_fp16_more_trained.rknn` |
| Keys | Default CHARS from load_data.py (no separate keys file) |
| Manifest | `prepared_labels/ccpd2019_hard_tilt/train_labels.txt` (old-style txt) |
| Rebasing manifest | Not yet identified |
| dataset_root | `/home/wzzz/LPRNet` |
| Can retrain | Yes |
| Can deploy | Yes (deployed) |
| Missing | Rebasing CSV manifest identification |
| Manual confirm | Verify rebasing manifest path |

## 2. green_plate_expert

| Field | Value |
|-------|-------|
| Purpose | Green (绿牌) province degradation OCR |
| Source experiment | `green_e12_province_degrade_unfreeze` |
| Primary checkpoint | `experiments/green_e12_province_degrade_unfreeze/Final_LPRNet_model.pth` |
| Board ONNX | `experiments/green_e12_province_degrade_unfreeze/prov_deg_fp16.onnx` |
| Board RKNN | `experiments/green_e12_province_degrade_unfreeze/prov_deg_fp16_no_rknnpre.rknn` |
| Keys | Default CHARS (no keys_file) |
| Rebasing manifest | `manifests_rebased/province_degrade_train_v1/train_province_degrade_v1.csv` |
| dataset_root | `/home/wzzz/LPRNet` |
| Can retrain | Yes |
| Can deploy | Yes (deployed) |

## 3. yellow_from_blue_expert

| Field | Value |
|-------|-------|
| Purpose | Yellow (黄牌) OCR, finetuned from blue |
| Source experiment | `yellow_single_v1_phase1` → `yellow_single_v1_phase2` |
| Primary checkpoint | `experiments/yellow_single_v1_phase2/Final_LPRNet_model.pth` |
| Backup checkpoint | `experiments/yellow_single_v1_phase1/Final_LPRNet_model.pth` |
| Board ONNX | `artifacts/yellow_LPRNet_v5_phase2.onnx` |
| Board RKNN | `artifacts/yellow_LPRNet_v5_phase2_fp16.rknn` |
| Earlier versions | v1 (yellow_LPRNet.onnx/.rknn), v3 (yellow_LPRNet_v3.onnx/.rknn) |
| Keys | `keys/yellow_keys.txt` (69 chars) |
| Rebasing manifest | `manifests_rebased/yellow_train.csv` (rebased_verified) |
| dataset_root | `/home/wzzz/LPRNet` |
| Can retrain | Yes |
| Can deploy | Yes (deployed) |
| Status | **Most complete expert** |

## 4. special_plate_expert

| Field | Value |
|-------|-------|
| Purpose | Special plate (警/使/领/澳/港) OCR, finetuned from yellow |
| Source experiment | `special_yellow_v5` |
| Primary checkpoint | `experiments/special_yellow_v5/Final_LPRNet_model.pth` |
| Board ONNX | `artifacts/special_LPRNet.onnx` |
| Board RKNN | `artifacts/special_LPRNet_fp16.rknn` |
| Keys | `keys/special_keys.txt` |
| Rebasing manifest | `manifests_rebased/special_train.csv` |
| dataset_root | `/home/wzzz/LPRNet` |
| Can retrain | Yes |
| Can deploy | Yes (deployed) |

## Build Full Archive

```bash
python tools/build_expert_archives.py --execute --copy-checkpoints --copy-board-artifacts
```

This copies all checkpoints (~18 MB) and ONNX/RKNN artifacts (~25 MB) into model_zoo/.
Only run when you need a self-contained package.
