# Blue Plate Expert Completion Plan

## Source

- **expert**: blue_plate_expert
- **source_training**: tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319
- **training_chain**: v2 -> v3 -> v5 -> v6 -> v7
- **primary_checkpoint**: experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth
- **backup_checkpoints**:
  - experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/LPRNet__iteration_8000.pth
  - experiments/first_board_baseline_v1/weights/Final_LPRNet_model.pth
  - experiments/first_char_guard_v1/weights/Final_LPRNet_model.pth
- **manifest**: prepared_labels/ccpd2019_hard_tilt/train_labels.txt (old-style txt)
- **rebased_manifest**: manifests_rebased/curriculum_gray3/val.csv (fallback)
- **keys_charset**: Default CHARS from load_data.py (67 chars, no separate keys file)
- **board_onnx**: experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/LPRNet_stage3_rk3568_fp16.onnx
- **board_rknn**: experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/LPRNet_stage3_rk3568_fp16_more_trained.rknn
- **backup_rknn_locations**:
  - experiments/first_board_baseline_v1/weights/LPRNet_stage3_rk3568_fp16.rknn
  - experiments/first_char_guard_v1/weights/LPRNet_stage3_rk3568_fp16.rknn
- **conversion_script**: scripts/train/run_next_tilt_finetune_from_v6.sh (stage3 export embedded)
- **status**: needs_rebased_manifest_identification