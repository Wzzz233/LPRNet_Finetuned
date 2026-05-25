# Green Plate Expert Completion Plan

## Source

- **expert**: green_plate_expert
- **source_training**: green_e12_province_degrade_unfreeze
- **primary_checkpoint**: experiments/green_e12_province_degrade_unfreeze/Final_LPRNet_model.pth
- **backup_checkpoints**:
  - experiments/green_e12_province_degrade_unfreeze/best_LPRNet_model.pth
- **manifest**: manifests_rebased/province_degrade_train_v1/train_province_degrade_v1.csv (rebased)
- **test_manifest**: manifests_rebased/unified_manifest_green_e12_pose_replace_test.csv (rebased)
- **keys_charset**: Default CHARS (no keys_file specified in training)
- **board_onnx**: experiments/green_e12_province_degrade_unfreeze/prov_deg_fp16.onnx
- **board_rknn**: experiments/green_e12_province_degrade_unfreeze/prov_deg_fp16_no_rknnpre.rknn
- **alternate_rknn**: experiments/green_e12_province_degrade_unfreeze/prov_deg_fp16.rknn
- **status**: needs_board_artifact_copy