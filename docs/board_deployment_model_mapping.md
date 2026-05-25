# Board Deployment Model Mapping

> Source: board inference command
> Path: /userdata/model/ on RK3568 board

## Active Board Models

| Role | Model File | Workspace Source | Training Source | Status |
|------|-----------|-----------------|----------------|--------|
| **OCR Blue** | `LPRNet_stage3_rk3568_fp16_more_trained.rknn` | `experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/` | tilt_ocr_obbwarp_v7 (from baseline → v3 → v5 → v6 → v7) | ✅ Found, documented |
| **OCR Green** | `prov_deg_fp16_no_rknnpre.rknn` | **NOT IN WORKSPACE** | unknown (province_degrade based) | ❌ Missing in workspace |
| **OCR Yellow** | `yellow_LPRNet_v5_phase2_fp16.rknn` | `artifacts/yellow_LPRNet_v5_phase2.onnx + .rknn` | yellow_single_v1_phase2 (from phase1) | ✅ Found, rebased_verified |
| **OCR Special** | `special_LPRNet_fp16.rknn` | `artifacts/special_LPRNet.onnx + .rknn` | special_yellow_v5 (from yellow) | ✅ Found |

## Supporting Files on Board

| File | Workspace Source |
|------|-----------------|
| `yellow_keys.txt` | `keys/yellow_keys.txt` (69 chars) |
| `special_keys.txt` | `keys/special_keys.txt` |
| `ocr_keys_lprnet.txt` | Not in workspace (general OCR keyset) |

## Additional RKNN Models in Workspace (Not in Current Board Command)

| Model | Location | Notes |
|-------|----------|-------|
| `LPRNet_stage3_rk3568_fp16.rknn` | `experiments/first_board_baseline_v1/weights/` | Blue baseline, earlier version |
| `LPRNet_stage3_rk3568_fp16.rknn` | `experiments/tilt_ocr_obbwarp_v3_20260318/weights_stageC/` | Blue v3, intermediate |
| `LPRNet_stage3_rk3568_fp16.rknn` | `experiments/first_char_guard_v1/weights/` | Blue with first-char guard |
| `LPRNet_stage3_rk3568_fp16.rknn` | `experiments/tilt_ocr_obbwarp_v6_from_v5_balanced_20260319/weights_stageC/` | Blue v6, predecessor to v7 |
| `LPRNet_stage3_rk3568_fp16.rknn` | `experiments/tilt_ocr_obbwarp_v5_from_v3_hardcase10k_20260319/weights_stageC/` | Blue v5 |
| `Final_LPRNet_multihead_rk3568_fp16.rknn` | `experiments/green_h25/H25D_rear_from_pos2/` | Green multihead, not in current command |
| `LPRNet_e2_multihead_rk3568_fp16.rknn` | `experiments/green_e2_v4_a3000/` | Green E2 multihead, not in current command |
| `yellow_LPRNet_fp16.rknn` | `artifacts/` | Yellow v1, replaced by v5_phase2 |
| `yellow_LPRNet_v3_fp16.rknn` | `artifacts/` | Yellow v3, intermediate |
| `special_LPRNet_v2_fp16.rknn` | `artifacts/` | Special v2, not in current command (v1 is deployed) |

## Expert Archive Priority (Updated)

1. **Yellow Expert** (highest priority, most complete):
   - Training: yellow_single_v1_phase1 (rebased_verified) → phase2 (smoke_pass)
   - ONNX: artifacts/yellow_LPRNet_v5_phase2.onnx
   - RKNN: artifacts/yellow_LPRNet_v5_phase2_fp16.rknn
   - Keys: keys/yellow_keys.txt
   - Rebased manifest: manifests_rebased/yellow_train.csv

2. **Special Expert**:
   - Training: special_yellow_v5
   - ONNX: artifacts/special_LPRNet.onnx
   - RKNN: artifacts/special_LPRNet_fp16.rknn
   - Keys: keys/special_keys.txt

3. **Blue Expert** (multiple candidates, needs selection):
   - Primary: tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319
   - RKNN: LPRNet_stage3_rk3568_fp16_more_trained.rknn
   - Needs: identify exact training config matching deployed model

4. **Green Expert**:
   - Deployed model NOT in workspace: prov_deg_fp16_no_rknnpre.rknn
   - Possible source: province_degrade_train_v1 experiments
   - Needs: find training config + export script that produced this model

## Missing Items

- `prov_deg_fp16_no_rknnpre.rknn` and its source training experiment are not in workspace
- `ocr_keys_lprnet.txt` not found (board-only)
- The green expert training chain needs manual reconstruction
- Multihead models (h25, e2) are not the deployed green model
