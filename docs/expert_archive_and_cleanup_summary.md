# Expert Archive and Cleanup Summary

## 1. Expert Candidates Overview

| Expert Type | Candidates | Primary | Board Artifacts Found | Status |
|------------|-----------|---------|----------------------|--------|
| Blue Plate | ~56 (mostly ccpd/official) | 47 auto-tagged | 3 RKNN (stage1, guard, v7) | 🟡 Needs manual selection (too many auto-tagged) |
| Green Plate | 1+ (green_h25, green_e2) | 2 tagged | 2 RKNN multihead (h25, e2) | 🟡 Needs manual selection |
| Yellow (from Blue) | 4 known | yellow_single family | 3 ONNX + 3 RKNN in artifacts/ | ✅ Clear lineage |
| Special Plate | 3 known | special_yellow_v5 v3 v2 | 2 ONNX + 2 RKNN in artifacts/ | ✅ Clear lineage |

## 2. Board Artifacts Available

| Artifact | Source | Exists |
|----------|--------|--------|
| `experiments/first_board_baseline_v1/weights/LPRNet_stage3_rk3568_fp16.rknn` | Blue OCR baseline | ✅ |
| `experiments/first_char_guard_v1/weights/LPRNet_stage3_rk3568_fp16.rknn` | Blue first-char guard | ✅ |
| `experiments/tilt_ocr_obbwarp_v7/.../LPRNet_stage3_rk3568_fp16_more_trained.rknn` | Blue tilt v7 | ✅ |
| `experiments/green_h25/H25D_rear_from_pos2/Final_LPRNet_multihead_rk3568_fp16.rknn` | Green multihead | ✅ |
| `experiments/green_e2_v4_a3000/LPRNet_e2_multihead_rk3568_fp16.rknn` | Green multihead e2 | ✅ |
| `artifacts/yellow_LPRNet_v5_phase2.onnx + .rknn` | Yellow phase2 | ✅ |
| `artifacts/yellow_LPRNet_v3.onnx + .rknn` | Yellow v3 | ✅ |
| `artifacts/special_LPRNet_v2.onnx + .rknn` | Special v2 | ✅ |

## 3. Dataset Overview

| Dataset | Size | Files | Category |
|---------|------|-------|----------|
| CCPD2019 | 25G | 355K | keep_expert_core (blue training) |
| CRPD_all | 19G | 67K | keep_expert_core (multi-purpose) |
| CBLPRD-330k_v1 | 2G | 654K | keep_expert_core (yellow/synth) |
| CCPD2020 | 905M | 12K | keep_expert_core (green training) |
| green_exact_quad_synthetic_v1 | 2.2G | 11K | keep_expert_core (green) |
| green_edgefit_* (8 variants) | ~700M | ~85K total | keep_expert_core (green edgefit) |
| CRPD_raw_ccpd_board_v1 | 176M | 43K | keep_board_deploy |
| plate_true_quad_pose | 3.1G | 812K | keep_legacy (pose training) |
| province_degrade_train_v1 | 1.7G | 10K | keep_active_reference |
| Various small QA/probe datasets | <200M | varying | cleanup_candidate or manual_review |

## 4. Manifest Classification

| Category | Count | Description |
|----------|-------|-------------|
| keep_expert_core | ~60 | Manifests directly used by expert experiments |
| keep_board_deploy_reference | ~30 | Manifests referenced by board deployment/testing |
| keep_active_reference | ~200 | Active experiments, rebased_verified, rebased_smoke_pass |
| keep_legacy_reference | ~60 | Historical but no active dependency |
| cleanup_candidate | ~50 | No active/recent reference in catalog/scripts/configs |
| manual_review | ~40 | Unclear or dynamic references |

## 5. Recommendations

1. **Expert archive build**: 4 model_zoo packages (blue/green/yellow/special)
   - Yellow: best documented, use yellow_single_v1_phase1 as template
   - Special: special_yellow_v5 as primary
   - Blue: needs manual selection (first_board_baseline_v1 as starting point)
   - Green: needs manual selection (green_h25 multihead as starting point)

2. **Manifest cleanup**: Only after expert archives are built
   - Move ~50 cleanup_candidate manifests to tmp/cleanup_candidates/
   - Never delete, only move

3. **Dataset cleanup**: Only after everything else is confirmed
   - No dataset moves yet (too risky)
   - Small QA/probe datasets (<50MB) are first candidates

4. **Prohibited**: No deletions, no dataset moves, no experiment archiving
   - All cleanup actions are dry-run until manually confirmed

## 6. Expert Archive Structure (Proposed)

```
model_zoo/
├── yellow_from_blue_expert/
│   ├── yellow_single_v1_phase1/ (rebased_verified)
│   ├── yellow_single_v1_phase2/ (rebased_smoke_pass)
│   ├── yellow_single_v2_weighted_phase1/ (rebased_smoke_pass)
│   ├── yellow_single_v2_weighted_phase2/ (rebased_smoke_pass)
│   ├── artifacts/ (ONNX + RKNN)
│   ├── lineage.json
│   └── README.md
├── special_plate_expert/
│   ├── special_yellow_v5/
│   ├── artifacts/ (ONNX + RKNN)
│   └── ...
├── blue_plate_expert/  [needs manual selection]
├── green_plate_expert/ [needs manual selection]
```

## 7. Files Generated

| File | Description |
|------|-------------|
| `expert_asset_candidates.json` | 57 expert candidates identified |
| `expert_archive_plan.json` | Archive structure plan for 4 expert types |
| `docs/expert_archive_and_cleanup_summary.md` | This summary |
