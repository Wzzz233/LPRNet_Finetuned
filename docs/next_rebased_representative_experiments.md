# Next Rebased Representative Experiments

## Based on verified base: yellow_single_v1_phase1

The following experiments represent different path types and should be validated next.

| # | Experiment | Reason | Manifest | Dataset Root | Suggested Step |
|---|-----------|--------|---------|-------------|----------------|
| 1 | curriculum_gray3_stageA_v3_realprimary_A | First multi-stage curriculum experiment. Tests mul | manifests_rebased/curriculum_gray3/ | /home/wzzz/LPRNet | 100-step smoke test |
| 2 | green_e12_pose_replace_append_stage2 | First green experiment in the E series. Different  | manifests_rebased/unified_manifest_ | /home/wzzz/LPRNet | 100-step smoke test |

## Details

### curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_aux

- **Reason**: First multi-stage curriculum experiment. Tests multi-manifest training (stageA + val). Different training pattern than yellow_single.
- **Manifest**: `manifests_rebased/curriculum_gray3/train_stageA.csv`
- **Test manifest**: `manifests_rebased/curriculum_gray3/val.csv`
- **Dataset root**: `/home/wzzz/LPRNet`
- **Risk**: medium - uses multiple manifests, different OCR preprocessing (gray3)
- **Suggested first step**: 100-step smoke test
- **Notes**: Verify that curriculum split_filter logic works; OCR crop_mode may differ.

### green_e12_pose_replace_append_stage2

- **Reason**: First green experiment in the E series. Different manifest structure (mixed real+synth).
- **Manifest**: `manifests_rebased/unified_manifest_green_e12_replace_pose_v3_append.csv`
- **Test manifest**: `manifests_rebased/yellow_real_val.csv (fallback)`
- **Dataset root**: `/home/wzzz/LPRNet`
- **Risk**: medium - green dataset with ccpd_board OCR preproc
- **Suggested first step**: 100-step smoke test
- **Notes**: Check that green8 plate types load correctly with ccpd_board preprocessing.

## Validation Order

1. `curriculum_gray3_stageA` — different training paradigm (curriculum, gray3)
2. Active green_edgefit experiment — special dataset_root edge case
3. Green E series — different OCR preprocessing (ccpd_board)
