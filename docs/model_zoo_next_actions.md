# Model Zoo Next Actions

## Current State After Completion

| Expert | README | lineage | Keys | Manifest | Checkpoint | Board Artifact | Status |
|-------|--------|---------|------|----------|------------|----------------|--------|
| yellow_from_blue_expert | ✅ | ✅ | ✅ yellow_keys.txt | ✅ train+val | ⬜ ref | ⬜ ref | 🟡 archive_partial |
| special_plate_expert | ✅ | ✅ | ✅ special_keys.txt | ✅ train+test | ⬜ ref | ⬜ ref | 🟡 archive_partial |
| blue_plate_expert | ✅ | ✅ | ✅ (default CHARS) | ✅ ref notes | ⬜ ref | ✅ ONNX+RKNN ref | 🟡 archive_partial |
| green_plate_expert | ✅ | ✅ | ✅ (default CHARS) | ✅ rebased | ⬜ ref | ✅ ONNX+RKNN ref | 🟡 archive_partial |

## Copy Checkpoints

```bash
python tools/build_expert_archives.py --execute --copy-checkpoints --copy-board-artifacts
```
This will add ~43 MB (18 MB checkpoints + 25 MB ONNX/RKNN) to model_zoo/.

## Remaining Gaps

1. **Blue expert**: Needs rebased CSV manifest identification. Currently uses old-style txt manifests.
2. **Green expert**: Verify prov_deg_fp16_no_rknnpre.rknn export command matches the deployed behavior.
3. **Both yellow/special**: Only missing physical copies of checkpoints and artifacts.

## Don't Do Yet

- Manifest cleanup: NOT ready (blue expert still needs manifest identification)
- Dataset cleanup: NOT ready (keep all datasets)
- Experiment archiving: NOT ready (green expert still active)

## Verified Complete Path (yellow trace example)

```
Training: yellow_single_v1_phase1 (rebased_verified) -> phase2
  -> manifests_rebased/yellow_train.csv (dataset_root=/home/wzzz/LPRNet)
  -> Export: ONNX -> artifacts/yellow_LPRNet_v5_phase2.onnx
  -> RKNN:  -> artifacts/yellow_LPRNet_v5_phase2_fp16.rknn
  -> Board: /userdata/model/yellow_LPRNet_v5_phase2_fp16.rknn
```