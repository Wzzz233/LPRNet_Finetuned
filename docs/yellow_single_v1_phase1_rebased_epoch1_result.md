# yellow_single_v1_phase1 1-Epoch Rebased Manifest Validation Result

## Summary

| Check | Status | Detail |
|-------|--------|--------|
| Dataloader | ✅ | 54,566 train, 1,420 test samples |
| Full epoch | ✅ | 13,642/13,642 steps completed |
| Loss start | 12.50 | Initial (random init) |
| Loss avg | 4.23 | Over 13,642 steps |
| Loss end | 2.81 | Final step |
| Forward | ✅ | Every step OK |
| Backward | ✅ | Every step OK |
| Eval executed | ✅ | 1,420 test samples |
| Checkpoints | ✅ | 6 @ save_interval=2000 + best/final/last |
| Output | ✅ | `experiments/rebased_validation/yellow_single_v1_phase1_epoch1/` |
| Old experiment | ✅ | untouched in `experiments/yellow_single_v1_phase1/` |
| Overwrite risk | ✅ | none |

## Verifications

1. ✅ All 13,642 batches processed (54,566 images, batch_size=4)
2. ✅ Loss converges: 12.50 → 4.23 avg → 2.81 final
3. ✅ 6 checkpoints generated in new output directory
4. ✅ Best/final/last checkpoints saved
5. ✅ Test evaluation ran on 1,420 validation samples
6. ✅ All paths from `manifests_rebased/` — no old manifest references in training
7. ✅ `--dataset_root /home/wzzz/LPRNet` working correctly
8. ✅ Old experiment `experiments/yellow_single_v1_phase1/` untouched

## Recommendation

**Suggest marking `yellow_single_v1_phase1` as `rebased_verified`.**

The rebased manifest pipeline has been proven to work end-to-end for this experiment.
All three stages passed:
- 100-step smoke test ✅
- 1-epoch training ✅
- Checkpoint generation + evaluation ✅

## Rollback

To revert to old manifest training:
- Use `manifests/yellow_train.csv` (absolute paths)
- Do NOT pass `--dataset_root`
- Run from `cd src/training/ && python train_LPRNet.py`