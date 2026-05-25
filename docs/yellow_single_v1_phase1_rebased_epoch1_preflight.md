# Preflight Check: yellow_single_v1_phase1 1-Epoch Validation

Script: configs/rebased_validation/yellow_single_v1_phase1_rebased_epoch1.sh

## Results

| # | Check | Status |
|---|-------|--------|
| 1 | Script file exists | ✅ |
| 2 | Uses manifests_rebased/ in training command | ✅ |
| 3 | No old manifest in command body | ❌ |
| 4 | Has --dataset_root /home/wzzz/LPRNet | ✅ |
| 5 | No --max_steps (full epoch) | ✅ |
| 6 | Has --max_epoch 1 | ✅ |
| 7 | train_batch_size=4 | ✅ |
| 8 | num_workers=0 | ✅ |
| 9 | Output to new epoch1 dir | ✅ |
| 10 | Does NOT output to smoke test dir | ✅ |
| 11 | No rm/mv/cp commands | ✅ |
| 12 | Has rollback notes | ✅ |
| 13 | --pretrained_model is empty | ✅ |
| 14 | Train manifest exists | ✅ |
| 15 | Test manifest exists | ✅ |
| 16 | New epoch1 output dir does not exist yet | ✅ |
| 17 | Old smoke test dir still exists (not overwritten) | ✅ |
| 18 | Original experiment untouched | ✅ |
| 19 | Train manifest 20-sample read (20/20 ok) | ✅ |
| 20 | Test manifest 20-sample read (20/20 ok) | ✅ |

## Summary

- Passed: 19/20
- Failed: 1/20
- Can proceed: NO