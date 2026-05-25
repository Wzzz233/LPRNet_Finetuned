# yellow_single_v1_phase1 Rebased Manifest Smoke Test Result

## Summary

| Check | Status |
|-------|--------|
| Dataloader | ✅ 54,566 train + 1,420 test samples |
| Forward | ✅ Loss: 12.50 → 8.68 over 100 steps |
| Backward | ✅ gradient flow normal |
| Loss avg | 9.82 |
| max_steps=100 | ✅ properly stopped |
| Output | experiments/rebased_validation/yellow_single_v1_phase1/ |
| Old experiment | untouched |
| Overwrite risk | none |
| Checkpoints | none (save_interval > steps) |

## Verifications

1. ✅ Rebased manifest `manifests_rebased/yellow_train.csv` loaded correctly
2. ✅ `--dataset_root /home/wzzz/LPRNet` resolved all paths
3. ✅ Dataloader returned batches with correct shape
4. ✅ Forward pass through LPRNet succeeded
5. ✅ CTC loss computed
6. ✅ Backward pass (loss.backward + optimizer.step) succeeded
7. ✅ max_steps=100 stopped at global_iter=100
8. ✅ Output to new directory (no overwrite)
9. ✅ Old experiment untouched

## Rollback

To revert to old manifest training:
- Use `manifests/yellow_train.csv` (absolute paths)
- Do NOT pass `--dataset_root`
- Run from `cd src/training/ && python train_LPRNet.py`

## Next Step Suggestion

Run 1 epoch validation (remove --max_steps, keep --max_epoch 1):
```bash
cd /home/wzzz/LPRNet
python src/training/train_LPRNet.py \
  --dataset_root /home/wzzz/LPRNet \
  --pretrained_model '' \
  --keys_file keys/yellow_keys.txt \
  --data_mode manifest \
  --train_manifest manifests_rebased/yellow_train.csv \
  --test_manifest manifests_rebased/yellow_real_val.csv \
  --max_epoch 1 \
  --train_batch_size 4 \
  --num_workers 0 \
  --save_folder experiments/rebased_validation/yellow_single_v1_phase1/ \
  --head_mode single --lpr_max_len 8 --learning_rate 0.002
```