# Yellow Single Rebased Batch Plan

Based on verified experiment: yellow_single_v1_phase1 (rebased_verified)

## Batch Plan

| # | Experiment | Old Manifest | Rebased Manifest | Dataset Root | Output Dir |
|---|-----------|-------------|-----------------|-------------|-----------|
| 1 | experiments/yellow_single_v1_phase2 | manifests/yellow_train.csv | manifests_rebased/yellow_train.csv | /home/wzzz/LPRNet | experiments/rebased_validation/yellow_single_v1_phase2 |
| 2 | experiments/yellow_single_v2_weight | manifests/yellow_train_weighte | manifests_rebased/yellow_train_weig | /home/wzzz/LPRNet | experiments/rebased_validation/yellow_single_v2_weighted_phase1 |
| 3 | experiments/yellow_single_v2_weight | manifests/yellow_train_weighte | manifests_rebased/yellow_train_weig | /home/wzzz/LPRNet | experiments/rebased_validation/yellow_single_v2_weighted_phase2 |

## Smoke Test Commands

### experiments/yellow_single_v1_phase2/

```bash
cd /home/wzzz/LPRNet && python src/training/train_LPRNet.py --dataset_root /home/wzzz/LPRNet --pretrained_model '' --keys_file keys/yellow_keys.txt --data_mode manifest --train_manifest manifests_rebased/yellow_train.csv --test_manifest manifests_rebased/yellow_real_val.csv --learning_rate 0.0005 --lr_schedule 5 10 15 --max_epoch 1 --max_steps 100 --train_batch_size 4 --num_workers 0 --save_folder experiments/rebased_validation/yellow_single_v1_phase2/ --head_mode single --lpr_max_len 8 --freeze_bn_stats True --cuda True
```

- Expected risk: low - same manifest and root as verified phase1
- Rollback: Use manifests/yellow_train.csv without --dataset_root

### experiments/yellow_single_v2_weighted_phase1/

```bash
cd /home/wzzz/LPRNet && python src/training/train_LPRNet.py --dataset_root /home/wzzz/LPRNet --pretrained_model '' --keys_file keys/yellow_keys.txt --data_mode manifest --train_manifest manifests_rebased/yellow_train_weighted.csv --test_manifest manifests_rebased/yellow_real_val.csv --learning_rate 0.002 --lr_schedule 3 6 --max_epoch 1 --max_steps 100 --train_batch_size 4 --num_workers 0 --save_folder experiments/rebased_validation/yellow_single_v2_weighted_phase1/ --head_mode single --lpr_max_len 8 --freeze_backbone True --cuda True
```

- Expected risk: low - uses weighted manifest, same root /home/wzzz/LPRNet
- Rollback: Use manifests/yellow_train_weighted.csv without --dataset_root

### experiments/yellow_single_v2_weighted_phase2/

```bash
cd /home/wzzz/LPRNet && python src/training/train_LPRNet.py --dataset_root /home/wzzz/LPRNet --pretrained_model '' --keys_file keys/yellow_keys.txt --data_mode manifest --train_manifest manifests_rebased/yellow_train_weighted.csv --test_manifest manifests_rebased/yellow_real_val.csv --learning_rate 0.0005 --lr_schedule 5 10 15 --max_epoch 1 --max_steps 100 --train_batch_size 4 --num_workers 0 --save_folder experiments/rebased_validation/yellow_single_v2_weighted_phase2/ --head_mode single --lpr_max_len 8 --freeze_bn_stats True --cuda True
```

- Expected risk: low - same pattern as phase1
- Rollback: Use manifests/yellow_train_weighted.csv without --dataset_root
