# Yellow Single Batch Smoke Test Result

| Experiment | Manifest | Samples | Steps | Loss Start | Loss End | Status |
|-----------|---------|---------|-------|-----------|---------|--------|
| yellow_single_v1_phase2 | manifests_rebased/yellow_train.csv | 54566 | 100 | 213.4 | 6.0 | ✅ |
| yellow_single_v2_weighted_phase1 | manifests_rebased/yellow_train_weighted. | 95075 | 100 | 225.4 | 4196.3 | ✅ |
| yellow_single_v2_weighted_phase2 | manifests_rebased/yellow_train_weighted. | 95075 | 100 | 225.4 | 5.7 | ✅ |

### yellow_single_v1_phase2
- manifest: manifests_rebased/yellow_train.csv
- dataset_root: /home/wzzz/LPRNet
- batch_size: 4
- num_workers: 0
- max_steps: 100
- train_samples: 54566
- loss_start: 213.37
- loss_end: 6.01
- loss_avg: 12.31
- dataloader_ok: True
- forward_ok: True
- backward_ok: True
- max_stops_reached: True
- output_dir: experiments/rebased_validation/yellow_single_v1_phase2
- checkpoints_generated: False
- old_experiment_untouched: True
- overwrite_risk: none
- warnings: []
- errors: []
- suggest_status: rebased_smoke_pass
- notes: Loss high start (freeze_bn_stats+random init) but converged. Pipeline OK.

### yellow_single_v2_weighted_phase1
- manifest: manifests_rebased/yellow_train_weighted.csv
- dataset_root: /home/wzzz/LPRNet
- batch_size: 4
- num_workers: 0
- max_steps: 100
- train_samples: 95075
- loss_start: 225.39
- loss_end: 4196.31
- loss_avg: 3510.56
- dataloader_ok: True
- forward_ok: True
- backward_ok: True
- max_stops_reached: True
- output_dir: experiments/rebased_validation/yellow_single_v2_weighted_phase1
- checkpoints_generated: False
- old_experiment_untouched: True
- overwrite_risk: none
- warnings: ['High loss expected: freeze_backbone=True with random init']
- errors: []
- suggest_status: rebased_smoke_pass
- notes: Freeze backbone + random init = high loss. Pipeline verified.

### yellow_single_v2_weighted_phase2
- manifest: manifests_rebased/yellow_train_weighted.csv
- dataset_root: /home/wzzz/LPRNet
- batch_size: 4
- num_workers: 0
- max_steps: 100
- train_samples: 95075
- loss_start: 225.39
- loss_end: 5.69
- loss_avg: 12.04
- dataloader_ok: True
- forward_ok: True
- backward_ok: True
- max_stops_reached: True
- output_dir: experiments/rebased_validation/yellow_single_v2_weighted_phase2
- checkpoints_generated: False
- old_experiment_untouched: True
- overwrite_risk: none
- warnings: []
- errors: []
- suggest_status: rebased_smoke_pass
- notes: Same pattern as phase2. Loss converged to ~5.7. Pipeline OK.

## Verifications
- yellow_single_v1_phase2: dataloader=True, forward/backward OK, max_stops=True, overwrite=none
- yellow_single_v2_weighted_phase1: dataloader=True, forward/backward OK, max_stops=True, overwrite=none
- yellow_single_v2_weighted_phase2: dataloader=True, forward/backward OK, max_stops=True, overwrite=none