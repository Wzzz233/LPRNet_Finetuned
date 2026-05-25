# Rebased Config Migration Suggestions

Total eligible experiments: 40

## Picked Experiments

| # | Experiment | Old Manifest | New Rebasing Manifest | Dataset Root |
|---|-----------|-------------|----------------------|-------------|
| 1 | special_special_v2 | ../../manifests/special_train. | manifests_rebased/special_train.csv | /home/wzzz/LPRNet |
| 2 | special_special_v1 | ../../manifests/special_train. | manifests_rebased/special_train.csv | /home/wzzz/LPRNet |
| 3 | special_yellow_v5_phase2 | ../../manifests/yellow_single_ | manifests_rebased/yellow_single_train_we | /home/wzzz/LPRNet |
| 4 | curriculum_gray3_stageB_v1_B1A_E1_modera | manifests/curriculum_gray3_sta | manifests_rebased/curriculum_gray3_stage | /home/wzzz/LPRNet |
| 5 | curriculum_gray3_stageB_v1_B1A_E3_struct | manifests/curriculum_gray3_sta | manifests_rebased/curriculum_gray3_stage | /home/wzzz/LPRNet |

## Migration Steps

### special_special_v2

1. **Old command**: `cd src/training && python train_LPRNet.py (reference old script)`
2. **Old manifest**: `../../manifests/special_train.csv`
3. **New rebased manifest**: `manifests_rebased/special_train.csv`
4. **Show test (100 steps)**:
```bash
cd /home/wzzz/LPRNet && python src/training/train_LPRNet.py --dataset_root /home/wzzz/LPRNet --pretrained_model '' --train_manifest manifests_rebased/special_train.csv --test_manifest manifests_rebased/yellow_real_val.csv --max_epoch 1 --max_steps 100 --train_batch_size 4 --num_workers 0 --save_folder experiments/rebased_rerun/special_special_v2_rebased_smoke/
```
5. **If smoke test pass, run validation**:
```bash
cd /home/wzzz/LPRNet
python src/training/train_LPRNet.py \
  --dataset_root /home/wzzz/LPRNet \
  --train_manifest manifests_rebased/special_train.csv \
  --test_manifest manifests_rebased/yellow_real_val.csv \
  --pretrained_model '' \
  --max_epoch 1 --max_steps 100 \
  --train_batch_size 4 --num_workers 0 \
  --save_folder experiments/rebased_rerun/special_special_v2_rebased/
```
6. **Risk**: low
7. **Rollback**: `Output in experiments/rebased_rerun/, old experiment untouched.`

### special_special_v1

1. **Old command**: `cd src/training && python train_LPRNet.py (reference old script)`
2. **Old manifest**: `../../manifests/special_train.csv`
3. **New rebased manifest**: `manifests_rebased/special_train.csv`
4. **Show test (100 steps)**:
```bash
cd /home/wzzz/LPRNet && python src/training/train_LPRNet.py --dataset_root /home/wzzz/LPRNet --pretrained_model '' --train_manifest manifests_rebased/special_train.csv --test_manifest manifests_rebased/yellow_real_val.csv --max_epoch 1 --max_steps 100 --train_batch_size 4 --num_workers 0 --save_folder experiments/rebased_rerun/special_special_v1_rebased_smoke/
```
5. **If smoke test pass, run validation**:
```bash
cd /home/wzzz/LPRNet
python src/training/train_LPRNet.py \
  --dataset_root /home/wzzz/LPRNet \
  --train_manifest manifests_rebased/special_train.csv \
  --test_manifest manifests_rebased/yellow_real_val.csv \
  --pretrained_model '' \
  --max_epoch 1 --max_steps 100 \
  --train_batch_size 4 --num_workers 0 \
  --save_folder experiments/rebased_rerun/special_special_v1_rebased/
```
6. **Risk**: low
7. **Rollback**: `Output in experiments/rebased_rerun/, old experiment untouched.`

### special_yellow_v5_phase2

1. **Old command**: `cd src/training && python train_LPRNet.py (reference old script)`
2. **Old manifest**: `../../manifests/yellow_single_train_weighted.csv`
3. **New rebased manifest**: `manifests_rebased/yellow_single_train_weighted.csv`
4. **Show test (100 steps)**:
```bash
cd /home/wzzz/LPRNet && python src/training/train_LPRNet.py --dataset_root /home/wzzz/LPRNet --pretrained_model '' --train_manifest manifests_rebased/yellow_single_train_weighted.csv --test_manifest manifests_rebased/yellow_real_val.csv --max_epoch 1 --max_steps 100 --train_batch_size 4 --num_workers 0 --save_folder experiments/rebased_rerun/special_yellow_v5_phase2_rebased_smoke/
```
5. **If smoke test pass, run validation**:
```bash
cd /home/wzzz/LPRNet
python src/training/train_LPRNet.py \
  --dataset_root /home/wzzz/LPRNet \
  --train_manifest manifests_rebased/yellow_single_train_weighted.csv \
  --test_manifest manifests_rebased/yellow_real_val.csv \
  --pretrained_model '' \
  --max_epoch 1 --max_steps 100 \
  --train_batch_size 4 --num_workers 0 \
  --save_folder experiments/rebased_rerun/special_yellow_v5_phase2_rebased/
```
6. **Risk**: low
7. **Rollback**: `Output in experiments/rebased_rerun/, old experiment untouched.`

### curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original

1. **Old command**: `cd src/training && python train_LPRNet.py (reference old script)`
2. **Old manifest**: `manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv`
3. **New rebased manifest**: `manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv`
4. **Show test (100 steps)**:
```bash
cd /home/wzzz/LPRNet && python src/training/train_LPRNet.py --dataset_root /home/wzzz/LPRNet --pretrained_model '' --train_manifest manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv --test_manifest manifests_rebased/yellow_real_val.csv --max_epoch 1 --max_steps 100 --train_batch_size 4 --num_workers 0 --save_folder experiments/rebased_rerun/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original_rebased_smoke/
```
5. **If smoke test pass, run validation**:
```bash
cd /home/wzzz/LPRNet
python src/training/train_LPRNet.py \
  --dataset_root /home/wzzz/LPRNet \
  --train_manifest manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv \
  --test_manifest manifests_rebased/yellow_real_val.csv \
  --pretrained_model '' \
  --max_epoch 1 --max_steps 100 \
  --train_batch_size 4 --num_workers 0 \
  --save_folder experiments/rebased_rerun/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original_rebased/
```
6. **Risk**: low
7. **Rollback**: `Output in experiments/rebased_rerun/, old experiment untouched.`

### curriculum_gray3_stageB_v1_B1A_E3_structural_anchor_slot_lmh_ccpdboard_eval_original

1. **Old command**: `cd src/training && python train_LPRNet.py (reference old script)`
2. **Old manifest**: `manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv`
3. **New rebased manifest**: `manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv`
4. **Show test (100 steps)**:
```bash
cd /home/wzzz/LPRNet && python src/training/train_LPRNet.py --dataset_root /home/wzzz/LPRNet --pretrained_model '' --train_manifest manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv --test_manifest manifests_rebased/yellow_real_val.csv --max_epoch 1 --max_steps 100 --train_batch_size 4 --num_workers 0 --save_folder experiments/rebased_rerun/curriculum_gray3_stageB_v1_B1A_E3_structural_anchor_slot_lmh_ccpdboard_eval_original_rebased_smoke/
```
5. **If smoke test pass, run validation**:
```bash
cd /home/wzzz/LPRNet
python src/training/train_LPRNet.py \
  --dataset_root /home/wzzz/LPRNet \
  --train_manifest manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv \
  --test_manifest manifests_rebased/yellow_real_val.csv \
  --pretrained_model '' \
  --max_epoch 1 --max_steps 100 \
  --train_batch_size 4 --num_workers 0 \
  --save_folder experiments/rebased_rerun/curriculum_gray3_stageB_v1_B1A_E3_structural_anchor_slot_lmh_ccpdboard_eval_original_rebased/
```
6. **Risk**: low
7. **Rollback**: `Output in experiments/rebased_rerun/, old experiment untouched.`

## Safety Rules

1. Do NOT modify old config scripts.
2. Output must go to experiments/rebased_rerun/ or experiments/rebased_validation/.
3. Must include --max_steps 100 in smoke test.
4. Old experiment directory untouched.
5. No overwrite risk.
