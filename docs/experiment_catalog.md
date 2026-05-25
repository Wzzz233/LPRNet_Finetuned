# Experiment Catalog

Total experiments: 271

## Summary

| Category | Count |
|----------|-------|
| Active (keep_active) | 177 |
| Keep reference | 87 |
| Failed candidate | 7 |
| Full reproducibility chain | 33 |
| Partial reproducibility | 126 |
| Weak reproducibility | 17 |
| Has rebased manifest match | 0 |
| Uses legacy absolute manifest | 238 |

## Active Experiments (priority for rebase migration)

| # | Experiment | Best Metric | Reproducibility | Manifest Type | Training Root |
|---|-----------|-------------|----------------|---------------|---------------|
| 1 | special_special_v2 | 0.9659173313995649 | partial | unknown | N/A |
| 2 | special_special_v1 | 0.9601160261058739 | partial | unknown | N/A |
| 3 | special_yellow_v5_phase2 | 0.9065727699530517 | partial | unknown | N/A |
| 4 | curriculum_gray3_stageB_v1_B1A_E1_modera | 0.8186666666666667 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 5 | curriculum_gray3_stageB_v1_B1A_E3_struct | 0.8183333333333334 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 6 | curriculum_gray3_stageB_v1_B1A_extreme_c | 0.8173333333333334 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 7 | curriculum_gray3_stageB_v1_B1A_A_train_v | 0.8166666666666667 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 8 | curriculum_gray3_stageB_v1_B1A_difficult | 0.8156666666666667 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 9 | curriculum_gray3_stageB_v1_B1A_C_train_v | 0.814 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 10 | curriculum_gray3_stageB_v1_B1B_E6AB_preb | 0.814 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 11 | curriculum_gray3_stageB2_A_softfreeze_ha | 0.8136666666666666 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 12 | curriculum_gray3_stageB_v1_B1A_E2_provan | 0.8136666666666666 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 13 | curriculum_gray3_stageB_v1_B1A_D_extreme | 0.8123333333333334 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 14 | curriculum_gray3_stageB_B2C_paradigm3_ob | 0.8113333333333334 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 15 | curriculum_gray3_stageB_B2D_paradigm3_pr | 0.8113333333333334 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 16 | curriculum_gray3_stageA_v3_realprimary_A | 0.8106666666666666 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 17 | curriculum_gray3_stageB2_A2_softfreeze_b | 0.8106666666666666 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 18 | curriculum_gray3_stageB_B2C_paradigm3_so | 0.8083333333333333 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 19 | curriculum_gray3_stageE_e3_control | 0.7863333333333333 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 20 | curriculum_gray3_stageE_v2_balanced | 0.784 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 21 | curriculum_gray3_stageE_e3_main | 0.7813333333333333 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 22 | curriculum_gray3_stageE_v1_extreme | 0.7736666666666666 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 23 | curriculum_gray3_pose_quad_training_v2 | 0.7646666666666667 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 24 | curriculum_gray3_stageA_v3_realprimary_A | 0.7336666666666667 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 25 | curriculum_gray3_stageA_v3_realprimary_A | 0.7253333333333334 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 26 | curriculum_gray3_stageA_v3_realprimary_A | 0.7153333333333334 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 27 | curriculum_gray3_stageA_v3_realprimary_A | 0.693 | full | relative_to_project_root | /home/wzzz/LPRNet |
| 28 | yellow_single_v2_weighted_phase2 | 0.671830985915493 | unknown | unknown | N/A |
| 29 | special_yellow_v3 | 0.6645177926766375 | partial | unknown | N/A |
| 30 | special_yellow_v2 | 0.6562661165549252 | partial | unknown | N/A |

## Experiments Referencing Rebasing Manifest

- curriculum_gray3_stageB_v1_B1A_E1_modera: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_E3_struct: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_extreme_c: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_A_train_v: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_difficult: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_C_train_v: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1B_E6AB_preb: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB2_A_softfreeze_ha: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_E2_provan: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_D_extreme: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_B2C_paradigm3_ob: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_B2D_paradigm3_pr: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageA_v3_realprimary_A: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB2_A2_softfreeze_b: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_B2C_paradigm3_so: rebased=manifests_rebased/curriculum_gray3_stage, root=/home/wzzz/LPRNet

## Full Reproducibility Experiments (33)

- curriculum_gray3_stageB_v1_B1A_E1_moderate_lm: metric=0.8186666666666667, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_E3_structural_: metric=0.8183333333333334, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_extreme_ccpdbo: metric=0.8173333333333334, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_A_train_v4e3_c: metric=0.8166666666666667, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_difficulty_con: metric=0.8156666666666667, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_C_train_v4e3_c: metric=0.814, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1B_E6AB_preblur_v: metric=0.814, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB2_A_softfreeze_hard_br: metric=0.8136666666666666, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_E2_provanchor_: metric=0.8136666666666666, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_v1_B1A_D_extreme900_v: metric=0.8123333333333334, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_B2C_paradigm3_obbquad: metric=0.8113333333333334, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_B2D_paradigm3_progres: metric=0.8113333333333334, root=/home/wzzz/LPRNet
- curriculum_gray3_stageA_v3_realprimary_A1D_gr: metric=0.8106666666666666, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB2_A2_softfreeze_backbo: metric=0.8106666666666666, root=/home/wzzz/LPRNet
- curriculum_gray3_stageB_B2C_paradigm3_softfre: metric=0.8083333333333333, root=/home/wzzz/LPRNet

## Legacy Manifest Experiments

- special_special_v2: manifests=['../../manifests/special_train.csv', '../../manifests/special_test.csv']
- special_special_v1: manifests=['../../manifests/special_train.csv', '../../manifests/special_test.csv']
- special_yellow_v5_phase2: manifests=['../../manifests/yellow_single_val.csv', '../../manifests/yellow_single_train_weighted.csv']
- yellow_single_v2_weighted_phase2: manifests=['../../manifests/yellow_real_val.csv', '../../manifests/yellow_train_weighted.csv']
- special_yellow_v3: manifests=['../../manifests/yellow_test.csv', '../../manifests/yellow_train.csv']
- special_yellow_v2: manifests=['../../manifests/yellow_test.csv', '../../manifests/yellow_train.csv']
- special_yellow_v5: manifests=['../../manifests/yellow_single_val.csv', '../../manifests/yellow_single_train_weighted.csv']
- H35D_edgefit_tier3_classbalanced_firstchar: manifests=[]
- G1_pos0_green8_only: manifests=[]
- green_e8b_gray3_brightness: manifests=[]
- H36A_pos0head_arch_only: manifests=[]
- H29A_anhui40_exact: manifests=[]
- H34B_edgefit_allprov_v2_zhe_guard: manifests=[]
- H32B_classbalanced_plus_focalctc: manifests=[]
- H35B_h32a_focalctc: manifests=[]