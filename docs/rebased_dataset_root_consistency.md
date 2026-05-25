# Rebased Manifest Dataset Root Consistency Audit

Generated: 2026-05-07
Active manifests: 303

## Critical Finding

The rebase stripped `/home/wzzz/LPRNet/` from absolute paths.
**For 302/303 manifests, the correct training dataset_root is:**
```
dataset_root = /home/wzzz/LPRNet   # project root
```
The plan's `detected_root` values (datasets, datasets/CCPD2020, etc.) are
**too specific** — they describe the first directory in the path, not the join root.

## Exception: green_edgefit_v3_allprov

Path = `green_edgefit_v3_allprov/images/...`, data at `datasets/green_edgefit_v3_allprov/`
Option 1: `dataset_root = /home/wzzz/LPRNet/datasets`  (works now)
Option 2: Create symlink -> `dataset_root = /home/wzzz/LPRNet`

## Plan Root vs Correct Root

- `datasets`: 243 manifests -> should be PROJECT_ROOT
- `datasets/green_exact_quad_synthetic_v1`: 28 manifests -> should be PROJECT_ROOT
- `datasets/CCPD2020`: 14 manifests -> should be PROJECT_ROOT
- `tmp`: 9 manifests -> should be PROJECT_ROOT
- `datasets/CCPD2019`: 4 manifests -> should be PROJECT_ROOT
- `datasets/CRPD_raw_ccpd_board_v1`: 3 manifests -> should be PROJECT_ROOT
- `datasets/green_edgefit_v3_allprov`: 1 manifests -> should be PROJECT_ROOT
- `datasets/CBLPRD-330k_v1`: 1 manifests -> should be PROJECT_ROOT

## Plan Root Needs Update

- manifests/ccpd2020_replace_extreme_v1/train_B2C_ccpd202: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/ccpd2020_replace_extreme_v1/val_B2C_ccpd2020_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/ccpd2020_replace_extreme_v2_additional/train_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/ccpd2020_replace_extreme_v3/train_extreme_v3.: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/ccpd2020_replace_extreme_v3/val_extreme_v3.cs: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/ccpd2020_replace_extreme_v4/train_extreme_v3.: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/ccpd2020_replace_extreme_v4/val_extreme_v3.cs: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/ccpd2020_replace_pose_v3/train_ccpd2020_repla: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/ccpd2020_replace_pose_v3/val_ccpd2020_replace: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/ccpd2020_replace_v1_obbquad/train_v1_obbquad.: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/ccpd2020_replace_v1_obbquad/val_v1_obbquad.cs: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/crpd_yellow_train_only.csv: plan=`datasets/CRPD_raw_ccpd_board_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/test_blue_hard.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/test_blue_simple.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/test_green_extreme.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/test_green_hard.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/test_green_real.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/test_green_simple.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/train_stageA.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/train_stageB.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/val.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/val_cblprd_blue.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/val_ccpd2019_blue.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/val_ccpd2020_green.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/val_crpd_blue.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3/val_nonccpd_green.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageE_e3_control/train_e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageE_e3_control/val_e3_con: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageE_e3_main/train_e3_main: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageE_e3_main/val_e3_main.c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageE_v1_extreme/train_E1.c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageE_v1_extreme/val_E1.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageE_v2_balanced/train_E2.: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageE_v2_balanced/val_E2.cs: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_redesign/proxy_stageA: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_redesign/proxy_stageA: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_redesign/proxy_stageA: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_redesign/train_stageA: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_redesign/val.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v2_foundation/proxy_b: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v2_foundation/proxy_g: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v2_foundation/proxy_g: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v2_foundation/proxy_s: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v2_foundation/train_s: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v2_foundation/val_sta: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/proxy_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/proxy_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/proxy_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/proxy_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/proxy_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/proxy_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/train_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/train_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/train_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/val_A0: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/val_A1: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stagea_v3_realprimary/val_A1: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccp: plan=`tmp` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_c: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccp: plan=`tmp` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccp: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`tmp` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lm: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`tmp` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axi: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`tmp` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`tmp` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_domina: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`tmp` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`tmp` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_obbq: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_obbq: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_soft: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_soft: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2D_paradigm3_prog: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2D_paradigm3_prog: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/trai: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/trai: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/trai: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/val_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/val_: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty/proxy_b: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty/proxy_b: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty/proxy_g: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty/proxy_g: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty/proxy_g: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty/proxy_g: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty/proxy_g: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty/proxy_s: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty/train_B: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty/val_B1A: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty_extreme: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty_extreme: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty_extreme: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty_extreme: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty_extreme: plan=`tmp` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty_extreme: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty_extreme: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty_extreme: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty_extreme: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_difficulty_extreme: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboa: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboa: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboa: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboa: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboa: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboa: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboa: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboa: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboa: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboa: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/firstchar_batch1/D1_firstchar_manifest_green8: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/firstchar_batch1/D2_firstchar_manifest_green8: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/firstchar_batch1/D3_firstchar_manifest_green8: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/firstchar_tiny_gray_alldata_v1/train.csv: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/firstchar_tiny_gray_green8only_v1/train.csv: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/province_degrade_train_v1/train_province_degr: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/province_stress_pose_val_v1/province_stress_p: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/special_test.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/special_train.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/subsets_e3_analysis_20260412/green8_test_all.: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/subsets_e3_analysis_20260412/green8_test_real: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/subsets_e3_analysis_20260412/green8_test_synt: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_e10_selfcheck_a50_replace.cs: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_firstchar_patch_dataset_v1.c: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_balance_lite_v1.csv: plan=`datasets/CCPD2019` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_balance_mid_v1.csv: plan=`datasets/CCPD2019` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_balance_round2_greenon: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e10a_boarddump_exact_t: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e10b_boarddump_overflo: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e11_e9c_append_aa0heav: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e12_e9c_append_boarddu: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e12_pose_replace_test.: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e12_replace_pose_v3_ap: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e13a_e9c_append_slotal: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e14a_image_local_probe: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e14a_image_local_probe: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e14a_image_local_probe: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e15a_prewarp_slot_prob: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e15a_prewarp_slot_prob: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e16a_nonanhui_ad_balan: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e16b_nonanhui_ad_balan: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e17a_cluster2_suffixba: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e17b_cluster3_transiti: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e17c_cluster3_tail_600: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e18b_su_bf_transition_: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e19c_su_bf_low_tail_de: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e20a_cluster2_beijing_: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e25a_stageB_full_reint: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e26_cluster3_tail_boos: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e27_cluster3_hardtail_: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e7_boardnative_v2.csv: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e8c_brightness_replace: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e9a_exact_template_5pr: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e9b_exact_template_all: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_e9c_exact_template_all: plan=`datasets/green_exact_quad_synthetic_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_green_edgefit_v3_allprov.csv: plan=`datasets/green_edgefit_v3_allprov` correct=`datasets/ (/home/wzzz/LPRNet/datasets)`
- manifests/unified_manifest_green_h34d_v3_three_tiers.cs: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_pos0_enhanced_v1_eval.csv: plan=`datasets/CCPD2019` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_pos0_enhanced_v1_train.csv: plan=`datasets/CCPD2020` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_v4_real_only_plain_plate_val: plan=`datasets/CBLPRD-330k_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/unified_manifest_v4_real_only_test_ccpd_board: plan=`datasets/CCPD2019` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/yellow_real_test.csv: plan=`datasets/CRPD_raw_ccpd_board_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/yellow_real_val.csv: plan=`datasets/CRPD_raw_ccpd_board_v1` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/yellow_single_hard_val.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/yellow_single_train.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/yellow_single_train_weighted.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/yellow_single_val.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/yellow_test.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/yellow_train.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`
- manifests/yellow_train_weighted.csv: plan=`datasets` correct=`PROJECT_ROOT (/home/wzzz/LPRNet)`

## All Results

| Manifest | Plan Root | Correct Root | Valid (project) | Valid (datasets) |
|---------|-----------|-------------|----------------|-----------------|
| manifests/ccpd2020_replace_extreme_v1/train_B | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/ccpd2020_replace_extreme_v1/val_B2C | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/ccpd2020_replace_extreme_v2_additio | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/ccpd2020_replace_extreme_v3/train_e | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/ccpd2020_replace_extreme_v3/val_ext | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/ccpd2020_replace_extreme_v4/train_e | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/ccpd2020_replace_extreme_v4/val_ext | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/ccpd2020_replace_pose_v3/train_ccpd | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/ccpd2020_replace_pose_v3/val_ccpd20 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/ccpd2020_replace_v1_obbquad/train_v | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/ccpd2020_replace_v1_obbquad/val_v1_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/crpd_yellow_train_only.csv | datasets/CRPD_raw_cc | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/curriculum_gray3/test_blue_hard.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/test_blue_simple.c | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/test_green_extreme | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/test_green_hard.cs | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/test_green_real.cs | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/test_green_simple. | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/train_stageA.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/train_stageB.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/val.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/val_cblprd_blue.cs | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/val_ccpd2019_blue. | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/val_ccpd2020_green | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/val_crpd_blue.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3/val_nonccpd_green. | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageE_e3_control/ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageE_e3_control/ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageE_e3_main/tra | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageE_e3_main/val | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageE_v1_extreme/ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageE_v1_extreme/ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageE_v2_balanced | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageE_v2_balanced | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_redesign/pr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_redesign/pr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_redesign/pr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_redesign/tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_redesign/va | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v2_foundati | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v2_foundati | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v2_foundati | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v2_foundati | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v2_foundati | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v2_foundati | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stagea_v3_realprim | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_ne | tmp | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ne | tmp | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | tmp | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | tmp | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | tmp | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | tmp | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | tmp | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | tmp | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2C_para | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2C_para | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2C_para | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2C_para | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2D_para | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2D_para | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2D_pose | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2D_pose | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2D_pose | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2D_pose | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_B2D_pose | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | tmp | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_difficul | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/firstchar_batch1/D1_firstchar_manif | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/firstchar_batch1/D2_firstchar_manif | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/firstchar_batch1/D3_firstchar_manif | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/firstchar_tiny_gray_alldata_v1/trai | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/firstchar_tiny_gray_green8only_v1/t | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/province_degrade_train_v1/train_pro | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/province_stress_pose_val_v1/provinc | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/special_test.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/special_train.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/subsets_e3_analysis_20260412/green8 | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/subsets_e3_analysis_20260412/green8 | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/subsets_e3_analysis_20260412/green8 | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_e10_selfcheck_a50_ | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_firstchar_patch_da | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_balance_lite | datasets/CCPD2019 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_balance_mid_ | datasets/CCPD2019 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_balance_roun | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e10a_boarddu | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e10b_boarddu | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e11_e9c_appe | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e12_e9c_appe | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e12_pose_rep | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e12_replace_ | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e13a_e9c_app | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e14a_image_l | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e14a_image_l | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e14a_image_l | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e15a_prewarp | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e15a_prewarp | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e16a_nonanhu | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e16b_nonanhu | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e17a_cluster | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e17b_cluster | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e17c_cluster | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e18b_su_bf_t | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e19c_su_bf_l | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e20a_cluster | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e25a_stageB_ | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e26_cluster3 | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e27_cluster3 | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e7_boardnati | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e8c_brightne | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e9a_exact_te | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e9b_exact_te | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_e9c_exact_te | datasets/green_exact | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_green_edgefit_v3_a | datasets/green_edgef | datasets/ (/home/wzzz/LPRNet/d | 0/20 | 20/20 |
| manifests/unified_manifest_green_h34d_v3_thre | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_official_gray3_blu | datasets/CCPD2020 | N/A | 0/20 | 0/20 |
| manifests/unified_manifest_pos0_enhanced_v1_e | datasets/CCPD2019 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_pos0_enhanced_v1_t | datasets/CCPD2020 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_v4_real_only_plain | datasets/CBLPRD-330k | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/unified_manifest_v4_real_only_test_ | datasets/CCPD2019 | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/yellow_real_test.csv | datasets/CRPD_raw_cc | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/yellow_real_val.csv | datasets/CRPD_raw_cc | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 20/20 |
| manifests/yellow_single_hard_val.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/yellow_single_train.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/yellow_single_train_weighted.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/yellow_single_val.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/yellow_test.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/yellow_train.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |
| manifests/yellow_train_weighted.csv | datasets | PROJECT_ROOT (/home/wzzz/LPRNe | 20/20 | 0/20 |

---