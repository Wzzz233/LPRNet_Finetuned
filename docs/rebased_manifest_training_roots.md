# Rebased Manifest Training Roots

Total entries: 336

## Path Contract Definition

| Contract | Meaning | Example Path | Joined With |
|---------|---------|-------------|-------------|
| `relative_to_project_root` | Path is relative to `/home/wzzz/LPRNet` | `datasets/CBLPRD-330k_v1/...` or `CCPD2020/...` | `/home/wzzz/LPRNet/` + path |
| `relative_to_datasets_root` | Path is relative to `/home/wzzz/LPRNet/datasets` | `green_edgefit_v3_allprov/...` | `/home/wzzz/LPRNet/datasets/` + path |

## Recommended training_dataset_root by Manifest Type

| Manifest Pattern | training_dataset_root | Count |
|-----------------|---------------------|-------|
| All except green_edgefit | `/home/wzzz/LPRNet` | 335 |
| green_edgefit_* | `/home/wzzz/LPRNet/datasets` | 1 |

## Key Manifest Details

### manifests/yellow_train.csv

- **detected_root**: `datasets`
- **training_dataset_root**: `/home/wzzz/LPRNet`
- **path_contract**: `relative_to_project_root`
- **validation**: passed (pass_rate=1.0)
- **sample path**: `datasets/CBLPRD-330k_v1/CBLPRD-330k/000390092.jpg`
- **joined path**: `/home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000390092.jpg`
- **exists**: True

### manifests/firstchar_tiny_gray_alldata_v1/train.csv

- **detected_root**: `datasets/CCPD2020`
- **training_dataset_root**: `/home/wzzz/LPRNet`
- **path_contract**: `relative_to_project_root`
- **validation**: passed (pass_rate=1.0)
- **sample path**: `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
- **joined path**: `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
- **exists**: True

### manifests/unified_manifest_green_edgefit_v3_allprov.csv

- **detected_root**: `datasets/green_edgefit_v3_allprov`
- **training_dataset_root**: `/home/wzzz/LPRNet/datasets`
- **path_contract**: `relative_to_datasets_root`
- **validation**: passed (pass_rate=1.0)
- **sample path**: `green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0000-京LFQ5792-12&7_244&13_244&71_8&63.jpg`
- **joined path**: `/home/wzzz/LPRNet/datasets/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0000-京LFQ5792-12&7_244&13_244&71_8&63.jpg`
- **exists**: True

### manifests/curriculum_gray3/val.csv

- **detected_root**: `datasets`
- **training_dataset_root**: `/home/wzzz/LPRNet`
- **path_contract**: `relative_to_project_root`
- **validation**: passed (pass_rate=1.0)
- **sample path**: `datasets/CRPD_all/CRPD_single/val/images/64_0049.jpg`
- **joined path**: `/home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0049.jpg`
- **exists**: True

## All Manifests

| # | Manifest | Detected Root | Training Root | Contract | Validation | Pass Rate |
|---|---------|-------------|-------------|---------|-----------|----------|
| 1 | manifests/unified_manifest_green_e12_pose_rep | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 2 | manifests/unified_manifest_green_e12_replace_ | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 3 | manifests/curriculum_gray3_stageE_e3_control/ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 4 | manifests/curriculum_gray3_stageE_e3_main/tra | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 5 | manifests/curriculum_gray3_stageb_v1_B2D_pose | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 6 | manifests/curriculum_gray3/val.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 7 | manifests/curriculum_gray3_stageE_e3_control/ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 8 | manifests/curriculum_gray3_stageE_e3_main/val | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 9 | manifests/curriculum_gray3_stagea_redesign/va | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 10 | manifests/curriculum_gray3_stageb_v1_B2D_pose | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 11 | manifests/firstchar_tiny_gray_alldata_v1/trai | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 12 | manifests/firstchar_tiny_gray_green8only_v1/t | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 13 | manifests/special_train.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 14 | manifests/yellow_single_train_weighted.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 15 | manifests/yellow_train.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 16 | manifests/yellow_train_weighted.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 17 | manifests/special_test.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 18 | manifests/yellow_real_val.csv | datasets/CRPD_raw_ccpd_bo | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 19 | manifests/yellow_single_val.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 20 | manifests/yellow_test.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 21 | manifests/unified_manifest_official_gray3_blu | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | partial | 0.0 |
| 22 | manifests/unified_manifest_pos0_enhanced_v1_t | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 23 | manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 24 | manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 25 | manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 26 | manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 27 | manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 28 | manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 29 | manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 30 | manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 31 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 32 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 33 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 34 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 35 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 36 | manifests/curriculum_gray3_stageb_v1_B2D_pose | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 37 | manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 38 | manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 39 | manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 40 | manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 41 | manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 42 | manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 43 | manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 44 | manifests/unified_manifest_green_e10a_boarddu | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 45 | manifests/unified_manifest_green_e10b_boarddu | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 46 | manifests/unified_manifest_green_e11_e9c_appe | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 47 | manifests/unified_manifest_green_e12_e9c_appe | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 48 | manifests/unified_manifest_green_e13a_e9c_app | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 49 | manifests/unified_manifest_green_e14a_image_l | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 50 | manifests/unified_manifest_green_e14a_image_l | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 51 | manifests/unified_manifest_green_e14a_image_l | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 52 | manifests/unified_manifest_green_e15a_prewarp | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 53 | manifests/unified_manifest_green_e15a_prewarp | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 54 | manifests/unified_manifest_green_e16a_nonanhu | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 55 | manifests/unified_manifest_green_e16b_nonanhu | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 56 | manifests/unified_manifest_green_e17a_cluster | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 57 | manifests/unified_manifest_green_e17b_cluster | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 58 | manifests/unified_manifest_green_e17c_cluster | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 59 | manifests/unified_manifest_green_e18b_su_bf_t | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 60 | manifests/unified_manifest_green_e19c_su_bf_l | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 61 | manifests/unified_manifest_green_e20a_cluster | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 62 | manifests/unified_manifest_green_e25a_stageB_ | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 63 | manifests/unified_manifest_green_e26_cluster3 | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 64 | manifests/unified_manifest_green_e27_cluster3 | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 65 | manifests/unified_manifest_green_e7_boardnati | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 66 | manifests/unified_manifest_green_e8c_brightne | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 67 | manifests/unified_manifest_green_e9a_exact_te | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 68 | manifests/unified_manifest_green_e9b_exact_te | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 69 | manifests/unified_manifest_green_e9c_exact_te | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 70 | manifests/unified_manifest_green_h34d_v3_thre | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 71 | manifests/unified_manifest_pos0_enhanced_v1_e | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 72 | manifests/unified_manifest_v4_real_only_plain | datasets/CBLPRD-330k_v1 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 73 | manifests/unified_manifest_v4_real_only_test_ | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 74 | manifests/curriculum_gray3/train_stageA.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 75 | manifests/curriculum_gray3/train_stageB.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 76 | manifests/curriculum_gray3_stageE_v1_extreme/ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 77 | manifests/curriculum_gray3_stageE_v2_balanced | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 78 | manifests/curriculum_gray3_stagea_redesign/tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 79 | manifests/curriculum_gray3_stagea_v2_foundati | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 80 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 81 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 82 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 83 | manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 84 | manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 85 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 86 | manifests/curriculum_gray3_stageb_v1_B2C_para | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 87 | manifests/curriculum_gray3_stageb_v1_B2C_para | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 88 | manifests/curriculum_gray3_stageb_v1_B2D_para | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 89 | manifests/curriculum_gray3_stageb_v1_B2D_pose | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 90 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 91 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 92 | manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 93 | manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 94 | manifests/unified_manifest_e10_selfcheck_a50_ | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 95 | manifests/unified_manifest_firstchar_patch_da | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 96 | manifests/unified_manifest_green_balance_lite | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 97 | manifests/unified_manifest_green_balance_mid_ | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 98 | manifests/unified_manifest_green_balance_roun | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 99 | manifests/curriculum_gray3/test_blue_hard.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 100 | manifests/curriculum_gray3/test_blue_simple.c | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 101 | manifests/curriculum_gray3/test_green_real.cs | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 102 | manifests/curriculum_gray3/test_green_simple. | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 103 | manifests/curriculum_gray3/val_cblprd_blue.cs | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 104 | manifests/curriculum_gray3/val_ccpd2019_blue. | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 105 | manifests/curriculum_gray3/val_crpd_blue.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 106 | manifests/curriculum_gray3/val_nonccpd_green. | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 107 | manifests/curriculum_gray3_stageE_v1_extreme/ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 108 | manifests/curriculum_gray3_stageE_v2_balanced | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 109 | manifests/curriculum_gray3_stagea_v2_foundati | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 110 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 111 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 112 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 113 | manifests/curriculum_gray3_stageb_v1_B1A_C_tr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 114 | manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 115 | manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 116 | manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 117 | manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 118 | manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 119 | manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 120 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 121 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 122 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 123 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 124 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 125 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 126 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 127 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 128 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 129 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 130 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 131 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 132 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 133 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 134 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 135 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 136 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 137 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 138 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 139 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 140 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 141 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 142 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 143 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 144 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 145 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 146 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 147 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 148 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 149 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 150 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 151 | manifests/curriculum_gray3_stageb_v1_B2C_para | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 152 | manifests/curriculum_gray3_stageb_v1_B2C_para | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 153 | manifests/curriculum_gray3_stageb_v1_B2D_para | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 154 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 155 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 156 | manifests/curriculum_gray3_stageb_v1_train_v4 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 157 | manifests/ccpd2020_replace_extreme_v1/train_B | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 158 | manifests/ccpd2020_replace_extreme_v2_additio | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 159 | manifests/ccpd2020_replace_pose_v3/train_ccpd | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 160 | manifests/ccpd2020_replace_v1_obbquad/train_v | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 161 | manifests/crpd_yellow_train_only.csv | datasets/CRPD_raw_ccpd_bo | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 162 | manifests/firstchar_batch1/D1_firstchar_manif | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 163 | manifests/firstchar_batch1/D2_firstchar_manif | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 164 | manifests/firstchar_batch1/D3_firstchar_manif | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 165 | manifests/province_degrade_train_v1/train_pro | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 166 | manifests/unified_manifest_green_edgefit_v3_a | datasets/green_edgefit_v3 | /home/wzzz/LPRNet/datasets | relative_to_datasets_root | passed | 1.0 |
| 167 | manifests/yellow_single_train.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 168 | manifests/curriculum_gray3/test_green_extreme | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 169 | manifests/curriculum_gray3/test_green_hard.cs | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 170 | manifests/curriculum_gray3_stagea_redesign/pr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 171 | manifests/curriculum_gray3_stagea_redesign/pr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 172 | manifests/curriculum_gray3_stagea_v2_foundati | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 173 | manifests/curriculum_gray3_stagea_v2_foundati | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 174 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 175 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 176 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 177 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 178 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 179 | manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 180 | manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 181 | manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 182 | manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 183 | manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 184 | manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 185 | manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 186 | manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 187 | manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 188 | manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 189 | manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 190 | manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 191 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 192 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 193 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 194 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 195 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 196 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 197 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 198 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 199 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 200 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 201 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 202 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 203 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 204 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 205 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 206 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 207 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 208 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 209 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 210 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 211 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 212 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 213 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 214 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 215 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 216 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 217 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 218 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 219 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 220 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 221 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 222 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 223 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 224 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 225 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 226 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 227 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 228 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 229 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 230 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 231 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 232 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 233 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 234 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 235 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 236 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 237 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 238 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 239 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 240 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 241 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 242 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 243 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 244 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 245 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 246 | manifests/province_stress_pose_val_v1/provinc | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 247 | manifests/subsets_e3_analysis_20260412/green8 | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 248 | manifests/subsets_e3_analysis_20260412/green8 | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 249 | manifests/subsets_e3_analysis_20260412/green8 | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 250 | manifests/curriculum_gray3/val_ccpd2020_green | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 251 | manifests/curriculum_gray3_stageb_v1_B1A_C_ne | tmp | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 252 | manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 253 | manifests/curriculum_gray3_stageb_v1_B1A_D_ex | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 254 | manifests/curriculum_gray3_stageb_v1_B1A_D_ne | tmp | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 255 | manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 256 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 257 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | tmp | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 258 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 259 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 260 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | tmp | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 261 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 262 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 263 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | tmp | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 264 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 265 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 266 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | tmp | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 267 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 268 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 269 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | tmp | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 270 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 271 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 272 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 273 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | tmp | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 274 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 275 | manifests/curriculum_gray3_stageb_v1_B2D_pose | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 276 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 277 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 278 | manifests/curriculum_gray3_stageb_v1_difficul | tmp | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 279 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 280 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 281 | manifests/ccpd2020_replace_extreme_v3/train_e | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 282 | manifests/ccpd2020_replace_extreme_v4/train_e | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 283 | manifests/curriculum_gray3_stagea_redesign/pr | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 284 | manifests/curriculum_gray3_stagea_v2_foundati | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 285 | manifests/curriculum_gray3_stagea_v2_foundati | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 286 | manifests/curriculum_gray3_stagea_v3_realprim | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 287 | manifests/curriculum_gray3_stageb_v1_B1A_C_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 288 | manifests/curriculum_gray3_stageb_v1_B1A_D_ne | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 289 | manifests/curriculum_gray3_stageb_v1_B1A_E1_m | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 290 | manifests/curriculum_gray3_stageb_v1_B1A_E6A_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 291 | manifests/curriculum_gray3_stageb_v1_B1A_E6B_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 292 | manifests/curriculum_gray3_stageb_v1_B1A_E6_a | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 293 | manifests/curriculum_gray3_stageb_v1_B1A_E6_b | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 294 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 295 | manifests/curriculum_gray3_stageb_v1_B1B_E6AB | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 296 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 297 | manifests/curriculum_gray3_stageb_v1_difficul | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 298 | manifests/ccpd2020_replace_extreme_v1/val_B2C | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 299 | manifests/ccpd2020_replace_extreme_v3/val_ext | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 300 | manifests/ccpd2020_replace_extreme_v4/val_ext | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 301 | manifests/ccpd2020_replace_pose_v3/val_ccpd20 | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 302 | manifests/ccpd2020_replace_v1_obbquad/val_v1_ | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 303 | manifests/yellow_real_test.csv | datasets/CRPD_raw_ccpd_bo | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 304 | manifests/yellow_single_hard_val.csv | datasets | /home/wzzz/LPRNet | relative_to_project_root | passed | 1.0 |
| 305 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 306 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 307 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 308 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 309 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 310 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 311 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 312 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 313 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 314 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 315 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 316 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 317 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 318 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 319 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 320 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 321 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 322 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 323 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 324 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 325 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 326 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 327 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 328 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 329 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 330 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 331 | manifests/Archive/unified_manifest_green_bala | datasets/CCPD2020 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 332 | manifests/Archive/unified_manifest_green_spec | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 333 | manifests/Archive/unified_manifest_green_spec | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 334 | manifests/Archive/unified_manifest_green_spec | datasets/green_exact_quad | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 335 | manifests/Archive/unified_manifest_v4_board_a | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |
| 336 | manifests/Archive/unified_manifest_v4_board_a | datasets/CCPD2019 | /home/wzzz/LPRNet | relative_to_project_root | archived | 0.0 |

---