# Manifest 路径依赖审计报告

生成时间: 2026-05-07 14:34:46

扫描路径: /home/wzzz/LPRNet

---

## 总览

| 指标 | 值 |
|------|-----|
| 扫描 manifest 数 | 336 |
| 高风险 | 0 |
| 中风险 | 2 |
| 低风险 | 334 |
| 绝对路径总数 | 0 |
| 无效路径总数 | 3811 |
| 采样样本总行数 | 711269 |

## 每份 manifest 详情

| 文件 | 类型 | 大小 | 绝对路径 | 相对路径 | 裸文件名 | 有效 | 无效 | 风险 |
|------|------|------|---------|---------|---------|------|------|------|
| manifests_rebased/manifests_rebased/unified_manifest_green_edgefit_v3_allprov.csv | csv | 3.5M | 0 | 3331 | 0 | 0 | 3331 | medium |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_specialist_official_v1_balanced_train.csv | csv | 26.6M | 0 | 5581 | 0 | 5101 | 480 | medium |
| manifests_rebased/manifests_rebased/yellow_train_weighted.csv | csv | 18.7M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/yellow_train.csv | csv | 6.9M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/yellow_test.csv | csv | 384.4K | 0 | 3878 | 0 | 3878 | 0 | low |
| manifests_rebased/manifests_rebased/yellow_single_val.csv | csv | 206.0K | 0 | 2130 | 0 | 2130 | 0 | low |
| manifests_rebased/manifests_rebased/yellow_single_train_weighted.csv | csv | 9.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/yellow_single_train.csv | csv | 5.7M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/yellow_single_hard_val.csv | csv | 92.1K | 0 | 933 | 0 | 933 | 0 | low |
| manifests_rebased/manifests_rebased/yellow_real_val.csv | csv | 441.3K | 0 | 1420 | 0 | 1420 | 0 | low |
| manifests_rebased/manifests_rebased/yellow_real_test.csv | csv | 77.4K | 0 | 249 | 0 | 249 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_v4_real_only_test_ccpd_board.csv | csv | 54.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_v4_real_only_plain_plate_val_proxy.csv | csv | 2.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_pos0_enhanced_v1_train.csv | csv | 162.7M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_pos0_enhanced_v1_eval.csv | csv | 220.7M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_official_gray3_bluegreen_u1c_trainable_nospecial.csv | csv | 227.7M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_h34d_v3_three_tiers.csv | csv | 34.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e9c_exact_template_allprov_1800.csv | csv | 112.0M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e9b_exact_template_allprov_11160.csv | csv | 117.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e9a_exact_template_5prov_1800.csv | csv | 112.0M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e8c_brightness_replace_5prov.csv | csv | 111.0M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e7_boardnative_v2.csv | csv | 112.4M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e27_cluster3_hardtail_450.csv | csv | 113.6M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e26_cluster3_tail_boost_900.csv | csv | 113.3M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e25a_stageB_full_reintegrate.csv | csv | 13.4M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e20a_cluster2_beijing_prefix_contrast_1200.csv | csv | 113.2M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e19c_su_bf_low_tail_dense_240.csv | csv | 112.9M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e18b_su_bf_transition_dense_600.csv | csv | 112.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e17c_cluster3_tail_600.csv | csv | 112.7M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e17b_cluster3_transition_900.csv | csv | 112.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e17a_cluster2_suffixbank_900.csv | csv | 112.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e16b_nonanhui_ad_balance_12k_dump_v1.csv | csv | 118.9M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e16a_nonanhui_ad_balance_12k_std_v1.csv | csv | 118.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e15a_prewarp_slot_probe_300_v1_fullrun_1776329788.csv | csv | 112.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e15a_prewarp_slot_probe_300_v1.csv | csv | 112.0M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e14a_image_local_probe_probe180_fix4_1776321624.csv | csv | 112.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e14a_image_local_probe_probe123_fix2_1776320460.csv | csv | 112.0M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e14a_image_local_probe_300_v1_fullrun_1776322133.csv | csv | 112.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e13a_e9c_append_slotalign_aa0_5prov_300.csv | csv | 112.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e12_replace_pose_v3_append.csv | csv | 116.6M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e12_pose_replace_test.csv | csv | 56.6M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e12_e9c_append_boarddump_anticollapse_5prov_1200.csv | csv | 112.5M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e11_e9c_append_aa0heavy_5prov_600.csv | csv | 112.2M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e10b_boarddump_overflowfocus_5prov_1800_replace.csv | csv | 111.0M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_e10a_boarddump_exact_template_5prov_1800_replace.csv | csv | 111.0M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_m1_v4_a3000.csv | csv | 111.5M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_balance_mid_v1.csv | csv | 221.5M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_green_balance_lite_v1.csv | csv | 217.0M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_firstchar_patch_dataset_v1.csv | csv | 104.6M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/unified_manifest_e10_selfcheck_a50_replace.csv | csv | 111.0M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/subsets_e3_analysis_20260412/green8_test_synth_only.csv | csv | 450.8K | 0 | 1076 | 0 | 1076 | 0 | low |
| manifests_rebased/manifests_rebased/subsets_e3_analysis_20260412/green8_test_real_only.csv | csv | 1.9M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/subsets_e3_analysis_20260412/green8_test_all.csv | csv | 2.4M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/special_train.csv | csv | 2.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/special_test.csv | csv | 146.9K | 0 | 1379 | 0 | 1379 | 0 | low |
| manifests_rebased/manifests_rebased/province_stress_pose_val_v1/province_stress_pose_val_v1.csv | csv | 402.4K | 0 | 1240 | 0 | 1240 | 0 | low |
| manifests_rebased/manifests_rebased/province_degrade_train_v1/train_province_degrade_v1.csv | csv | 3.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/firstchar_tiny_gray_green8only_v1/train.csv | csv | 42.8M | 0 | 4997 | 0 | 4997 | 0 | low |
| manifests_rebased/manifests_rebased/firstchar_tiny_gray_alldata_v1/train.csv | csv | 62.5M | 0 | 4997 | 0 | 4997 | 0 | low |
| manifests_rebased/manifests_rebased/firstchar_batch1/D3_firstchar_manifest_green8_normal7_selectedspecial_v1_train.csv | csv | 79.6M | 0 | 4997 | 0 | 4997 | 0 | low |
| manifests_rebased/manifests_rebased/firstchar_batch1/D2_firstchar_manifest_green8_normal7_v1_train.csv | csv | 62.5M | 0 | 4997 | 0 | 4997 | 0 | low |
| manifests_rebased/manifests_rebased/firstchar_batch1/D1_firstchar_manifest_green8_only_v1_train.csv | csv | 42.8M | 0 | 4997 | 0 | 4997 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/val_B1A_original_eval.csv | csv | 2.4M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/train_B1A_train_v4e3_ccpdboard_eval_original.csv | csv | 16.0M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_support_cblprd.csv | csv | 193.1K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 299.0K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_green_edgefit_hard.csv | csv | 87.3K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv | csv | 36.6K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_green_ccpd2020_real.csv | csv | 274.7K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_green_bridge_exactquad.csv | csv | 203.3K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_blue_crpd_real.csv | csv | 463.0K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_blue_ccpd2019_real.csv | csv | 531.1K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/val_B1A_extreme_ccpdboard_v4e3.csv | csv | 2.5M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/train_B1A_extreme_ccpdboard_v4e3.csv | csv | 16.0M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_support_cblprd.csv | csv | 184.8K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_green_nonanhui_template_synth.csv | csv | 288.6K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_green_edgefit_hard.csv | csv | 85.1K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_green_edgefit_extreme.csv | csv | 77.7K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_green_ccpd2020_real.csv | csv | 267.8K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_green_bridge_exactquad.csv | csv | 197.7K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_blue_crpd_real.csv | csv | 449.2K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_blue_ccpd2019_real.csv | csv | 517.3K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty/val_B1A.csv | csv | 2.4M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty/train_B1A.csv | csv | 15.5M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty/proxy_support_cblprd.csv | csv | 184.8K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty/proxy_green_nonanhui_template_synth.csv | csv | 288.6K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty/proxy_green_edgefit_hard.csv | csv | 85.1K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty/proxy_green_edgefit_extreme.csv | csv | 35.7K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty/proxy_green_ccpd2020_real.csv | csv | 267.8K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty/proxy_green_bridge_exactquad.csv | csv | 197.7K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty/proxy_blue_crpd_real.csv | csv | 449.2K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_difficulty/proxy_blue_ccpd2019_real.csv | csv | 517.3K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2D_pose_quad/val_replace_extreme.csv | csv | 90.1K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2D_pose_quad/val_pose_quad.csv | csv | 2.5M | 0 | 3297 | 0 | 3297 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2D_pose_quad/train_replace_extreme_as_test.csv | csv | 814.5K | 0 | 2790 | 0 | 2790 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2D_pose_quad/train_replace_extreme.csv | csv | 817.2K | 0 | 2790 | 0 | 2790 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2D_pose_quad/train_pose_quad.csv | csv | 16.4M | 0 | 3325 | 0 | 3325 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2D_paradigm3_progress/val_B2D_paradigm3_progress.csv | csv | 2.4M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2D_paradigm3_progress/train_B2D_paradigm3_progress.csv | csv | 16.8M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2C_paradigm3_softfreeze/val_B2C_paradigm3_softfreeze.csv | csv | 2.4M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2C_paradigm3_softfreeze/train_B2C_paradigm3_softfreeze.csv | csv | 16.1M | 0 | 3325 | 0 | 3325 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2C_paradigm3_obbquad/val_B2C_paradigm3_obbquad.csv | csv | 2.4M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B2C_paradigm3_obbquad/train_B2C_paradigm3_obbquad.csv | csv | 16.1M | 0 | 3325 | 0 | 3325 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_edgefit_extreme.csv | csv | 75.7K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/val_B1B_E6AB_combined.csv | csv | 2.5M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/train_B1B_E6AB_preblur_v3_combined.csv | csv | 16.4M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_green_edgefit_extreme.csv | csv | 37.6K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_green_edgefit_extreme.csv | csv | 80.8K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/val_B1A_E6_original_eval.csv | csv | 2.5M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/train_B1A_E6_both_tilt_perspective_eval_original.csv | csv | 16.4M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_green_edgefit_extreme.csv | csv | 37.6K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_green_edgefit_extreme.csv | csv | 81.9K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/val_B1A_E6_original_eval.csv | csv | 2.5M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/train_B1A_E6_axis_dominant_perspective_eval_original.csv | csv | 16.4M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_green_edgefit_extreme.csv | csv | 37.6K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_green_edgefit_extreme.csv | csv | 79.4K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/val_B1A_E6B_original_eval.csv | csv | 2.5M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/train_B1A_E6B_compound_visible_eval_original.csv | csv | 16.4M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_green_edgefit_extreme.csv | csv | 37.6K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_green_edgefit_extreme.csv | csv | 79.4K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/val_B1A_E6A_original_eval.csv | csv | 2.5M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/train_B1A_E6A_single_axis_visible_eval_original.csv | csv | 16.4M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_edgefit_extreme.csv | csv | 37.6K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_edgefit_extreme.csv | csv | 76.0K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/val_B1A_E1_original_eval.csv | csv | 2.5M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv | csv | 16.4M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_support_cblprd.csv | csv | 201.4K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 309.3K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_edgefit_hard.csv | csv | 89.5K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv | csv | 37.6K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_ccpd2020_real.csv | csv | 281.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_bridge_exactquad.csv | csv | 208.9K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_blue_crpd_real.csv | csv | 476.7K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_blue_ccpd2019_real.csv | csv | 544.8K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_support_cblprd.csv | csv | 199.0K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_green_nonanhui_template_synth.csv | csv | 306.4K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_green_edgefit_hard.csv | csv | 88.8K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_green_edgefit_extreme.csv | csv | 80.2K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_green_ccpd2020_real.csv | csv | 279.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_green_bridge_exactquad.csv | csv | 207.3K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_blue_crpd_real.csv | csv | 472.8K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_blue_ccpd2019_real.csv | csv | 540.9K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/val_B1A_D_original_eval.csv | csv | 2.5M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/train_B1A_D_extreme900_v4e3_ccpdboard_eval_original.csv | csv | 16.7M | 0 | 3323 | 0 | 3323 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_support_cblprd.csv | csv | 199.0K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 306.4K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_green_edgefit_hard.csv | csv | 88.8K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv | csv | 37.3K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_green_ccpd2020_real.csv | csv | 279.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_green_bridge_exactquad.csv | csv | 207.3K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_blue_crpd_real.csv | csv | 472.8K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_blue_ccpd2019_real.csv | csv | 540.9K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/val_B1A_C_original_eval.csv | csv | 2.5M | 0 | 3298 | 0 | 3298 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/train_B1A_C_train_v4e3_ccpdboard_eval_original.csv | csv | 16.3M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_support_cblprd.csv | csv | 199.0K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 306.4K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_green_edgefit_hard.csv | csv | 88.8K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv | csv | 37.3K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_green_ccpd2020_real.csv | csv | 279.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_green_bridge_exactquad.csv | csv | 207.3K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_blue_crpd_real.csv | csv | 472.8K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_blue_ccpd2019_real.csv | csv | 540.9K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_support_cblprd.csv | csv | 199.0K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_green_nonanhui_template_synth.csv | csv | 306.4K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_green_edgefit_hard.csv | csv | 88.8K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_green_edgefit_extreme.csv | csv | 79.5K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_green_ccpd2020_real.csv | csv | 279.6K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_green_bridge_exactquad.csv | csv | 207.3K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_blue_crpd_real.csv | csv | 472.8K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_blue_ccpd2019_real.csv | csv | 540.9K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/val_A1B.csv | csv | 2.3M | 0 | 3303 | 0 | 3303 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/val_A1.csv | csv | 2.3M | 0 | 3303 | 0 | 3303 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/val_A0.csv | csv | 2.3M | 0 | 3303 | 0 | 3303 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/train_A1B.csv | csv | 15.1M | 0 | 3319 | 0 | 3319 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/train_A1.csv | csv | 14.0M | 0 | 3320 | 0 | 3320 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/train_A0.csv | csv | 14.7M | 0 | 3317 | 0 | 3317 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/proxy_support_cblprd.csv | csv | 184.8K | 0 | 1200 | 0 | 1200 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/proxy_green_nonanhui_template_synth.csv | csv | 288.6K | 0 | 1500 | 0 | 1500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/proxy_green_ccpd2020_real.csv | csv | 267.8K | 0 | 1001 | 0 | 1001 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/proxy_green_bridge_exactquad.csv | csv | 197.7K | 0 | 800 | 0 | 800 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/proxy_blue_crpd_real.csv | csv | 449.2K | 0 | 1882 | 0 | 1882 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v3_realprimary/proxy_blue_ccpd2019_real.csv | csv | 517.3K | 0 | 2000 | 0 | 2000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v2_foundation/val_stageA_v2.csv | csv | 2.1M | 0 | 3312 | 0 | 3312 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v2_foundation/train_stageA_v2.csv | csv | 10.7M | 0 | 3318 | 0 | 3318 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v2_foundation/proxy_support.csv | csv | 200.8K | 0 | 1300 | 0 | 1300 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v2_foundation/proxy_green_real_foundation.csv | csv | 28.3K | 0 | 105 | 0 | 105 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v2_foundation/proxy_green_bridge.csv | csv | 125.7K | 0 | 500 | 0 | 500 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_v2_foundation/proxy_blue_real_foundation.csv | csv | 335.1K | 0 | 1363 | 0 | 1363 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_redesign/val.csv | csv | 1.7M | 0 | 3308 | 0 | 3308 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_redesign/train_stageA.csv | csv | 12.5M | 0 | 3327 | 0 | 3327 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_redesign/proxy_stageA_mixed_foundation.csv | csv | 481.3K | 0 | 2965 | 0 | 2965 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_redesign/proxy_stageA_green_simple.csv | csv | 148.8K | 0 | 1000 | 0 | 1000 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stagea_redesign/proxy_stageA_blue_simple.csv | csv | 332.7K | 0 | 1965 | 0 | 1965 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageE_v2_balanced/val_E2.csv | csv | 2.5M | 0 | 3297 | 0 | 3297 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageE_v2_balanced/train_E2.csv | csv | 18.4M | 0 | 3318 | 0 | 3318 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageE_v1_extreme/val_E1.csv | csv | 2.5M | 0 | 3297 | 0 | 3297 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageE_v1_extreme/train_E1.csv | csv | 24.0M | 0 | 3323 | 0 | 3323 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageE_e3_main/val_e3_main.csv | csv | 2.5M | 0 | 3297 | 0 | 3297 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageE_e3_main/train_e3_main.csv | csv | 18.4M | 0 | 3318 | 0 | 3318 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageE_e3_control/val_e3_control.csv | csv | 2.5M | 0 | 3297 | 0 | 3297 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3_stageE_e3_control/train_e3_control.csv | csv | 18.4M | 0 | 3318 | 0 | 3318 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/val_nonccpd_green.csv | csv | 783.3K | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/val_crpd_blue.csv | csv | 534.6K | 0 | 2884 | 0 | 2884 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv | csv | 183.0K | 0 | 833 | 0 | 833 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/val_ccpd2019_blue.csv | csv | 241.3K | 0 | 1144 | 0 | 1144 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/val_cblprd_blue.csv | csv | 189.0K | 0 | 1598 | 0 | 1598 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/val.csv | csv | 1.9M | 0 | 3309 | 0 | 3309 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/train_stageB.csv | csv | 15.1M | 0 | 3328 | 0 | 3328 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/train_stageA.csv | csv | 24.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/test_green_simple.csv | csv | 459.5K | 0 | 2130 | 0 | 2130 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/test_green_real.csv | csv | 1.6M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/test_green_hard.csv | csv | 70.2K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/test_green_extreme.csv | csv | 29.4K | 0 | 124 | 0 | 124 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/test_blue_simple.csv | csv | 418.9K | 0 | 1997 | 0 | 1997 | 0 | low |
| manifests_rebased/manifests_rebased/curriculum_gray3/test_blue_hard.csv | csv | 27.9M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/crpd_yellow_train_only.csv | csv | 1.8M | 0 | 4501 | 0 | 4501 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_v1_obbquad/val_v1_obbquad.csv | csv | 93.7K | 0 | 300 | 0 | 300 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_v1_obbquad/train_v1_obbquad.csv | csv | 851.6K | 0 | 2700 | 0 | 2700 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_pose_v3/val_ccpd2020_replace_pose_v3.csv | csv | 100.1K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_pose_v3/train_ccpd2020_replace_pose_v3.csv | csv | 909.9K | 0 | 2790 | 0 | 2790 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_extreme_v4/val_extreme_v3.csv | csv | 94.7K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_extreme_v4/train_extreme_v3.csv | csv | 285.4K | 0 | 930 | 0 | 930 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_extreme_v3/val_extreme_v3.csv | csv | 94.4K | 0 | 310 | 0 | 310 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_extreme_v3/train_extreme_v3.csv | csv | 285.8K | 0 | 930 | 0 | 930 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_extreme_v2_additional/train_B2D_additional.csv | csv | 783.8K | 0 | 2700 | 0 | 2700 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_extreme_v1/val_B2C_ccpd2020_replace_extreme.csv | csv | 82.0K | 0 | 300 | 0 | 300 | 0 | low |
| manifests_rebased/manifests_rebased/ccpd2020_replace_extreme_v1/train_B2C_ccpd2020_replace_extreme.csv | csv | 747.0K | 0 | 2700 | 0 | 2700 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_v4_board_aligned_real_only_crpd_raw.csv | csv | 248.9M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_v4_board_aligned_real_only.csv | csv | 250.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_specialist_official_v2_balanced_stable.csv | csv | 18.7M | 0 | 5611 | 0 | 5611 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_specialist_official_v1_balanced_existing.csv | csv | 10.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_h30a_targeted_tail.csv | csv | 106.1M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv | csv | 105.6M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e7_boardnative_provbal.csv | csv | 111.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e6c_boardnative_append_v2_a800_20260413.csv | csv | 111.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e6c_boardnative_append_v1_20260413.csv | csv | 111.6M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e5_dumplike_boarddump_bright_v1_a3100_20260412.csv | csv | 112.5M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e4_extreme_append10_20260412.csv | csv | 111.7M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e3_equalprov_b_20260412.csv | csv | 111.5M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e3_equalprov_a_20260412.csv | csv | 114.7M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_plus_dumplike_boarddump_bright_v1_20260412.csv | csv | 111.7M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv | csv | 111.5M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed.csv | csv | 111.9M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative.csv | csv | 111.7M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v2.csv | csv | 110.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v4_realistic_b.csv | csv | 111.2M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v4_realistic.csv | csv | 110.9M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v3_zhe_guard_yuehu_restore.csv | csv | 111.9M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v2_zhe_guard.csv | csv | 111.0M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v1.csv | csv | 110.4M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak.csv | csv | 104.4M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40_exact.csv | csv | 105.5M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b_anhui40.csv | csv | 105.3M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_greenonly_b.csv | csv | 104.4M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_round2_conservative_a.csv | csv | 195.8M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_baseline_v1.csv | csv | 216.9M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_aggr_v1_existing_paths.csv | csv | 204.5M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_aggr_v1_existing_only.csv | csv | 218.4M | 0 | 3331 | 0 | 3331 | 0 | low |
| manifests_rebased/manifests_rebased/archived/unified_manifest_green_balance_aggr_v1.csv | csv | 220.0M | 0 | 3331 | 0 | 3331 | 0 | low |

## 路径类型分布汇总

| 类型 | 合计 |
|------|------|
| 绝对路径 | 0 |
| 相对路径 | 711269 |
| 裸文件名 | 0 |
| URL | 0 |
| 空值 | 0 |
| 有效 | 707458 |
| 无效 | 3811 |

## 风险与建议

### ❌ 无效路径风险 (发现 3811 个无效路径)

部分 manifest 中的路径在当前文件系统下不存在。可能原因：
- 数据集路径已变更
- 数据被移动或删除
- 软链接断链

**建议**: 首先检查软链接是否正常，然后验证数据完整性。

### 推荐迁移方案

1. **不改动原始 manifest** — 所有修改通过生成新 manifest 实现
2. **使用软链接兼容** — 对旧绝对路径通过 `ln -s` 兼容
3. **dataset_root 配置化** — 训练脚本读取 config 中的 `dataset_root` 字段
4. **新 manifest 统一相对路径** — 所有路径基于 `dataset_root`

---

*报告由 audit_manifest_paths.py 自动生成，336 份 manifest 已扫描*