# Manifest 路径依赖审计报告

生成时间: 2026-05-07 12:33:46

扫描路径: /home/wzzz/LPRNet

---

## 总览

| 指标 | 值 |
|------|-----|
| 扫描 manifest 数 | 440 |
| 高风险 | 376 |
| 中风险 | 1 |
| 低风险 | 63 |
| 绝对路径总数 | 754479 |
| 无效路径总数 | 53585 |
| 采样样本总行数 | 1128482 |

## 每份 manifest 详情

| 文件 | 类型 | 大小 | 绝对路径 | 相对路径 | 裸文件名 | 有效 | 无效 | 风险 |
|------|------|------|---------|---------|---------|------|------|------|
| config/requirements-train.txt | text | 40B | 0 | 0 | 5 | 0 | 5 | medium |
| manifests/unified_manifest_v4_round1_green_conservative.summary.json | json | 1.1K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_v4_real_only_test_plain_plate.csv | csv | 312B | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_v4_geom_audited.summary.json | json | 1.0K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_v4_board_aligned_real_only.summary.json | json | 1.3K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_v3_smoketest.summary.json | json | 2.7K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_v3_round1_green_conservative.summary.json | json | 1.8K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_v3.summary.json | json | 2.7K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_v2_with_pseudo_geom.summary.json | json | 1010B | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_v1.summary.json | json | 453B | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_pos0_enhanced_v1_summary.json | json | 1.8K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_official_gray3_bluegreen_u1c_trainable_nospecial.summary.json | json | 1.3K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_official_gray3_bluegreen_u1c.summary.json | json | 28.5K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_official_gray3_bluegreen_u1b.summary.json | json | 27.7K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_official_gray3_bluegreen_u1.summary.json | json | 10.8K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_specialist_official_v2_balanced_stable.summary.json | json | 10.7K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_specialist_official_v1_balanced_train.summary.json | json | 20.5K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_specialist_official_v1_balanced_existing.summary.json | json | 578B | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_e9c_exact_template_allprov_1800.report.json | json | 1.2K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_e9b_exact_template_allprov_11160.report.json | json | 1.2K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_e9a_exact_template_5prov_1800.report.json | json | 839B | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_e13a_e9c_append_slotalign_aa0_5prov_300.report.json | json | 3.7K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_e12_replace_pose_v3_append.report.json | json | 891B | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_e12_e9c_append_boarddump_anticollapse_5prov_1200.report.json | json | 3.8K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_e11_e9c_append_aa0heavy_5prov_600.report.json | json | 1.6K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_e10b_boarddump_overflowfocus_5prov_1800_replace.summary.json | json | 10.3K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_e10a_boarddump_exact_template_5prov_1800_replace.summary.json | json | 10.4K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_balance_round2_summary.json | json | 655B | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_green_balance_aggr_v1_summary.json | json | 1.6K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/unified_manifest_e10_selfcheck_a50_replace.summary.json | json | 8.4K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/firstchar_tiny_gray_green8only_v1/test.csv | csv | 2.0M | 0 | 6662 | 0 | 6662 | 0 | low |
| manifests/firstchar_tiny_gray_green8only_v1/summary.json | json | 238B | 0 | 0 | 0 | 0 | 0 | low |
| manifests/firstchar_tiny_gray_alldata_v1_smoke512/test.csv | csv | 152.5K | 0 | 1024 | 0 | 1024 | 0 | low |
| manifests/firstchar_tiny_gray_alldata_v1/test.csv | csv | 3.0M | 0 | 6662 | 0 | 6662 | 0 | low |
| manifests/firstchar_tiny_gray_alldata_v1/summary.json | json | 2.0K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/firstchar_tiny_gray3_fullcrop_bal31_v1/summary.json | json | 1.3K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/firstchar_batch1/firstchar_batch1_manifest_summary.json | json | 3.9K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/summary.json | json | 1.2K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/summary.json | json | 1.6K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_difficulty/summary_B1A.json | json | 2.2K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/summary_new_proxy.json | json | 3.6K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/summary.json | json | 3.6K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/summary_E6_new_proxy.json | json | 3.2K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/summary_E6.json | json | 3.2K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/summary_E6_new_proxy.json | json | 3.4K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/summary_E6.json | json | 3.4K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/summary_E6B_new_proxy.json | json | 2.8K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/summary_E6B.json | json | 2.8K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/summary_E6A_new_proxy.json | json | 2.8K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/summary_E6A.json | json | 2.8K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/summary_E1_new_proxy.json | json | 3.4K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/summary_E1.json | json | 3.4K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/summary_D_new_proxy.json | json | 3.8K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/summary_D.json | json | 3.8K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/summary_C.json | json | 3.4K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/summary_C_new_proxy.json | json | 3.4K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stagea_v3_realprimary/summary_A1B.json | json | 3.6K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stagea_v3_realprimary/summary_A1.json | json | 3.6K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stagea_v3_realprimary/summary_A0.json | json | 3.6K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/curriculum_gray3_stagea_v2_foundation/summary.json | json | 3.6K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/cluster_special_validation_v1/summary.json | json | 766B | 0 | 0 | 0 | 0 | 0 | low |
| manifests/ccpd2020_replace_extreme_v2_additional/val_B2D_additional.csv | csv | 194B | 0 | 0 | 0 | 0 | 0 | low |
| manifests/cblprd_cv_geom_manifest.csv | csv | 89.9M | 0 | 6662 | 0 | 6662 | 0 | low |
| manifests/Archive/manifest_archive_index.json | json | 10.4K | 0 | 0 | 0 | 0 | 0 | low |
| manifests/yellow_train_weighted.csv | csv | 20.4M | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/yellow_train.csv | csv | 7.8M | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/yellow_test.csv | csv | 452.6K | 3878 | 0 | 0 | 3878 | 0 | high |
| manifests/yellow_single_val.csv | csv | 243.4K | 2130 | 0 | 0 | 2130 | 0 | high |
| manifests/yellow_single_train_weighted.csv | csv | 10.8M | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/yellow_single_train.csv | csv | 6.7M | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/yellow_single_hard_val.csv | csv | 108.5K | 933 | 0 | 0 | 933 | 0 | high |
| manifests/yellow_real_val.csv | csv | 466.3K | 1420 | 1420 | 0 | 2840 | 0 | high |
| manifests/yellow_real_test.csv | csv | 81.8K | 249 | 249 | 0 | 498 | 0 | high |
| manifests/unified_manifest_v4_round1_green_conservative.csv | csv | 207.2M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/unified_manifest_v4_real_only_test_ccpd_board.csv | csv | 56.9M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_v4_real_only_plain_plate_val_proxy.csv | csv | 2.3M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_v4_geom_audited.csv | csv | 267.7M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/unified_manifest_v3_smoketest.csv | csv | 251.0M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/unified_manifest_v3_round1_green_conservative.csv | csv | 207.0M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/unified_manifest_pos0_enhanced_v1_train.csv | csv | 168.0M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_pos0_enhanced_v1_eval.csv | csv | 232.1M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_official_gray3_bluegreen_u1c_trainable_nospecial.csv | csv | 237.7M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_official_gray3_bluegreen_u1c.csv | csv | 275.1M | 3331 | 3331 | 0 | 4996 | 1666 | high |
| manifests/unified_manifest_official_gray3_bluegreen_u1b.csv | csv | 269.9M | 3331 | 3331 | 0 | 4996 | 1666 | high |
| manifests/unified_manifest_official_gray3_bluegreen_u1.csv | csv | 246.9M | 3331 | 3331 | 0 | 4996 | 1666 | high |
| manifests/unified_manifest_green_h34d_v3_three_tiers.csv | csv | 38.0M | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/unified_manifest_green_edgefit_v4_e4_extreme_append10_20260412_train.csv | csv | 181.7K | 310 | 310 | 0 | 310 | 310 | high |
| manifests/unified_manifest_green_edgefit_v4_e3_equalprov_b_20260412_train.csv | csv | 2.3M | 4340 | 4340 | 0 | 4340 | 4340 | high |
| manifests/unified_manifest_green_edgefit_v4_e3_equalprov_a_20260412_train.csv | csv | 7.6M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/unified_manifest_green_edgefit_v4_e2_20260411_train.csv | csv | 1.5M | 3000 | 3000 | 0 | 3000 | 3000 | high |
| manifests/unified_manifest_green_edgefit_v3_allprov.csv | csv | 3.7M | 3331 | 0 | 0 | 0 | 3331 | high |
| manifests/unified_manifest_green_e9c_exact_template_allprov_1800.csv | csv | 117.5M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e9b_exact_template_allprov_11160.csv | csv | 122.8M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e9a_exact_template_5prov_1800.csv | csv | 117.5M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e8c_brightness_replace_5prov.csv | csv | 116.5M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e7_boardnative_v2.csv | csv | 117.9M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e27_cluster3_hardtail_450.csv | csv | 119.1M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e26_cluster3_tail_boost_900.csv | csv | 118.8M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e25b_stageB_lite_reintegrate.csv | csv | 6.9M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/unified_manifest_green_e25a_stageB_full_reintegrate.csv | csv | 13.9M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e25a_stageA_targeted_repr_rebuild.csv | csv | 3.8M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/unified_manifest_green_e20a_cluster2_beijing_prefix_contrast_1200.csv | csv | 118.7M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e19c_su_bf_low_tail_dense_240.csv | csv | 118.5M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e18b_su_bf_transition_dense_600.csv | csv | 118.3M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e17c_cluster3_tail_600.csv | csv | 118.2M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e17b_cluster3_transition_900.csv | csv | 118.3M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e17a_cluster2_suffixbank_900.csv | csv | 118.3M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e16b_nonanhui_ad_balance_12k_dump_v1.csv | csv | 124.6M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e16a_nonanhui_ad_balance_12k_std_v1.csv | csv | 124.5M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e15a_prewarp_slot_probe_300_v1_fullrun_1776329788.csv | csv | 117.6M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e15a_prewarp_slot_probe_300_v1.csv | csv | 117.5M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e14a_image_local_probe_probe180_fix4_1776321624.csv | csv | 117.6M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e14a_image_local_probe_probe123_fix2_1776320460.csv | csv | 117.5M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e14a_image_local_probe_300_v1_fullrun_1776322133.csv | csv | 117.6M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e13a_e9c_append_slotalign_aa0_5prov_300.csv | csv | 117.6M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e12_replace_pose_v3_append.csv | csv | 122.2M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e12_pose_replace_test.csv | csv | 59.4M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e12_e9c_append_boarddump_anticollapse_5prov_1200.csv | csv | 118.0M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e11_e9c_append_aa0heavy_5prov_600.csv | csv | 117.7M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e10b_boarddump_overflowfocus_5prov_1800_replace.csv | csv | 116.4M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_e10a_boarddump_exact_template_5prov_1800_replace.csv | csv | 116.4M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_m1_v4_a3000.csv | csv | 117.0M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_balance_mid_v1.csv | csv | 233.0M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_green_balance_lite_v1.csv | csv | 228.3M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_firstchar_patch_dataset_v1.csv | csv | 109.9M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/unified_manifest_e10_selfcheck_a50_replace.csv | csv | 116.5M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/subsets_e3_analysis_20260412/green8_test_synth_only.csv | csv | 469.7K | 1076 | 1076 | 0 | 2152 | 0 | high |
| manifests/subsets_e3_analysis_20260412/green8_test_real_only.csv | csv | 2.0M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/subsets_e3_analysis_20260412/green8_test_all.csv | csv | 2.5M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/special_train.csv | csv | 3.3M | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/special_test.csv | csv | 171.1K | 1379 | 42 | 0 | 1421 | 0 | high |
| manifests/province_stress_pose_val_v1/province_stress_pose_val_v1.csv | csv | 424.2K | 1240 | 0 | 0 | 1240 | 0 | high |
| manifests/province_degrade_train_v1/train_province_degrade_v1.csv | csv | 3.3M | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/normal7_test_only_v1.csv | csv | 53.2M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/firstchar_tiny_gray_green8only_v1/train.csv | csv | 43.0M | 1665 | 4997 | 0 | 6662 | 0 | high |
| manifests/firstchar_tiny_gray_alldata_v1/train.csv | csv | 62.7M | 1665 | 4997 | 0 | 6662 | 0 | high |
| manifests/firstchar_tiny_gray3_fullcrop_bal31_v1_smoke2k/train.csv | csv | 614.8K | 120 | 3880 | 0 | 4000 | 0 | high |
| manifests/firstchar_tiny_gray3_fullcrop_bal31_v1_smoke10k/train.csv | csv | 3.0M | 201 | 6461 | 0 | 6662 | 0 | high |
| manifests/firstchar_tiny_gray3_fullcrop_bal31_v1/train.csv | csv | 59.1M | 190 | 6472 | 0 | 6662 | 0 | high |
| manifests/firstchar_batch1/D3_firstchar_manifest_green8_normal7_selectedspecial_v1_train.csv | csv | 79.9M | 1665 | 4997 | 0 | 6662 | 0 | high |
| manifests/firstchar_batch1/D2_firstchar_manifest_green8_normal7_v1_train.csv | csv | 62.7M | 1665 | 4997 | 0 | 6662 | 0 | high |
| manifests/firstchar_batch1/D1_firstchar_manifest_green8_only_v1_train.csv | csv | 43.0M | 1665 | 4997 | 0 | 6662 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/val_B1A_original_eval.csv | csv | 2.6M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/train_B1A_train_v4e3_ccpdboard_eval_original.csv | csv | 17.1M | 3320 | 17 | 0 | 3337 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_support_cblprd.csv | csv | 214.2K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 325.4K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_green_edgefit_hard.csv | csv | 92.7K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv | csv | 38.8K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_green_ccpd2020_real.csv | csv | 292.3K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_green_bridge_exactquad.csv | csv | 217.4K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_blue_crpd_real.csv | csv | 498.1K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/proxy_blue_ccpd2019_real.csv | csv | 566.2K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/extreme_train_swap_mapping.csv | csv | 123.2K | 600 | 0 | 0 | 600 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/val_B1A_extreme_ccpdboard_v4e3.csv | csv | 2.7M | 3298 | 37 | 0 | 3335 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/train_B1A_extreme_ccpdboard_v4e3.csv | csv | 17.1M | 3320 | 17 | 0 | 3337 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_support_cblprd.csv | csv | 205.9K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_green_nonanhui_template_synth.csv | csv | 315.0K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_green_edgefit_hard.csv | csv | 90.5K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_green_edgefit_extreme.csv | csv | 79.8K | 124 | 124 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_green_ccpd2020_real.csv | csv | 285.4K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_green_bridge_exactquad.csv | csv | 211.8K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_blue_crpd_real.csv | csv | 484.4K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/proxy_blue_ccpd2019_real.csv | csv | 552.4K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/extreme_swap_mapping.csv | csv | 177.3K | 848 | 0 | 0 | 848 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty/val_B1A.csv | csv | 2.5M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty/train_B1A.csv | csv | 16.6M | 3320 | 0 | 0 | 3320 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty/proxy_support_cblprd.csv | csv | 205.9K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty/proxy_green_nonanhui_template_synth.csv | csv | 315.0K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty/proxy_green_edgefit_hard.csv | csv | 90.5K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty/proxy_green_edgefit_extreme.csv | csv | 37.9K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty/proxy_green_ccpd2020_real.csv | csv | 285.4K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty/proxy_green_bridge_exactquad.csv | csv | 211.8K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty/proxy_blue_crpd_real.csv | csv | 484.4K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_difficulty/proxy_blue_ccpd2019_real.csv | csv | 552.4K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/val_replace_extreme.csv | csv | 95.5K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/val_pose_quad.csv | csv | 2.6M | 3297 | 0 | 0 | 3297 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/train_replace_extreme_as_test.csv | csv | 863.6K | 2790 | 0 | 0 | 2790 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/train_replace_extreme.csv | csv | 866.3K | 2790 | 0 | 0 | 2790 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/train_pose_quad.csv | csv | 17.5M | 3325 | 0 | 0 | 3325 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2D_paradigm3_progress/val_B2D_paradigm3_progress.csv | csv | 2.5M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2D_paradigm3_progress/train_B2D_paradigm3_progress.csv | csv | 18.0M | 3320 | 0 | 0 | 3320 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_softfreeze/val_B2C_paradigm3_softfreeze.csv | csv | 2.5M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_softfreeze/train_B2C_paradigm3_softfreeze.csv | csv | 17.2M | 3325 | 0 | 0 | 3325 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_obbquad/val_B2C_paradigm3_obbquad.csv | csv | 2.5M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_obbquad/train_B2C_paradigm3_obbquad.csv | csv | 17.3M | 3325 | 0 | 0 | 3325 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_edgefit_extreme.csv | csv | 77.9K | 124 | 124 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/extreme_proxy_swap_mapping.csv | csv | 49.0K | 248 | 0 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/val_B1B_E6AB_combined.csv | csv | 2.7M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/train_B1B_E6AB_preblur_v3_combined.csv | csv | 17.6M | 3320 | 17 | 0 | 3337 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_green_edgefit_extreme.csv | csv | 39.8K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/extreme_train_swap_mapping.csv | csv | 119.1K | 600 | 0 | 0 | 600 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_green_edgefit_extreme.csv | csv | 83.0K | 124 | 124 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/extreme_proxy_swap_mapping.csv | csv | 50.1K | 248 | 0 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/val_B1A_E6_original_eval.csv | csv | 2.7M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/train_B1A_E6_both_tilt_perspective_eval_original.csv | csv | 17.6M | 3320 | 17 | 0 | 3337 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_green_edgefit_extreme.csv | csv | 39.8K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/extreme_train_swap_mapping.csv | csv | 121.6K | 600 | 0 | 0 | 600 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_green_edgefit_extreme.csv | csv | 84.0K | 124 | 124 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/extreme_proxy_swap_mapping.csv | csv | 50.0K | 248 | 0 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/val_B1A_E6_original_eval.csv | csv | 2.7M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/train_B1A_E6_axis_dominant_perspective_eval_original.csv | csv | 17.6M | 3320 | 17 | 0 | 3337 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_green_edgefit_extreme.csv | csv | 39.8K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/extreme_train_swap_mapping.csv | csv | 121.4K | 600 | 0 | 0 | 600 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_green_edgefit_extreme.csv | csv | 81.6K | 124 | 124 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/extreme_proxy_swap_mapping.csv | csv | 49.8K | 248 | 0 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/val_B1A_E6B_original_eval.csv | csv | 2.7M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/train_B1A_E6B_compound_visible_eval_original.csv | csv | 17.6M | 3320 | 17 | 0 | 3337 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_green_edgefit_extreme.csv | csv | 39.8K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/extreme_train_swap_mapping.csv | csv | 121.0K | 600 | 0 | 0 | 600 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_green_edgefit_extreme.csv | csv | 81.5K | 124 | 124 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/extreme_proxy_swap_mapping.csv | csv | 49.1K | 248 | 0 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/val_B1A_E6A_original_eval.csv | csv | 2.7M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/train_B1A_E6A_single_axis_visible_eval_original.csv | csv | 17.6M | 3320 | 17 | 0 | 3337 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_edgefit_extreme.csv | csv | 39.8K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/extreme_train_swap_mapping.csv | csv | 119.3K | 600 | 0 | 0 | 600 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_edgefit_extreme.csv | csv | 78.1K | 124 | 124 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/extreme_proxy_swap_mapping.csv | csv | 48.7K | 248 | 0 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/val_B1A_E1_original_eval.csv | csv | 2.7M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv | csv | 17.6M | 3320 | 17 | 0 | 3337 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_support_cblprd.csv | csv | 222.5K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 335.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_edgefit_hard.csv | csv | 94.9K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv | csv | 39.8K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_ccpd2020_real.csv | csv | 299.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_bridge_exactquad.csv | csv | 222.9K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_blue_crpd_real.csv | csv | 511.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_blue_ccpd2019_real.csv | csv | 580.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/extreme_train_swap_mapping.csv | csv | 118.3K | 600 | 0 | 0 | 600 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_support_cblprd.csv | csv | 220.1K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_green_nonanhui_template_synth.csv | csv | 332.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_green_edgefit_hard.csv | csv | 94.3K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_green_edgefit_extreme.csv | csv | 82.4K | 124 | 124 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_green_ccpd2020_real.csv | csv | 297.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_green_bridge_exactquad.csv | csv | 221.3K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_blue_crpd_real.csv | csv | 507.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/proxy_blue_ccpd2019_real.csv | csv | 576.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/extreme_proxy_swap_mapping.csv | csv | 51.4K | 248 | 0 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/val_B1A_D_original_eval.csv | csv | 2.7M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/train_B1A_D_extreme900_v4e3_ccpdboard_eval_original.csv | csv | 17.8M | 3323 | 0 | 0 | 3323 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_support_cblprd.csv | csv | 220.1K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 332.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_green_edgefit_hard.csv | csv | 94.3K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv | csv | 39.5K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_green_ccpd2020_real.csv | csv | 297.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_green_bridge_exactquad.csv | csv | 221.3K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_blue_crpd_real.csv | csv | 507.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/proxy_blue_ccpd2019_real.csv | csv | 576.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/extreme_train900_swap_mapping.csv | csv | 377.0K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/val_B1A_C_original_eval.csv | csv | 2.7M | 3298 | 0 | 0 | 3298 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/train_B1A_C_train_v4e3_ccpdboard_eval_original.csv | csv | 17.4M | 3320 | 17 | 0 | 3337 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_support_cblprd.csv | csv | 220.1K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_green_nonanhui_template_synth.csv | csv | 332.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_green_edgefit_hard.csv | csv | 94.3K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv | csv | 39.5K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_green_ccpd2020_real.csv | csv | 297.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_green_bridge_exactquad.csv | csv | 221.3K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_blue_crpd_real.csv | csv | 507.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/proxy_blue_ccpd2019_real.csv | csv | 576.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/extreme_train_swap_mapping.csv | csv | 124.9K | 600 | 0 | 0 | 600 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_support_cblprd.csv | csv | 220.1K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_green_nonanhui_template_synth.csv | csv | 332.7K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_green_edgefit_hard.csv | csv | 94.3K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_green_edgefit_extreme.csv | csv | 81.7K | 124 | 124 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_green_ccpd2020_real.csv | csv | 297.2K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_green_bridge_exactquad.csv | csv | 221.3K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_blue_crpd_real.csv | csv | 507.9K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/proxy_blue_ccpd2019_real.csv | csv | 576.0K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/extreme_proxy_swap_mapping.csv | csv | 51.4K | 248 | 0 | 0 | 248 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/val_A1B.csv | csv | 2.4M | 3303 | 0 | 0 | 3303 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/val_A1.csv | csv | 2.4M | 3303 | 0 | 0 | 3303 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/val_A0.csv | csv | 2.4M | 3303 | 0 | 0 | 3303 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/train_A1B.csv | csv | 16.2M | 3319 | 0 | 0 | 3319 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/train_A1.csv | csv | 15.0M | 3320 | 0 | 0 | 3320 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/train_A0.csv | csv | 15.8M | 3317 | 0 | 0 | 3317 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/proxy_support_cblprd.csv | csv | 205.9K | 1200 | 0 | 0 | 1200 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/proxy_green_nonanhui_template_synth.csv | csv | 315.0K | 1500 | 0 | 0 | 1500 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/proxy_green_ccpd2020_real.csv | csv | 285.4K | 1001 | 0 | 0 | 1001 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/proxy_green_bridge_exactquad.csv | csv | 211.8K | 800 | 0 | 0 | 800 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/proxy_blue_crpd_real.csv | csv | 484.4K | 1882 | 0 | 0 | 1882 | 0 | high |
| manifests/curriculum_gray3_stagea_v3_realprimary/proxy_blue_ccpd2019_real.csv | csv | 552.4K | 2000 | 0 | 0 | 2000 | 0 | high |
| manifests/curriculum_gray3_stagea_v2_foundation/val_stageA_v2.csv | csv | 2.3M | 3312 | 0 | 0 | 3312 | 0 | high |
| manifests/curriculum_gray3_stagea_v2_foundation/train_stageA_v2.csv | csv | 11.6M | 3318 | 0 | 0 | 3318 | 0 | high |
| manifests/curriculum_gray3_stagea_v2_foundation/proxy_support.csv | csv | 223.7K | 1300 | 0 | 0 | 1300 | 0 | high |
| manifests/curriculum_gray3_stagea_v2_foundation/proxy_green_real_foundation.csv | csv | 30.2K | 105 | 0 | 0 | 105 | 0 | high |
| manifests/curriculum_gray3_stagea_v2_foundation/proxy_green_bridge.csv | csv | 134.4K | 500 | 0 | 0 | 500 | 0 | high |
| manifests/curriculum_gray3_stagea_v2_foundation/proxy_blue_real_foundation.csv | csv | 359.7K | 1363 | 0 | 0 | 1363 | 0 | high |
| manifests/curriculum_gray3_stagea_redesign/val.csv | csv | 1.9M | 3308 | 0 | 0 | 3308 | 0 | high |
| manifests/curriculum_gray3_stagea_redesign/train_stageA.csv | csv | 13.9M | 3327 | 0 | 0 | 3327 | 0 | high |
| manifests/curriculum_gray3_stagea_redesign/proxy_stageA_mixed_foundation.csv | csv | 534.0K | 2965 | 0 | 0 | 2965 | 0 | high |
| manifests/curriculum_gray3_stagea_redesign/proxy_stageA_green_simple.csv | csv | 166.4K | 1000 | 0 | 0 | 1000 | 0 | high |
| manifests/curriculum_gray3_stagea_redesign/proxy_stageA_blue_simple.csv | csv | 367.8K | 1965 | 0 | 0 | 1965 | 0 | high |
| manifests/curriculum_gray3_stageE_v2_balanced/val_E2.csv | csv | 2.6M | 3297 | 0 | 0 | 3297 | 0 | high |
| manifests/curriculum_gray3_stageE_v2_balanced/train_E2.csv | csv | 19.7M | 3318 | 0 | 0 | 3318 | 0 | high |
| manifests/curriculum_gray3_stageE_v1_extreme/val_E1.csv | csv | 2.6M | 3297 | 0 | 0 | 3297 | 0 | high |
| manifests/curriculum_gray3_stageE_v1_extreme/train_E1.csv | csv | 25.6M | 3323 | 0 | 0 | 3323 | 0 | high |
| manifests/curriculum_gray3_stageE_e3_main/val_e3_main.csv | csv | 2.6M | 3297 | 0 | 0 | 3297 | 0 | high |
| manifests/curriculum_gray3_stageE_e3_main/train_e3_main.csv | csv | 19.7M | 3318 | 0 | 0 | 3318 | 0 | high |
| manifests/curriculum_gray3_stageE_e3_control/val_e3_control.csv | csv | 2.6M | 3297 | 0 | 0 | 3297 | 0 | high |
| manifests/curriculum_gray3_stageE_e3_control/train_e3_control.csv | csv | 19.7M | 3318 | 0 | 0 | 3318 | 0 | high |
| manifests/curriculum_gray3/val_nonccpd_green.csv | csv | 874.1K | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/curriculum_gray3/val_crpd_blue.csv | csv | 590.2K | 2884 | 0 | 0 | 2884 | 0 | high |
| manifests/curriculum_gray3/val_ccpd2020_green.csv | csv | 197.6K | 833 | 0 | 0 | 833 | 0 | high |
| manifests/curriculum_gray3/val_ccpd2019_blue.csv | csv | 261.4K | 1144 | 0 | 0 | 1144 | 0 | high |
| manifests/curriculum_gray3/val_cblprd_blue.csv | csv | 217.1K | 1598 | 0 | 0 | 1598 | 0 | high |
| manifests/curriculum_gray3/val.csv | csv | 2.1M | 3309 | 0 | 0 | 3309 | 0 | high |
| manifests/curriculum_gray3/train_stageB.csv | csv | 17.8M | 3328 | 0 | 0 | 3328 | 0 | high |
| manifests/curriculum_gray3/train_stageA.csv | csv | 26.9M | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/curriculum_gray3/test_green_simple.csv | csv | 496.9K | 2130 | 0 | 0 | 2130 | 0 | high |
| manifests/curriculum_gray3/test_green_real.csv | csv | 1.7M | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/curriculum_gray3/test_green_hard.csv | csv | 75.6K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/curriculum_gray3/test_green_extreme.csv | csv | 31.5K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/curriculum_gray3/test_blue_simple.csv | csv | 454.0K | 1997 | 0 | 0 | 1997 | 0 | high |
| manifests/curriculum_gray3/test_blue_hard.csv | csv | 30.3M | 3331 | 0 | 0 | 3331 | 0 | high |
| manifests/crpd_yellow_train_only.csv | csv | 1.9M | 4501 | 4501 | 0 | 9002 | 0 | high |
| manifests/cluster_special_validation_v1/cluster_special_validation_v1.csv | csv | 45.8K | 124 | 0 | 0 | 124 | 0 | high |
| manifests/ccpd2020_replace_v1_obbquad/val_v1_obbquad.csv | csv | 99.0K | 300 | 0 | 0 | 300 | 0 | high |
| manifests/ccpd2020_replace_v1_obbquad/train_v1_obbquad.csv | csv | 899.0K | 2700 | 0 | 0 | 2700 | 0 | high |
| manifests/ccpd2020_replace_pose_v3/val_ccpd2020_replace_pose_v3.csv | csv | 105.5K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/ccpd2020_replace_pose_v3/train_ccpd2020_replace_pose_v3.csv | csv | 958.9K | 2790 | 0 | 0 | 2790 | 0 | high |
| manifests/ccpd2020_replace_extreme_v4/val_extreme_v3.csv | csv | 100.1K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/ccpd2020_replace_extreme_v4/train_extreme_v3.csv | csv | 301.8K | 930 | 0 | 0 | 930 | 0 | high |
| manifests/ccpd2020_replace_extreme_v3/val_extreme_v3.csv | csv | 99.8K | 310 | 0 | 0 | 310 | 0 | high |
| manifests/ccpd2020_replace_extreme_v3/train_extreme_v3.csv | csv | 302.2K | 930 | 0 | 0 | 930 | 0 | high |
| manifests/ccpd2020_replace_extreme_v2_additional/train_B2D_additional.csv | csv | 831.3K | 2700 | 0 | 0 | 2700 | 0 | high |
| manifests/ccpd2020_replace_extreme_v1/val_B2C_ccpd2020_replace_extreme.csv | csv | 87.3K | 300 | 0 | 0 | 300 | 0 | high |
| manifests/ccpd2020_replace_extreme_v1/train_B2C_ccpd2020_replace_extreme.csv | csv | 794.4K | 2700 | 0 | 0 | 2700 | 0 | high |
| manifests/Archive/unified_manifest_v4_board_aligned_real_only_crpd_raw.csv | csv | 262.3M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_v4_board_aligned_real_only.csv | csv | 264.2M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_v3.csv | csv | 267.5M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/Archive/unified_manifest_v2_with_pseudo_geom.csv | csv | 164.3M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/Archive/unified_manifest_v1.csv | csv | 159.0M | 3331 | 3331 | 0 | 3331 | 3331 | high |
| manifests/Archive/unified_manifest_green_specialist_official_v2_balanced_stable.csv | csv | 19.7M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_specialist_official_v1_balanced_train.csv | csv | 27.2M | 1081 | 5581 | 0 | 5702 | 960 | high |
| manifests/Archive/unified_manifest_green_specialist_official_v1_balanced_existing.csv | csv | 11.3M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_h30a_targeted_tail.csv | csv | 111.4M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_h20b_subtype_match.csv | csv | 10.3M | 2899 | 2899 | 0 | 5798 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv | csv | 110.9M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e7_boardnative_provbal.csv | csv | 117.3M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e6c_boardnative_append_v2_a800_20260413.csv | csv | 117.3M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e6c_boardnative_append_v1_20260413.csv | csv | 117.1M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e5_dumplike_boarddump_bright_v1_a3100_20260412.csv | csv | 118.0M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e4_extreme_append10_20260412.csv | csv | 117.2M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e3_equalprov_b_20260412.csv | csv | 117.0M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e3_equalprov_a_20260412.csv | csv | 120.3M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_plus_dumplike_boarddump_bright_v1_20260412.csv | csv | 117.1M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv | csv | 117.0M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed.csv | csv | 117.4M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative.csv | csv | 117.2M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v2.csv | csv | 116.2M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v4_realistic_b.csv | csv | 116.7M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v4_realistic.csv | csv | 116.4M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v3_zhe_guard_yuehu_restore.csv | csv | 117.4M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v2_zhe_guard.csv | csv | 116.5M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v1.csv | csv | 115.9M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak.csv | csv | 109.7M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_exact.csv | csv | 110.8M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40.csv | csv | 110.6M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_greenonly_b.csv | csv | 109.7M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_round2_conservative_a.csv | csv | 205.8M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_baseline_v1.csv | csv | 228.2M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_aggr_v1_existing_paths.csv | csv | 215.1M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_aggr_v1_existing_only.csv | csv | 229.7M | 3331 | 3331 | 0 | 6662 | 0 | high |
| manifests/Archive/unified_manifest_green_balance_aggr_v1.csv | csv | 231.4M | 3331 | 3331 | 0 | 6662 | 0 | high |

## 高风险 Manifest 分析

### manifests/Archive/unified_manifest_green_balance_aggr_v1.csv

- 类型: csv
- 大小: 231.4M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/C`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg`
  - `CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg`
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_501&458-0_0_33_18_25_26_26-166-27.jpg`
  - `CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_501&458-0_0_33_18_25_26_26-166-27.jpg`
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_483&477-0_0_2_27_9_26_24-178-36.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (231.4M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_aggr_v1_existing_only.csv

- 类型: csv
- 大小: 229.7M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/C`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg`
  - `CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg`
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_501&458-0_0_33_18_25_26_26-166-27.jpg`
  - `CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_501&458-0_0_33_18_25_26_26-166-27.jpg`
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_483&477-0_0_2_27_9_26_24-178-36.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (229.7M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_aggr_v1_existing_paths.csv

- 类型: csv
- 大小: 215.1M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/C`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg`
  - `CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg`
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_501&458-0_0_33_18_25_26_26-166-27.jpg`
  - `CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_501&458-0_0_33_18_25_26_26-166-27.jpg`
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_483&477-0_0_2_27_9_26_24-178-36.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (215.1M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_baseline_v1.csv

- 类型: csv
- 大小: 228.2M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/C`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg`
  - `CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg`
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_501&458-0_0_33_18_25_26_26-166-27.jpg`
  - `CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_501&458-0_0_33_18_25_26_26-166-27.jpg`
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_483&477-0_0_2_27_9_26_24-178-36.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (228.2M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_conservative_a.csv

- 类型: csv
- 大小: 205.8M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD2019`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD2019/ccpd_base`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg`
  - `CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg`
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_501&458-0_0_33_18_25_26_26-166-27.jpg`
  - `CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_501&458-0_0_33_18_25_26_26-166-27.jpg`
  - `/home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_483&477-0_0_2_27_9_26_24-178-36.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (205.8M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b.csv

- 类型: csv
- 大小: 109.7M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (109.7M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40.csv

- 类型: csv
- 大小: 110.6M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (110.6M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_exact.csv

- 类型: csv
- 大小: 110.8M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (110.8M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak.csv

- 类型: csv
- 大小: 109.7M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (109.7M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v1.csv

- 类型: csv
- 大小: 115.9M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (115.9M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v2_zhe_guard.csv

- 类型: csv
- 大小: 116.5M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (116.5M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v3_zhe_guard_yuehu_restore.csv

- 类型: csv
- 大小: 117.4M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (117.4M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v4_realistic.csv

- 类型: csv
- 大小: 116.4M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (116.4M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v4_realistic_b.csv

- 类型: csv
- 大小: 116.7M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (116.7M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v2.csv

- 类型: csv
- 大小: 116.2M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (116.2M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative.csv

- 类型: csv
- 大小: 117.2M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (117.2M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed.csv

- 类型: csv
- 大小: 117.4M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (117.4M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv

- 类型: csv
- 大小: 117.0M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (117.0M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_plus_dumplike_boarddump_bright_v1_20260412.csv

- 类型: csv
- 大小: 117.1M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (117.1M)，采样可能不完整

### manifests/Archive/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e3_equalprov_a_20260412.csv

- 类型: csv
- 大小: 120.3M
- 采样行数: 3332 (总计约3332)
- 绝对路径: 3331
- 有效路径: 6662
- 无效路径: 0
- 疑似 dataset_root: `/home/wzzz`
- 疑似 dataset_root: `/home/wzzz/LPRNet`
- 疑似 dataset_root: `/home/wzzz/LPRNet/CCPD20`

路径示例:
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg`
  - `/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg`

风险说明:
  - ⚠️ 含 3331 个绝对路径，迁移后失效风险高
  - ⚠️ 大文件 (120.3M)，采样可能不完整

## 路径类型分布汇总

| 类型 | 合计 |
|------|------|
| 绝对路径 | 754479 |
| 相对路径 | 373998 |
| 裸文件名 | 5 |
| URL | 0 |
| 空值 | 0 |
| 有效 | 1074897 |
| 无效 | 53585 |

## 风险与建议

### ⛔ 绝对路径风险 (发现 754479 个绝对路径)

绝对路径绑定到当前机器路径 `/home/wzzz/`。如果：
- 迁移到其他机器或容器
- 变更用户名或路径
- 在云端环境运行
则所有绝对路径立即失效。

**建议**: 统一改为 `dataset_root + relative_path` 格式。
对应的 dataset_root 可在训练配置中声明。

### ❌ 无效路径风险 (发现 53585 个无效路径)

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

*报告由 audit_manifest_paths.py 自动生成，440 份 manifest 已扫描*