# Manifest Rebase Dry-Run Report

生成时间: 2026-05-07 13:05:05
模式: DRY-RUN（仅采样分析，未生成任何新 manifest）

## 总览

| 指标 | 值 |
|------|-----|
| Auto-rebase manifest | 336 |
| 采样成功 | 336 |
| 采样失败 | 0 |
| Warning 总数 | 10 |
| 需验证 (validation_required) | 2 |

## 需验证的 Manifest

以下 manifest 旧根目录软链接缺失，但 canonical 路径存在，rebase 后应可修复：

- [manifests/unified_manifest_green_edgefit_v3_allprov.csv](manifests/unified_manifest_green_edgefit_v3_allprov.csv)
  - detected_root: datasets/green_edgefit_v3_allprov
  - broken_before: 3331, broken_after: 0
  - 旧软链接缺失，但 canonical datasets 路径存在，可通过 rebase 修复（需验证）

- [manifests/Archive/unified_manifest_green_specialist_official_v1_balanced_train.csv](manifests/Archive/unified_manifest_green_specialist_official_v1_balanced_train.csv)
  - detected_root: datasets/green_exact_quad_synthetic_v1
  - broken_before: 960, broken_after: 0
  - 旧软链接缺失，但 canonical datasets 路径存在，可通过 rebase 修复（需验证）

## manifests/

### unified_manifest_green_e12_pose_replace_test.csv

- total_lines=163213, abs=163213 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e12_replace_pose_v3_append.csv

- total_lines=324444, abs=324444 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### special_train.csv

- total_lines=28284, abs=28284 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000433593.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000433593.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000090844.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000090844.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000141451.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000141451.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000198418.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000198418.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000335423.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000335423.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000099698.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000099698.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000033535.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000033535.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000172586.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000172586.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000235896.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000235896.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000417737.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000417737.jpg

### yellow_single_train_weighted.csv

- total_lines=94834, abs=94834 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000390092.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000390092.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000237330.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000237330.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000348759.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000348759.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000089205.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000089205.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000084244.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000084244.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000362643.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000362643.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000149707.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000149707.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000438664.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000438664.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000417439.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000417439.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000450706.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000450706.jpg

### yellow_train.csv

- total_lines=54566, abs=54566 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000390092.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000390092.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000237330.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000237330.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000348759.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000348759.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000436719.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000436719.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000066216.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000066216.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000089205.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000089205.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000084244.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000084244.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000362643.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000362643.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000132724.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000132724.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000149707.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000149707.jpg

### yellow_train_weighted.csv

- total_lines=95075, abs=95075 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000390092.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000390092.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000237330.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000237330.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000348759.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000348759.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000436719.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000436719.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000066216.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000066216.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000089205.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000089205.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000084244.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000084244.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000362643.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000362643.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000132724.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000132724.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000149707.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000149707.jpg

### special_test.csv

- total_lines=1379, abs=1379 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000013734.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000013734.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000447115.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000447115.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000195930.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000195930.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000110760.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000110760.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000096531.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000096531.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000303273.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000303273.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000174312.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000174312.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000250927.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000250927.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000244054.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000244054.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000290045.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000290045.jpg

### yellow_real_val.csv

- total_lines=1420, abs=1420 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-323&667_454&
  After [0]: CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-323&667_454&711-323&667_454&66
  Before[1]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-280&682_410&
  After [1]: CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-280&682_410&728-280&685_407&68
  Before[2]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-620&741_732&
  After [2]: CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-620&741_732&779-620&741_732&74
  Before[3]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-524&751_636&
  After [3]: CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-524&751_636&789-524&752_636&75
  Before[4]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-578&324_677&
  After [4]: CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-578&324_677&359-581&324_677&32
  Before[5]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-416&694_558&
  After [5]: CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-416&694_558&737-418&694_557&69
  Before[6]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-485&693_635&
  After [6]: CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-485&693_635&743-485&694_635&69
  Before[7]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-404&528_536&
  After [7]: CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-404&528_536&575-405&529_535&52
  Before[8]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-1449&785_156
  After [8]: CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-1449&785_1569&828-1451&787_156
  Before[9]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-1365&825_148
  After [9]: CRPD_raw_ccpd_board_v1/yellow_single/val/crpd-raw-1365&825_1482&865-1367&825_148

### yellow_single_val.csv

- total_lines=2130, abs=2130 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000351364.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000351364.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000223461.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000223461.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000497800.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000497800.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000359754.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000359754.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000029082.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000029082.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000249407.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000249407.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000071236.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000071236.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000070443.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000070443.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000289867.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000289867.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000494393.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000494393.jpg

### yellow_test.csv

- total_lines=3878, abs=3878 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000208356.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000208356.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000351364.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000351364.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000223461.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000223461.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000403487.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000403487.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000186261.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000186261.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000461956.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000461956.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000056487.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000056487.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000497800.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000497800.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000359754.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000359754.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000233885.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000233885.jpg

### unified_manifest_official_gray3_bluegreen_u1c_trainable_nospecial.csv

- total_lines=577758, abs=577758 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_pos0_enhanced_v1_train.csv

- total_lines=502931, abs=308815 rel=194116
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_e10a_boarddump_exact_template_5prov_1800_replace.csv

- total_lines=318344, abs=318344 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e10b_boarddump_overflowfocus_5prov_1800_replace.csv

- total_lines=318344, abs=318344 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e11_e9c_append_aa0heavy_5prov_600.csv

- total_lines=320744, abs=320744 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e12_e9c_append_boarddump_anticollapse_5prov_1200.csv

- total_lines=321344, abs=321344 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e13a_e9c_append_slotalign_aa0_5prov_300.csv

- total_lines=320444, abs=320444 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e14a_image_local_probe_300_v1_fullrun_1776322133.csv

- total_lines=320444, abs=320444 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e14a_image_local_probe_probe123_fix2_1776320460.csv

- total_lines=320267, abs=320267 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e14a_image_local_probe_probe180_fix4_1776321624.csv

- total_lines=320324, abs=320324 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e15a_prewarp_slot_probe_300_v1.csv

- total_lines=320152, abs=320152 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e15a_prewarp_slot_probe_300_v1_fullrun_1776329788.csv

- total_lines=320444, abs=320444 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e16a_nonanhui_ad_balance_12k_std_v1.csv

- total_lines=332144, abs=332144 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e16b_nonanhui_ad_balance_12k_dump_v1.csv

- total_lines=332444, abs=332444 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e17a_cluster2_suffixbank_900.csv

- total_lines=322244, abs=322244 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e17b_cluster3_transition_900.csv

- total_lines=322244, abs=322244 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e17c_cluster3_tail_600.csv

- total_lines=321944, abs=321944 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e18b_su_bf_transition_dense_600.csv

- total_lines=321944, abs=321944 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e19c_su_bf_low_tail_dense_240.csv

- total_lines=322184, abs=322184 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e20a_cluster2_beijing_prefix_contrast_1200.csv

- total_lines=322544, abs=322544 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e25a_stageB_full_reintegrate.csv

- total_lines=30686, abs=30686 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_e26_cluster3_tail_boost_900.csv

- total_lines=322844, abs=322844 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e27_cluster3_hardtail_450.csv

- total_lines=323294, abs=323294 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e7_boardnative_v2.csv

- total_lines=321857, abs=321857 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_e8c_brightness_replace_5prov.csv

- total_lines=318344, abs=318344 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e9a_exact_template_5prov_1800.csv

- total_lines=320144, abs=320144 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e9b_exact_template_allprov_11160.csv

- total_lines=329504, abs=329504 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_e9c_exact_template_allprov_1800.csv

- total_lines=320144, abs=320144 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_green_h34d_v3_three_tiers.csv

- total_lines=226226, abs=226226 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [4]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [5]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [6]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [7]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [8]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00465711805556-89_90-284&470_400&511
  After [9]: CCPD2020/ccpd_green/train/00465711805556-89_90-284&470_400&511-400&508_286&511_2

### unified_manifest_pos0_enhanced_v1_eval.csv

- total_lines=664320, abs=664320 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554
  After [0]: CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_3
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519
  After [1]: CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_5
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_
  After [2]: CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_48
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520
  After [3]: CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520_256&454_239&349_5
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_
  After [4]: CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_95&595_85&525_268&
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&57
  After [5]: CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&578_300&589_297&523_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623
  After [6]: CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623_198&644_193&535_5
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_
  After [7]: CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_128&481_137&367_46
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&67
  After [8]: CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&675_526&696_519&637_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608
  After [9]: CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608_179&598_194&493_4

### unified_manifest_v4_real_only_plain_plate_val_proxy.csv

- total_lines=10281, abs=10281 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k/000351364.jpg
  After [0]: CBLPRD-330k_v1/CBLPRD-330k/000351364.jpg
  Before[1]: /home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k/000471563.jpg
  After [1]: CBLPRD-330k_v1/CBLPRD-330k/000471563.jpg
  Before[2]: /home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k/000258500.jpg
  After [2]: CBLPRD-330k_v1/CBLPRD-330k/000258500.jpg
  Before[3]: /home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k/000117095.jpg
  After [3]: CBLPRD-330k_v1/CBLPRD-330k/000117095.jpg
  Before[4]: /home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k/000300627.jpg
  After [4]: CBLPRD-330k_v1/CBLPRD-330k/000300627.jpg
  Before[5]: /home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k/000082237.jpg
  After [5]: CBLPRD-330k_v1/CBLPRD-330k/000082237.jpg
  Before[6]: /home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k/000391989.jpg
  After [6]: CBLPRD-330k_v1/CBLPRD-330k/000391989.jpg
  Before[7]: /home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k/000461956.jpg
  After [7]: CBLPRD-330k_v1/CBLPRD-330k/000461956.jpg
  Before[8]: /home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k/000109612.jpg
  After [8]: CBLPRD-330k_v1/CBLPRD-330k/000109612.jpg
  Before[9]: /home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k/000170488.jpg
  After [9]: CBLPRD-330k_v1/CBLPRD-330k/000170488.jpg

### unified_manifest_v4_real_only_test_ccpd_board.csv

- total_lines=160339, abs=160339 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_blur/0359-5_21-151&285_417&398-417&398_179&377_1
  After [0]: CCPD2019/ccpd_blur/0359-5_21-151&285_417&398-417&398_179&377_151&285_389&306-0_0
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_blur/0080-8_18-266&539_368&605-368&605_275&591_2
  After [1]: CCPD2019/ccpd_blur/0080-8_18-266&539_368&605-368&605_275&591_266&539_359&553-0_0
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_blur/0483-4_12-234&480_491&637-491&637_252&617_2
  After [2]: CCPD2019/ccpd_blur/0483-4_12-234&480_491&637-491&637_252&617_234&480_473&500-0_0
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_blur/0059-0_4-238&403_333&455-329&455_238&455_24
  After [3]: CCPD2019/ccpd_blur/0059-0_4-238&403_333&455-329&455_238&455_242&403_333&403-0_0_
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_blur/0567-11_23-279&509_587&663-565&607_279&663_
  After [4]: CCPD2019/ccpd_blur/0567-11_23-279&509_587&663-565&607_279&663_301&565_587&509-0_
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_blur/0486-0_1-206&421_539&543-536&541_206&543_20
  After [5]: CCPD2019/ccpd_blur/0486-0_1-206&421_539&543-536&541_206&543_209&423_539&421-0_0_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_blur/0152-0_1-286&486_476&553-475&551_286&553_28
  After [6]: CCPD2019/ccpd_blur/0152-0_1-286&486_476&553-475&551_286&553_287&488_476&486-0_0_
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_blur/0048-0_1-295&511_389&554-388&553_295&554_29
  After [7]: CCPD2019/ccpd_blur/0048-0_1-295&511_389&554-388&553_295&554_296&512_389&511-0_0_
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_blur/0374-0_5-206&453_522&552-522&551_215&552_20
  After [8]: CCPD2019/ccpd_blur/0374-0_5-206&453_522&552-522&551_215&552_206&454_513&453-0_0_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_blur/0215-1_6-294&500_513&582-513&582_301&577_29
  After [9]: CCPD2019/ccpd_blur/0215-1_6-294&500_513&582-513&582_301&577_294&500_506&505-0_0_

### unified_manifest_e10_selfcheck_a50_replace.csv

- total_lines=318344, abs=318344 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### unified_manifest_firstchar_patch_dataset_v1.csv

- total_lines=306750, abs=306750 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_lite_v1.csv

- total_lines=656922, abs=656922 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554
  After [0]: CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_3
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519
  After [1]: CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_5
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_
  After [2]: CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_48
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520
  After [3]: CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520_256&454_239&349_5
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_
  After [4]: CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_95&595_85&525_268&
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&57
  After [5]: CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&578_300&589_297&523_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623
  After [6]: CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623_198&644_193&535_5
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_
  After [7]: CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_128&481_137&367_46
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&67
  After [8]: CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&675_526&696_519&637_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608
  After [9]: CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608_179&598_194&493_4

### unified_manifest_green_balance_mid_v1.csv

- total_lines=667893, abs=667893 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554
  After [0]: CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_3
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519
  After [1]: CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_5
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_
  After [2]: CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_48
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520
  After [3]: CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520_256&454_239&349_5
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_
  After [4]: CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_95&595_85&525_268&
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&57
  After [5]: CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&578_300&589_297&523_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623
  After [6]: CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623_198&644_193&535_5
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_
  After [7]: CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_128&481_137&367_46
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&67
  After [8]: CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&675_526&696_519&637_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608
  After [9]: CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608_179&598_194&493_4

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_m1_v4_a3000.csv

- total_lines=319191, abs=319191 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### crpd_yellow_train_only.csv

- total_lines=4501, abs=4501 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-134&634_30
  After [0]: CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-134&634_303&687-134&634_303&
  Before[1]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-309&832_48
  After [1]: CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-309&832_484&886-309&833_484&
  Before[2]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-1175&124_1
  After [2]: CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-1175&124_1290&158-1175&124_1
  Before[3]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-1098&866_1
  After [3]: CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-1098&866_1233&908-1098&871_1
  Before[4]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-244&744_41
  After [4]: CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-244&744_412&797-245&744_412&
  Before[5]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-356&646_52
  After [5]: CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-356&646_524&696-356&646_524&
  Before[6]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-289&674_42
  After [6]: CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-289&674_422&715-289&674_422&
  Before[7]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-1109&665_1
  After [7]: CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-1109&665_1245&711-1110&665_1
  Before[8]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-137&349_24
  After [8]: CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-137&349_249&386-138&352_249&
  Before[9]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-94&666_231
  After [9]: CRPD_raw_ccpd_board_v1/yellow_single/train/crpd-raw-94&666_231&709-95&666_231&66

### unified_manifest_green_edgefit_v3_allprov.csv [VALIDATE]

- total_lines=10196, abs=10196 rel=0
- converted_paths_in_sample=10
- ⚠️   row 1:   WARNING: path not found: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0000-京LFQ5792-12&7_244&13_244&71_8&63.jpg
- ⚠️   row 2:   WARNING: path not found: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0001-京UD36101-10&3_243&9_244&71_12&62.jpg
- ⚠️   row 3:   WARNING: path not found: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0002-京ZFU6850-13&0_244&12_243&71_6&66.jpg
- ⚠️   row 4:   WARNING: path not found: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0003-京JDN2170-10&9_242&7_244&67_5&68.jpg
- ⚠️   row 5:   WARNING: path not found: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0004-京JDT0926-9&0_242&9_244&71_6&62.jpg
- ⚠️   row 6:   WARNING: path not found: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0005-京AFH3107-11&12_243&7_244&67_5&71.jpg
- ⚠️   row 7:   WARNING: path not found: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0006-京VD40842-13&10_240&11_243&70_12&71.jpg
- ⚠️   row 8:   WARNING: path not found: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0007-京BF16027-10&2_242&15_243&71_4&69.jpg
- ⚠️   row 9:   WARNING: path not found: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0008-京WD69083-2&4_240&6_244&67_15&69.jpg
- ⚠️   row 10:   WARNING: path not found: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-simple-京-0009-京HF53346-13&0_243&14_244&71_11&66.jpg
  Before[0]: /home/wzzz/LPRNet/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit
  After [0]: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-sim
  Before[1]: /home/wzzz/LPRNet/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit
  After [1]: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-sim
  Before[2]: /home/wzzz/LPRNet/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit
  After [2]: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-sim
  Before[3]: /home/wzzz/LPRNet/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit
  After [3]: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-sim
  Before[4]: /home/wzzz/LPRNet/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit
  After [4]: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-sim
  Before[5]: /home/wzzz/LPRNet/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit
  After [5]: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-sim
  Before[6]: /home/wzzz/LPRNet/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit
  After [6]: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-sim
  Before[7]: /home/wzzz/LPRNet/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit
  After [7]: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-sim
  Before[8]: /home/wzzz/LPRNet/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit
  After [8]: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-sim
  Before[9]: /home/wzzz/LPRNet/green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit
  After [9]: green_edgefit_v3_allprov/images/train/simple/p00_u4eac/edgefit3-simple-train-sim

### yellow_single_train.csv

- total_lines=59460, abs=59460 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000390092.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000390092.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000237330.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000237330.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000348759.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000348759.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000436719.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000436719.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000352310.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000352310.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000066216.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000066216.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000086008.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000086008.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000089205.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000089205.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000084244.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000084244.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000362643.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000362643.jpg

### yellow_real_test.csv

- total_lines=249, abs=249 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-182&755_333
  After [0]: CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-182&755_333&804-182&756_333&7
  Before[1]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-250&719_378
  After [1]: CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-250&719_378&761-251&719_372&7
  Before[2]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-493&684_632
  After [2]: CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-493&684_632&735-493&684_631&6
  Before[3]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-1379&753_14
  After [3]: CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-1379&753_1496&799-1379&758_14
  Before[4]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-246&832_361
  After [4]: CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-246&832_361&872-247&832_361&8
  Before[5]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-395&626_513
  After [5]: CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-395&626_513&664-397&626_513&6
  Before[6]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-425&660_565
  After [6]: CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-425&660_565&714-425&664_565&6
  Before[7]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-400&598_517
  After [7]: CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-400&598_517&640-400&598_517&5
  Before[8]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-315&717_444
  After [8]: CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-315&717_444&759-315&719_443&7
  Before[9]: /home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-363&702_487
  After [9]: CRPD_raw_ccpd_board_v1/yellow_single/test/crpd-raw-363&702_487&742-363&702_485&7

### yellow_single_hard_val.csv

- total_lines=933, abs=933 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000208356.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000208356.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000403487.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000403487.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000186261.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000186261.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000056487.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000056487.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000353280.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000353280.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000362591.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000362591.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000167792.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000167792.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000056076.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000056076.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000174102.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000174102.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000032761.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000032761.jpg

## manifests/Archive/

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv

- total_lines=308815, abs=308815 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_aggr_v1.csv

- total_lines=664320, abs=664320 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554
  After [0]: CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_3
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519
  After [1]: CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_5
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_
  After [2]: CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_48
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520
  After [3]: CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520_256&454_239&349_5
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_
  After [4]: CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_95&595_85&525_268&
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&57
  After [5]: CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&578_300&589_297&523_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623
  After [6]: CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623_198&644_193&535_5
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_
  After [7]: CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_128&481_137&367_46
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&67
  After [8]: CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&675_526&696_519&637_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608
  After [9]: CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608_179&598_194&493_4

### unified_manifest_green_balance_aggr_v1_existing_only.csv

- total_lines=659997, abs=659997 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554
  After [0]: CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_3
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519
  After [1]: CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_5
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_
  After [2]: CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_48
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520
  After [3]: CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520_256&454_239&349_5
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_
  After [4]: CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_95&595_85&525_268&
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&57
  After [5]: CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&578_300&589_297&523_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623
  After [6]: CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623_198&644_193&535_5
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_
  After [7]: CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_128&481_137&367_46
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&67
  After [8]: CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&675_526&696_519&637_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608
  After [9]: CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608_179&598_194&493_4

### unified_manifest_green_balance_aggr_v1_existing_paths.csv

- total_lines=622814, abs=622814 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554
  After [0]: CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_3
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519
  After [1]: CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_5
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_
  After [2]: CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_48
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520
  After [3]: CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520_256&454_239&349_5
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_
  After [4]: CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_95&595_85&525_268&
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&57
  After [5]: CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&578_300&589_297&523_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623
  After [6]: CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623_198&644_193&535_5
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_
  After [7]: CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_128&481_137&367_46
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&67
  After [8]: CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&675_526&696_519&637_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608
  After [9]: CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608_179&598_194&493_4

### unified_manifest_green_balance_baseline_v1.csv

- total_lines=656724, abs=656724 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554
  After [0]: CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_3
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519
  After [1]: CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_5
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_
  After [2]: CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_48
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520
  After [3]: CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520_256&454_239&349_5
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_
  After [4]: CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_95&595_85&525_268&
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&57
  After [5]: CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&578_300&589_297&523_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623
  After [6]: CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623_198&644_193&535_5
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_
  After [7]: CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_128&481_137&367_46
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&67
  After [8]: CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&675_526&696_519&637_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608
  After [9]: CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608_179&598_194&493_4

### unified_manifest_green_balance_round2_conservative_a.csv

- total_lines=578305, abs=578305 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554
  After [0]: CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_3
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519
  After [1]: CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_5
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_
  After [2]: CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_48
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520
  After [3]: CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520_256&454_239&349_5
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_
  After [4]: CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_95&595_85&525_268&
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&57
  After [5]: CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&578_300&589_297&523_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623
  After [6]: CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623_198&644_193&535_5
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_
  After [7]: CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_128&481_137&367_46
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&67
  After [8]: CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&675_526&696_519&637_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608
  After [9]: CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608_179&598_194&493_4

### unified_manifest_green_balance_round2_greenonly_b.csv

- total_lines=306051, abs=306051 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40.csv

- total_lines=308247, abs=308247 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_exact.csv

- total_lines=308815, abs=308815 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak.csv

- total_lines=306051, abs=306051 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v1.csv

- total_lines=319071, abs=319071 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v2_zhe_guard.csv

- total_lines=319071, abs=319071 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v3_zhe_guard_yuehu_restore.csv

- total_lines=319191, abs=319191 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v4_realistic.csv

- total_lines=319191, abs=319191 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v4_realistic_b.csv

- total_lines=319191, abs=319191 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v2.csv

- total_lines=319191, abs=319191 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative.csv

- total_lines=319191, abs=319191 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed.csv

- total_lines=319191, abs=319191 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv

- total_lines=319191, abs=319191 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_plus_dumplike_boarddump_bright_v1_20260412.csv

- total_lines=319591, abs=319591 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e3_equalprov_a_20260412.csv

- total_lines=326051, abs=326051 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e3_equalprov_b_20260412.csv

- total_lines=319191, abs=319191 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e4_extreme_append10_20260412.csv

- total_lines=319501, abs=319501 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e5_dumplike_boarddump_bright_v1_a3100_20260412.csv

- total_lines=322291, abs=322291 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e6c_boardnative_append_v1_20260413.csv

- total_lines=319401, abs=319401 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e6c_boardnative_append_v2_a800_20260413.csv

- total_lines=319991, abs=319991 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e7_boardnative_provbal.csv

- total_lines=319971, abs=319971 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_balance_round2_greenonly_b_h30a_targeted_tail.csv

- total_lines=310047, abs=310047 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### unified_manifest_green_specialist_official_v1_balanced_existing.csv

- total_lines=27997, abs=27997 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-69
  After [0]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-69&67_1051&372-70&11
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-83
  After [1]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-83&71_1037&439-84&18
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-52
  After [2]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-52&72_990&341-64&73_
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-58
  After [3]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-58&68_999&476-59&219
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-53
  After [4]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-53&73_977&327-53&73_
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-51
  After [5]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-51&86_993&343-52&86_
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-49
  After [6]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-49&73_1007&327-50&74
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-65
  After [7]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-65&79_1002&376-69&11
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-54
  After [8]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-54&78_1005&426-54&17
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-48
  After [9]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-48&84_995&393-48&85_

### unified_manifest_green_specialist_official_v1_balanced_train.csv [VALIDATE]

- total_lines=73637, abs=35317 rel=38320
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-69
  After [0]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-69&67_1051&372-70&11
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-83
  After [1]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-83&71_1037&439-84&18
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-52
  After [2]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-52&72_990&341-64&73_
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-58
  After [3]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-58&68_999&476-59&219
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-53
  After [4]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-53&73_977&327-53&73_
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-51
  After [5]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-51&86_993&343-52&86_
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-49
  After [6]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-49&73_1007&327-50&74
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-65
  After [7]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-65&79_1002&376-69&11
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-54
  After [8]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-54&78_1005&426-54&17
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-48
  After [9]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-48&84_995&393-48&85_

### unified_manifest_green_specialist_official_v2_balanced_stable.csv

- total_lines=55037, abs=55037 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-69
  After [0]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-69&67_1051&372-70&11
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-83
  After [1]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-83&71_1037&439-84&18
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-52
  After [2]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-52&72_990&341-64&73_
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-58
  After [3]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-58&68_999&476-59&219
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-53
  After [4]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-53&73_977&327-53&73_
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-51
  After [5]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-51&86_993&343-52&86_
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-49
  After [6]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-49&73_1007&327-50&74
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-65
  After [7]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-65&79_1002&376-69&11
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-54
  After [8]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-54&78_1005&426-54&17
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-48
  After [9]: green_exact_quad_synthetic_v1/images/train/p24_u4e91/genx-0-48&84_995&393-48&85_

### unified_manifest_v4_board_aligned_real_only.csv

- total_lines=781641, abs=781641 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554
  After [0]: CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_3
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519
  After [1]: CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_5
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_
  After [2]: CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_48
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520
  After [3]: CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520_256&454_239&349_5
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_
  After [4]: CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_95&595_85&525_268&
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&57
  After [5]: CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&578_300&589_297&523_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623
  After [6]: CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623_198&644_193&535_5
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_
  After [7]: CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_128&481_137&367_46
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&67
  After [8]: CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&675_526&696_519&637_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608
  After [9]: CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608_179&598_194&493_4

### unified_manifest_v4_board_aligned_real_only_crpd_raw.csv

- total_lines=781641, abs=781641 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554
  After [0]: CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_3
  Before[1]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519
  After [1]: CCPD2019/ccpd_base/0104418103448-91_84-329&442_511&520-515&519_340&508_326&447_5
  Before[2]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_
  After [2]: CCPD2019/ccpd_base/023275862069-90_86-173&473_468&557-485&563_189&555_187&469_48
  Before[3]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520
  After [3]: CCPD2019/ccpd_base/0344827586207-92_75-255&369_564&505-560&520_256&454_239&349_5
  Before[4]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_
  After [4]: CCPD2019/ccpd_base/0144516283524-97_72-90&538_280&616-278&629_95&595_85&525_268&
  Before[5]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&57
  After [5]: CCPD2019/ccpd_base/00885536398467-90_89-301&521_492&580-501&578_300&589_297&523_
  Before[6]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623
  After [6]: CCPD2019/ccpd_base/0288697318007-88_89-195&525_496&636-508&623_198&644_193&535_5
  Before[7]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_
  After [7]: CCPD2019/ccpd_base/048429118774-85_96-114&333_470&486-451&443_128&481_137&367_46
  Before[8]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&67
  After [8]: CCPD2019/ccpd_base/00865900383142-83_97-516&622_643&695-642&675_526&696_519&637_
  Before[9]: /home/wzzz/LPRNet/CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608
  After [9]: CCPD2019/ccpd_base/0274125957855-94_83-167&496_450&588-439&608_179&598_194&493_4

## manifests/ccpd2020_replace_extreme_v1/

### train_B2C_ccpd2020_replace_extreme.csv

- total_lines=2700, abs=2700 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/train/306717447916
  After [0]: datasets/ccpd2020_replace_extreme_v1/images/train/306717447916666666-93_254-131&
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/train/040753113026
  After [1]: datasets/ccpd2020_replace_extreme_v1/images/train/04075311302681992-88_260-102&5
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/train/045562739463
  After [2]: datasets/ccpd2020_replace_extreme_v1/images/train/04556273946360153-92_251-168&5
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/train/033845785440
  After [3]: datasets/ccpd2020_replace_extreme_v1/images/train/03384578544061303-90_230-166&4
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/train/031453544061
  After [4]: datasets/ccpd2020_replace_extreme_v1/images/train/031453544061302685-111_286-281
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/train/054310344827
  After [5]: datasets/ccpd2020_replace_extreme_v1/images/train/054310344827586204-90_233-44&4
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/train/046623263888
  After [6]: datasets/ccpd2020_replace_extreme_v1/images/train/0466232638889-83_251-227&491_5
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/train/046780411877
  After [7]: datasets/ccpd2020_replace_extreme_v1/images/train/04678041187739464-90_249-146&5
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/train/054695881226
  After [8]: datasets/ccpd2020_replace_extreme_v1/images/train/05469588122605364-90_251-101&4
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/train/055689655172
  After [9]: datasets/ccpd2020_replace_extreme_v1/images/train/055689655172413796-96_233-219&

### val_B2C_ccpd2020_replace_extreme.csv

- total_lines=300, abs=300 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/val/04445043103448
  After [0]: datasets/ccpd2020_replace_extreme_v1/images/val/044450431034482756-93_240-179&47
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/val/04385416666666
  After [1]: datasets/ccpd2020_replace_extreme_v1/images/val/043854166666666666-91_241-147&49
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/val/03521312260536
  After [2]: datasets/ccpd2020_replace_extreme_v1/images/val/03521312260536399-90_266-241&513
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/val/04568965517241
  After [3]: datasets/ccpd2020_replace_extreme_v1/images/val/045689655172413794-90_251-130&50
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/val/03903256704980
  After [4]: datasets/ccpd2020_replace_extreme_v1/images/val/03903256704980843-89_260-133&588
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/val/02065613026819
  After [5]: datasets/ccpd2020_replace_extreme_v1/images/val/020656130268199235-93_257-226&49
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/val/04238386015325
  After [6]: datasets/ccpd2020_replace_extreme_v1/images/val/04238386015325671-90_240-180&523
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/val/00989942528735
  After [7]: datasets/ccpd2020_replace_extreme_v1/images/val/009899425287356323-93_262-285&51
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/val/05796216475095
  After [8]: datasets/ccpd2020_replace_extreme_v1/images/val/057962164750957855-92_252-143&48
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v1/images/val/04332016283524
  After [9]: datasets/ccpd2020_replace_extreme_v1/images/val/04332016283524904-92_242-169&511

## manifests/ccpd2020_replace_extreme_v2_additional/

### train_B2D_additional.csv

- total_lines=2700, abs=2700 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v2_additional/images/train/0
  After [0]: datasets/ccpd2020_replace_extreme_v2_additional/images/train/03380208333333334-1
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v2_additional/images/train/0
  After [1]: datasets/ccpd2020_replace_extreme_v2_additional/images/train/0682219827586207-95
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v2_additional/images/train/0
  After [2]: datasets/ccpd2020_replace_extreme_v2_additional/images/train/031376953125-99_255
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v2_additional/images/train/0
  After [3]: datasets/ccpd2020_replace_extreme_v2_additional/images/train/03878831417624521-9
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v2_additional/images/train/0
  After [4]: datasets/ccpd2020_replace_extreme_v2_additional/images/train/06689655172413793-9
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v2_additional/images/train/0
  After [5]: datasets/ccpd2020_replace_extreme_v2_additional/images/train/0348742816091954-90
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v2_additional/images/train/0
  After [6]: datasets/ccpd2020_replace_extreme_v2_additional/images/train/038308189655172416-
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v2_additional/images/train/0
  After [7]: datasets/ccpd2020_replace_extreme_v2_additional/images/train/05647749042145594-9
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v2_additional/images/train/0
  After [8]: datasets/ccpd2020_replace_extreme_v2_additional/images/train/04180675287356322-8
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v2_additional/images/train/0
  After [9]: datasets/ccpd2020_replace_extreme_v2_additional/images/train/03668103448275862-8

## manifests/ccpd2020_replace_extreme_v3/

### train_extreme_v3.csv

- total_lines=930, abs=930 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/train/042241379310
  After [0]: datasets/ccpd2020_replace_extreme_v3/images/train/04224137931034483-92_237-136&5
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/train/042241379310
  After [1]: datasets/ccpd2020_replace_extreme_v3/images/train/04224137931034483-92_237-136&5
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/train/042241379310
  After [2]: datasets/ccpd2020_replace_extreme_v3/images/train/04224137931034483-92_237-136&5
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/train/042241379310
  After [3]: datasets/ccpd2020_replace_extreme_v3/images/train/04224137931034483-92_237-136&5
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/train/042241379310
  After [4]: datasets/ccpd2020_replace_extreme_v3/images/train/04224137931034483-92_237-136&5
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/train/042241379310
  After [5]: datasets/ccpd2020_replace_extreme_v3/images/train/04224137931034483-92_237-136&5
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/train/042241379310
  After [6]: datasets/ccpd2020_replace_extreme_v3/images/train/04224137931034483-92_237-136&5
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/train/042241379310
  After [7]: datasets/ccpd2020_replace_extreme_v3/images/train/04224137931034483-92_237-136&5
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/train/042241379310
  After [8]: datasets/ccpd2020_replace_extreme_v3/images/train/04224137931034483-92_237-136&5
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/train/042241379310
  After [9]: datasets/ccpd2020_replace_extreme_v3/images/train/04224137931034483-92_237-136&5

### val_extreme_v3.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/val/04388888888888
  After [0]: datasets/ccpd2020_replace_extreme_v3/images/val/04388888888888889-98_232-201&456
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/val/0348046875-94_
  After [1]: datasets/ccpd2020_replace_extreme_v3/images/val/0348046875-94_241-216&458_513&57
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/val/0548697916667-
  After [2]: datasets/ccpd2020_replace_extreme_v3/images/val/0548697916667-87_254-159&420_551
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/val/30301388888888
  After [3]: datasets/ccpd2020_replace_extreme_v3/images/val/303013888888888889-80_104-253&42
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/val/03603208812260
  After [4]: datasets/ccpd2020_replace_extreme_v3/images/val/03603208812260537-89_248-127&516
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/val/0387868923611-
  After [5]: datasets/ccpd2020_replace_extreme_v3/images/val/0387868923611-99_108-234&423_527
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/val/02452705938697
  After [6]: datasets/ccpd2020_replace_extreme_v3/images/val/02452705938697318-93_252-203&504
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/val/057421875-96_2
  After [7]: datasets/ccpd2020_replace_extreme_v3/images/val/057421875-96_249-186&466_564&619
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/val/04242337164750
  After [8]: datasets/ccpd2020_replace_extreme_v3/images/val/042423371647509575-91_248-179&55
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v3/images/val/03892241379310
  After [9]: datasets/ccpd2020_replace_extreme_v3/images/val/03892241379310345-88_254-176&554

## manifests/ccpd2020_replace_extreme_v4/

### train_extreme_v3.csv

- total_lines=930, abs=930 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/train/304921875-95
  After [0]: datasets/ccpd2020_replace_extreme_v4/images/train/304921875-95_241-154&445_514&5
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/train/051903735632
  After [1]: datasets/ccpd2020_replace_extreme_v4/images/train/05190373563218391-90_229-80&54
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/train/038009982638
  After [2]: datasets/ccpd2020_replace_extreme_v4/images/train/0380099826389-95_251-121&398_4
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/train/034559386973
  After [3]: datasets/ccpd2020_replace_extreme_v4/images/train/034559386973180076-90_250-181&
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/train/035739942528
  After [4]: datasets/ccpd2020_replace_extreme_v4/images/train/035739942528735635-90_240-290&
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/train/041307471264
  After [5]: datasets/ccpd2020_replace_extreme_v4/images/train/04130747126436782-89_242-148&4
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/train/032890325670
  After [6]: datasets/ccpd2020_replace_extreme_v4/images/train/032890325670498086-89_251-241&
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/train/024527059386
  After [7]: datasets/ccpd2020_replace_extreme_v4/images/train/02452705938697318-93_252-203&5
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/train/010470545977
  After [8]: datasets/ccpd2020_replace_extreme_v4/images/train/010470545977011494-91_256-201&
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/train/053002873563
  After [9]: datasets/ccpd2020_replace_extreme_v4/images/train/05300287356321839-87_255-164&4

### val_extreme_v3.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034
  After [0]: datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034483-92_237-136&500
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034
  After [1]: datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034483-92_237-136&500
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034
  After [2]: datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034483-92_237-136&500
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034
  After [3]: datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034483-92_237-136&500
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034
  After [4]: datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034483-92_237-136&500
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034
  After [5]: datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034483-92_237-136&500
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034
  After [6]: datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034483-92_237-136&500
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034
  After [7]: datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034483-92_237-136&500
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034
  After [8]: datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034483-92_237-136&500
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034
  After [9]: datasets/ccpd2020_replace_extreme_v4/images/val/04224137931034483-92_237-136&500

## manifests/ccpd2020_replace_pose_v3/

### train_ccpd2020_replace_pose_v3.csv

- total_lines=2790, abs=2790 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/0241276041667-1
  After [0]: datasets/ccpd2020_replace_pose_v3/images/train/0241276041667-100_256-258&418_462
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/039128352490421
  After [1]: datasets/ccpd2020_replace_pose_v3/images/train/03912835249042146-90_262-202&523_
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/051494252873563
  After [2]: datasets/ccpd2020_replace_pose_v3/images/train/05149425287356322-115_231-300&403
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/033845785440613
  After [3]: datasets/ccpd2020_replace_pose_v3/images/train/03384578544061303-90_230-166&478_
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/0362782118056-7
  After [4]: datasets/ccpd2020_replace_pose_v3/images/train/0362782118056-71_127-318&462_547&
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/036027298850574
  After [5]: datasets/ccpd2020_replace_pose_v3/images/train/036027298850574714-90_263-200&513
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/050056273946360
  After [6]: datasets/ccpd2020_replace_pose_v3/images/train/050056273946360155-93_225-132&490
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/060798611111111
  After [7]: datasets/ccpd2020_replace_pose_v3/images/train/06079861111111111-91_242-112&513_
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/055050287356321
  After [8]: datasets/ccpd2020_replace_pose_v3/images/train/055050287356321836-122_300-190&40
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/040330459770114
  After [9]: datasets/ccpd2020_replace_pose_v3/images/train/04033045977011494-91_226-137&515_

### val_ccpd2020_replace_pose_v3.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/03625718390804598
  After [0]: datasets/ccpd2020_replace_pose_v3/images/val/03625718390804598-84_225-147&494_45
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/04622844827586207
  After [1]: datasets/ccpd2020_replace_pose_v3/images/val/04622844827586207-90_228-159&467_58
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/01020833333333333
  After [2]: datasets/ccpd2020_replace_pose_v3/images/val/010208333333333333-87_251-336&496_4
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/03235991379310345
  After [3]: datasets/ccpd2020_replace_pose_v3/images/val/03235991379310345-90_245-191&523_54
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/04471384099616858
  After [4]: datasets/ccpd2020_replace_pose_v3/images/val/04471384099616858-94_226-221&494_60
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/03625718390804598
  After [5]: datasets/ccpd2020_replace_pose_v3/images/val/03625718390804598-84_225-147&494_45
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/04166666666666666
  After [6]: datasets/ccpd2020_replace_pose_v3/images/val/041666666666666664-90_263-127&511_4
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/02487428160919540
  After [7]: datasets/ccpd2020_replace_pose_v3/images/val/024874281609195404-88_229-207&489_4
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/05005627394636015
  After [8]: datasets/ccpd2020_replace_pose_v3/images/val/050056273946360155-93_225-132&490_5
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/10128591954022989
  After [9]: datasets/ccpd2020_replace_pose_v3/images/val/10128591954022989-94_219-48&425_661

## manifests/ccpd2020_replace_v1_obbquad/

### train_v1_obbquad.csv

- total_lines=2700, abs=2700 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_0
  After [0]: datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_000001_replaced_宁PD
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_0
  After [1]: datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_000002_replaced_鄂CD
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_0
  After [2]: datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_000003_replaced_云LF
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_0
  After [3]: datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_000004_replaced_宁AF
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_0
  After [4]: datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_000005_replaced_陕AF
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_0
  After [5]: datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_000006_replaced_京AF
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_0
  After [6]: datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_000007_replaced_豫SF
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_0
  After [7]: datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_000008_replaced_黑YD
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_0
  After [8]: datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_000009_replaced_琼PD
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_0
  After [9]: datasets/ccpd2020_replace_v1_obbquad/images/train/obbquad_v1_000010_replaced_陕UD

### val_v1_obbquad.csv

- total_lines=300, abs=300 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000
  After [0]: datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000001_replaced_浙LDF1
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000
  After [1]: datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000002_replaced_赣JF57
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000
  After [2]: datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000003_replaced_赣HD90
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000
  After [3]: datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000004_replaced_晋QDQ7
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000
  After [4]: datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000005_replaced_黑HFF9
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000
  After [5]: datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000006_replaced_鲁CF89
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000
  After [6]: datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000007_replaced_闽NFV4
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000
  After [7]: datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000008_replaced_新DF08
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000
  After [8]: datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000009_replaced_鄂SFA9
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000
  After [9]: datasets/ccpd2020_replace_v1_obbquad/images/val/obbquad_v1_000010_replaced_京XDU7

## manifests/curriculum_gray3/

### val.csv

- total_lines=12000, abs=12000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0049.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_0049.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000057318.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000057318.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p00_u4eac/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p00_u4eac/genx-0-52&87_993&476
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0125.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/64_0125.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0170869252874-89_88-223&374_456&45
  After [4]: datasets/CCPD2019/ccpd_base/0170869252874-89_88-223&374_456&453-451&445_238&450_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0136314655172-90_85-262&565_480&63
  After [5]: datasets/CCPD2019/ccpd_base/0136314655172-90_85-262&565_480&634-479&635_268&630_
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p26_u9655/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p26_u9655/genx-0-50&80_997&459
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0262787356322-96_78-220&436_481&55
  After [7]: datasets/CCPD2019/ccpd_base/0262787356322-96_78-220&436_481&559-484&565_224&529_
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000008476.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000008476.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p15_u9c
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p15_u9c81/edgefit-tier3-1

### train_stageA.csv

- total_lines=160000, abs=160000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000376248.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000376248.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000278714.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000278714.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000407437.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000407437.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000368423.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000368423.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000422813.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000422813.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026875-87_86-199&472_482&584-469&5
  After [5]: datasets/CCPD2019/ccpd_base/026875-87_86-199&472_482&584-469&564_204&581_202&492
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000051346.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000051346.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/47_0652.jpg
  After [7]: datasets/CRPD_all/CRPD_single/train/images/47_0652.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_multi/train/images/39_0740.jpg
  After [8]: datasets/CRPD_all/CRPD_multi/train/images/39_0740.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000401018.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000401018.jpg

### train_stageB.csv

- total_lines=162208, abs=162208 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000164281.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000164281.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000008073.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000008073.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/43_0741.jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/43_0741.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000429990.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000429990.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000081852.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000081852.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p18_u6e58/
  After [5]: datasets/green_exact_quad_synthetic_v1/images/train/p18_u6e58/genx-0-67&78_1000&
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000384400.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000384400.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000350851.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000350851.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000096612.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000096612.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000026550.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000026550.jpg

### test_blue_hard.csv

- total_lines=141982, abs=141982 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_blur/0359-5_21-151&285_417&398-417&398_
  After [0]: datasets/CCPD2019/ccpd_blur/0359-5_21-151&285_417&398-417&398_179&377_151&285_38
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_blur/0080-8_18-266&539_368&605-368&605_
  After [1]: datasets/CCPD2019/ccpd_blur/0080-8_18-266&539_368&605-368&605_275&591_266&539_35
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_blur/0483-4_12-234&480_491&637-491&637_
  After [2]: datasets/CCPD2019/ccpd_blur/0483-4_12-234&480_491&637-491&637_252&617_234&480_47
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_blur/0059-0_4-238&403_333&455-329&455_2
  After [3]: datasets/CCPD2019/ccpd_blur/0059-0_4-238&403_333&455-329&455_238&455_242&403_333
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_blur/0567-11_23-279&509_587&663-565&607
  After [4]: datasets/CCPD2019/ccpd_blur/0567-11_23-279&509_587&663-565&607_279&663_301&565_5
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_blur/0486-0_1-206&421_539&543-536&541_2
  After [5]: datasets/CCPD2019/ccpd_blur/0486-0_1-206&421_539&543-536&541_206&543_209&423_539
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_blur/0152-0_1-286&486_476&553-475&551_2
  After [6]: datasets/CCPD2019/ccpd_blur/0152-0_1-286&486_476&553-475&551_286&553_287&488_476
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_blur/0048-0_1-295&511_389&554-388&553_2
  After [7]: datasets/CCPD2019/ccpd_blur/0048-0_1-295&511_389&554-388&553_295&554_296&512_389
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_blur/0374-0_5-206&453_522&552-522&551_2
  After [8]: datasets/CCPD2019/ccpd_blur/0374-0_5-206&453_522&552-522&551_215&552_206&454_513
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_blur/0215-1_6-294&500_513&582-513&582_3
  After [9]: datasets/CCPD2019/ccpd_blur/0215-1_6-294&500_513&582-513&582_301&577_294&500_506

### test_blue_simple.csv

- total_lines=1997, abs=1997 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0259602490421-90_90-112&408_429&50
  After [0]: datasets/CCPD2019/ccpd_base/0259602490421-90_90-112&408_429&508-423&504_122&498_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0407519157088-99_75-274&334_550&48
  After [1]: datasets/CCPD2019/ccpd_base/0407519157088-99_75-274&334_550&484-566&498_273&423_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0148060344828-91_88-137&350_399&41
  After [2]: datasets/CCPD2019/ccpd_base/0148060344828-91_88-137&350_399&419-399&425_131&414_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0226041666667-97_79-236&398_472&51
  After [3]: datasets/CCPD2019/ccpd_base/0226041666667-97_79-236&398_472&511-475&512_240&474_
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0174329501916-92_80-288&604_521&68
  After [4]: datasets/CCPD2019/ccpd_base/0174329501916-92_80-288&604_521&688-511&688_285&676_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0126724137931-90_90-212&473_422&54
  After [5]: datasets/CCPD2019/ccpd_base/0126724137931-90_90-212&473_422&543-423&553_200&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0186446360153-90_88-236&442_518&52
  After [6]: datasets/CCPD2019/ccpd_base/0186446360153-90_88-236&442_518&520-521&529_228&536_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0270833333333-91_86-151&466_458&57
  After [7]: datasets/CCPD2019/ccpd_base/0270833333333-91_86-151&466_458&575-457&571_164&563_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0256465517241-90_88-183&422_507&51
  After [8]: datasets/CCPD2019/ccpd_base/0256465517241-90_88-183&422_507&512-514&507_180&514_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0258429118774-90_93-182&494_482&58
  After [9]: datasets/CCPD2019/ccpd_base/0258429118774-90_93-182&494_482&588-478&598_176&589_

### test_green_real.csv

- total_lines=9058, abs=9058 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/test/0014128352490421455-90_90-21
  After [0]: datasets/CCPD2020/ccpd_green/test/0014128352490421455-90_90-212&467_271&489-271&
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/test/0016594827586206896-90_90-34
  After [1]: datasets/CCPD2020/ccpd_green/test/0016594827586206896-90_90-341&550_407&573-407&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/test/0017959770114942528-89_265-2
  After [2]: datasets/CCPD2020/ccpd_green/test/0017959770114942528-89_265-240&542_315&564-313
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/test/0021264367816091955-90_267-3
  After [3]: datasets/CCPD2020/ccpd_green/test/0021264367816091955-90_267-311&542_385&569-385
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/test/0021264367816091955-92_264-3
  After [4]: datasets/CCPD2020/ccpd_green/test/0021264367816091955-92_264-333&534_407&560-405
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/test/0021479885057471265-90_262-3
  After [5]: datasets/CCPD2020/ccpd_green/test/0021479885057471265-90_262-315&502_393&527-393
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/test/002454501915708812-91_264-31
  After [6]: datasets/CCPD2020/ccpd_green/test/002454501915708812-91_264-310&567_392&595-391&
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/test/0026041666666666665-91_265-3
  After [7]: datasets/CCPD2020/ccpd_green/test/0026041666666666665-91_265-360&505_435&537-435
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/test/0026484674329501916-93_90-28
  After [8]: datasets/CCPD2020/ccpd_green/test/0026484674329501916-93_90-283&513_362&543-360&
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/test/00272389846743295-90_90-371&
  After [9]: datasets/CCPD2020/ccpd_green/test/00272389846743295-90_90-371&537_436&575-436&57

### test_green_simple.csv

- total_lines=2130, abs=2130 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/test/p02_u6d25/g
  After [0]: datasets/green_exact_quad_synthetic_v1/images/test/p02_u6d25/genx-0-50&69_986&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/test/p04_u5180/g
  After [1]: datasets/green_exact_quad_synthetic_v1/images/test/p04_u5180/genx-0-51&70_984&38
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/test/p13_u95fd/g
  After [2]: datasets/green_exact_quad_synthetic_v1/images/test/p13_u95fd/genx-0-89&87_1026&4
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/test/p18_u6e58/g
  After [3]: datasets/green_exact_quad_synthetic_v1/images/test/p18_u6e58/genx-0-58&69_990&44
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/test/p17_u9102/g
  After [4]: datasets/green_exact_quad_synthetic_v1/images/test/p17_u9102/genx-0-50&58_1002&3
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/test/p03_u6e1d/g
  After [5]: datasets/green_exact_quad_synthetic_v1/images/test/p03_u6e1d/genx-0-50&68_981&32
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/test/p28_u9752/g
  After [6]: datasets/green_exact_quad_synthetic_v1/images/test/p28_u9752/genx-0-59&95_997&47
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/test/p22_u5ddd/g
  After [7]: datasets/green_exact_quad_synthetic_v1/images/test/p22_u5ddd/genx-0-52&71_998&35
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/test/p24_u4e91/g
  After [8]: datasets/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&3
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/test/p11_u6d59/g
  After [9]: datasets/green_exact_quad_synthetic_v1/images/test/p11_u6d59/genx-0-49&70_990&32

### val_cblprd_blue.csv

- total_lines=1598, abs=1598 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000319385.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000319385.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000208875.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000208875.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000280476.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000280476.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000249390.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000249390.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000017922.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000017922.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462028.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462028.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000243336.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000243336.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000000280.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000000280.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000122483.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000122483.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000115850.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000115850.jpg

### val_ccpd2019_blue.csv

- total_lines=1144, abs=1144 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0129909003831-90_85-301&372_485&45
  After [0]: datasets/CCPD2019/ccpd_base/0129909003831-90_85-301&372_485&453-486&448_309&440_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0503663793103-86_91-255&309_611&44
  After [1]: datasets/CCPD2019/ccpd_base/0503663793103-86_91-255&309_611&446-615&427_253&450_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0173096264367-83_98-122&433_320&52
  After [2]: datasets/CCPD2019/ccpd_base/0173096264367-83_98-122&433_320&521-328&491_129&518_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0139463601533-92_84-307&610_506&68
  After [3]: datasets/CCPD2019/ccpd_base/0139463601533-92_84-307&610_506&688-507&687_307&679_
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0186949233717-96_83-173&409_394&51
  After [4]: datasets/CCPD2019/ccpd_base/0186949233717-96_83-173&409_394&510-399&511_177&481_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0275251436781-85_94-237&463_489&56
  After [5]: datasets/CCPD2019/ccpd_base/0275251436781-85_94-237&463_489&569-491&539_249&570_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00651939655172-91_89-303&483_454&5
  After [6]: datasets/CCPD2019/ccpd_base/00651939655172-91_89-303&483_454&544-462&549_304&536
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0223958333333-96_74-288&621_519&72
  After [7]: datasets/CCPD2019/ccpd_base/0223958333333-96_74-288&621_519&725-513&719_302&693_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0234375-90_92-276&502_562&603-560&
  After [8]: datasets/CCPD2019/ccpd_base/0234375-90_92-276&502_562&603-560&600_272&602_274&49
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0262787356322-96_78-220&436_481&55
  After [9]: datasets/CCPD2019/ccpd_base/0262787356322-96_78-220&436_481&559-484&565_224&529_

### val_crpd_blue.csv

- total_lines=3164, abs=3164 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1053.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/48_1053.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/48_0895.jpg
  After [1]: datasets/CRPD_all/CRPD_double/val/images/48_0895.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/48_0096.jpg
  After [2]: datasets/CRPD_all/CRPD_double/val/images/48_0096.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/50_0940.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/50_0940.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_1031.jpg
  After [4]: datasets/CRPD_all/CRPD_double/val/images/43_1031.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/48_0761.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/48_0761.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_0978.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/48_0978.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0090.jpg
  After [7]: datasets/CRPD_all/CRPD_double/val/images/43_0090.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0036.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/37_0036.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/47_0657.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/47_0657.jpg

### val_nonccpd_green.csv

- total_lines=5167, abs=5167 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000465970.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000465970.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p02_u6d
  After [1]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p02_u6d25/edgefit-tier3-1
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000025542.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000025542.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000293601.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000293601.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000223573.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000223573.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000117020.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000117020.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000282945.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000282945.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000134092.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000134092.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000403029.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000403029.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p19_u7ca4/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p19_u7ca4/genx-0-53&69_973&372

### test_green_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p00_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p00_u4eac/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p00_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p00_u4eac/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p00_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p00_u4eac/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p00_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p00_u4eac/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p02_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p02_u6d25/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p02_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p02_u6d25/edgefit-tier3

### test_green_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-13
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-16
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-9&
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-15
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-15
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-11
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-20
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-23
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-14

### val_ccpd2020_green.csv

- total_lines=833, abs=833 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0782248263889-93_251-116&509_
  After [0]: datasets/CCPD2020/ccpd_green/val/0782248263889-93_251-116&509_654&656-649&656_14
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0462152777778-86_269-215&419_
  After [1]: datasets/CCPD2020/ccpd_green/val/0462152777778-86_269-215&419_567&551-567&523_22
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03249348958333333-93_260-166&
  After [2]: datasets/CCPD2020/ccpd_green/val/03249348958333333-93_260-166&467_488&569-486&56
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/01880859375-88_90-224&465_438
  After [3]: datasets/CCPD2020/ccpd_green/val/01880859375-88_90-224&465_438&553-432&534_224&5
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/304039713541666667-92_96-120&
  After [4]: datasets/CCPD2020/ccpd_green/val/304039713541666667-92_96-120&444_485&555-485&55
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0249251302083-89_261-188&461_
  After [5]: datasets/CCPD2020/ccpd_green/val/0249251302083-89_261-188&461_435&563-431&544_18
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0270290798611-94_98-222&434_4
  After [6]: datasets/CCPD2020/ccpd_green/val/0270290798611-94_98-222&434_487&537-487&537_222
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0301909722222-94_99-202&396_4
  After [7]: datasets/CCPD2020/ccpd_green/val/0301909722222-94_99-202&396_498&499-498&499_215
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0475-102_106-241&470_545&628-
  After [8]: datasets/CCPD2020/ccpd_green/val/0475-102_106-241&470_545&628-545&628_254&558_24
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0365386284722-95_260-197&425_
  After [9]: datasets/CCPD2020/ccpd_green/val/0365386284722-95_260-197&425_495&549-495&549_20

## manifests/curriculum_gray3_stageE_e3_control/

### train_e3_control.csv

- total_lines=74575, abs=74575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_e3_control.csv

- total_lines=10645, abs=10645 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

## manifests/curriculum_gray3_stageE_e3_main/

### train_e3_main.csv

- total_lines=74575, abs=74575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_e3_main.csv

- total_lines=10645, abs=10645 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

## manifests/curriculum_gray3_stageE_v1_extreme/

### train_E1.csv

- total_lines=93175, abs=93175 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_E1.csv

- total_lines=10645, abs=10645 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

## manifests/curriculum_gray3_stageE_v2_balanced/

### train_E2.csv

- total_lines=74575, abs=74575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_E2.csv

- total_lines=10645, abs=10645 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

## manifests/curriculum_gray3_stagea_redesign/

### val.csv

- total_lines=10190, abs=10190 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0240.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/51_0240.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000065152.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000065152.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000457043.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000457043.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0125718390805-88_93-405&624_585&70
  After [3]: datasets/CCPD2019/ccpd_base/0125718390805-88_93-405&624_585&700-588&682_416&698_
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-53&71_999&338
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462742.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462742.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000381097.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000381097.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000001868.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000001868.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0426714409722-92_262-148&445_
  After [8]: datasets/CCPD2020/ccpd_green/val/0426714409722-92_262-148&445_519&561-512&561_15
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000246371.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000246371.jpg

### train_stageA.csv

- total_lines=81855, abs=81855 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000467938.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000467938.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/47_0052.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/47_0052.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/54_0434.jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/54_0434.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000175031.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000175031.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000350299.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000350299.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0239224137931-105_64-361&473_558&5
  After [5]: datasets/CCPD2019/ccpd_base/0239224137931-105_64-361&473_558&594-550&597_372&536
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p08_u5409/
  After [6]: datasets/green_exact_quad_synthetic_v1/images/train/p08_u5409/genx-0-59&72_1017&
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000162488.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000162488.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000259295.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000259295.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000012228.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000012228.jpg

### proxy_stageA_blue_simple.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165768678161-90_88-241&500_455&57
  After [0]: datasets/CCPD2019/ccpd_base/0165768678161-90_88-241&500_455&576-455&581_246&568_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0178520114943-86_92-202&492_435&57
  After [1]: datasets/CCPD2019/ccpd_base/0178520114943-86_92-202&492_435&571-427&557_205&576_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0321264367816-96_77-167&447_436&57
  After [2]: datasets/CCPD2019/ccpd_base/0321264367816-96_77-167&447_436&571-444&575_179&531_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0325957854407-90_82-113&533_463&64
  After [3]: datasets/CCPD2019/ccpd_base/0325957854407-90_82-113&533_463&646-464&648_125&638_
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0403220785441-86_92-249&381_554&52
  After [4]: datasets/CCPD2019/ccpd_base/0403220785441-86_92-249&381_554&527-560&480_261&524_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00806034482759-91_93-321&407_495&4
  After [5]: datasets/CCPD2019/ccpd_base/00806034482759-91_93-321&407_495&459-497&468_323&465
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0226053639847-90_83-287&553_530&65
  After [6]: datasets/CCPD2019/ccpd_base/0226053639847-90_83-287&553_530&650-535&648_296&634_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0152490421455-90_92-203&499_423&58
  After [7]: datasets/CCPD2019/ccpd_base/0152490421455-90_92-203&499_423&584-428&572_219&578_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0264583333333-93_87-294&439_570&54
  After [8]: datasets/CCPD2019/ccpd_base/0264583333333-93_87-294&439_570&542-574&546_303&526_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0160739942529-98_75-249&448_449&52
  After [9]: datasets/CCPD2019/ccpd_base/0160739942529-98_75-249&448_449&528-433&549_244&510_

### proxy_stageA_mixed_foundation.csv

- total_lines=3000, abs=3000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0163888888889-91_85-251&433_544&50
  After [0]: datasets/CCPD2019/ccpd_base/0163888888889-91_85-251&433_544&506-541&529_256&511_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0378591954023-101_65-120&443_380&6
  After [1]: datasets/CCPD2019/ccpd_base/0378591954023-101_65-120&443_380&600-392&603_129&539
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/50_0949.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/50_0949.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3027126736111111112-95_104-24
  After [3]: datasets/CCPD2020/ccpd_green/val/3027126736111111112-95_104-241&439_491&549-491&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0447318007662-90_87-100&485_633&57
  After [4]: datasets/CCPD2019/ccpd_base/0447318007662-90_87-100&485_633&577-630&582_117&587_
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000080528.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000080528.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0236979166667-86_267-216&355_
  After [6]: datasets/CCPD2020/ccpd_green/val/0236979166667-86_267-216&355_476&447-476&424_22
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000408781.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000408781.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000487073.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000487073.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0330.jpg
  After [9]: datasets/CRPD_all/CRPD_single/val/images/64_0330.jpg

### proxy_stageA_green_simple.csv

- total_lines=1000, abs=1000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/303899088541666667-92_261-189
  After [0]: datasets/CCPD2020/ccpd_green/val/303899088541666667-92_261-189&400_507&524-507&5
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0200390625-95_108-19&388_247&
  After [1]: datasets/CCPD2020/ccpd_green/val/0200390625-95_108-19&388_247&477-247&477_35&454
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/302935221354166667-106_75-300
  After [2]: datasets/CCPD2020/ccpd_green/val/302935221354166667-106_75-300&511_513&650-512&6
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3022059461805555555-95_251-36
  After [3]: datasets/CCPD2020/ccpd_green/val/3022059461805555555-95_251-360&515_574&619-574&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/033125-87_271-173&457_461&573
  After [4]: datasets/CCPD2020/ccpd_green/val/033125-87_271-173&457_461&573-461&541_176&573_1
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0230501302083-98_275-263&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0230501302083-98_275-263&406_482&512-478&512_26
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0232421875-92_259-224&451_479
  After [6]: datasets/CCPD2020/ccpd_green/val/0232421875-92_259-224&451_479&543-479&543_234&5
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0502853732639-96_245-140&442_
  After [7]: datasets/CCPD2020/ccpd_green/val/0502853732639-96_245-140&442_523&574-523&574_16
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3015421006944444444-97_105-29
  After [8]: datasets/CCPD2020/ccpd_green/val/3015421006944444444-97_105-298&506_485&589-485&
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0353298611111-87_271-248&451_
  After [9]: datasets/CCPD2020/ccpd_green/val/0353298611111-87_271-248&451_544&572-544&542_25

## manifests/curriculum_gray3_stagea_v2_foundation/

### train_stageA_v2.csv

- total_lines=51142, abs=51142 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0136338601533-89_95-277&469_492&55
  After [0]: datasets/CCPD2019/ccpd_base/0136338601533-89_95-277&469_492&550-506&555_268&565_
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/train/images/49_0779.jpg
  After [1]: datasets/CRPD_all/CRPD_double/train/images/49_0779.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000176436.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000176436.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000256357.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000256357.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/train/simple/p26_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/train/simple/p26_u9655/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0150454980843-92_79-208&498_440&58
  After [5]: datasets/CCPD2019/ccpd_base/0150454980843-92_79-208&498_440&580-457&597_211&577_
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000290996.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000290996.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000384994.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000384994.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/57_0973.jpg
  After [8]: datasets/CRPD_all/CRPD_single/train/images/57_0973.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000073857.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000073857.jpg

### val_stageA_v2.csv

- total_lines=10085, abs=10085 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p04_u5180/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p04_u5180/genx-0-51&72_981&327
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0170438218391-93_78-228&670_443&75
  After [1]: datasets/CCPD2019/ccpd_base/0170438218391-93_78-228&670_443&752-439&760_247&737_
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000073214.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000073214.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000380652.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000380652.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p18_u6e
  After [4]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p18_u6e58/edgefit-tier3-1
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000116772.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000116772.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p00_u4eac/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p00_u4eac/genx-0-49&74_1001&32
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-49&68_988&327
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0948.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/38_0948.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_1084.jpg
  After [9]: datasets/CRPD_all/CRPD_single/val/images/37_1084.jpg

### proxy_blue_real_foundation.csv

- total_lines=1400, abs=1400 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0103.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_0103.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0715.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0715.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0247916666667-95_82-185&313_443&41
  After [2]: datasets/CCPD2019/ccpd_base/0247916666667-95_82-185&313_443&416-448&421_198&387_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0623.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/38_0623.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0498886494253-86_91-200&421_547&56
  After [4]: datasets/CCPD2019/ccpd_base/0498886494253-86_91-200&421_547&564-552&540_205&573_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0185177203065-92_80-269&489_496&58
  After [5]: datasets/CCPD2019/ccpd_base/0185177203065-92_80-269&489_496&585-496&581_283&569_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/022658045977-91_84-317&486_590&584
  After [6]: datasets/CCPD2019/ccpd_base/022658045977-91_84-317&486_590&584-606&588_318&578_3
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0177011494253-90_91-246&525_494&59
  After [7]: datasets/CCPD2019/ccpd_base/0177011494253-90_91-246&525_494&593-485&614_250&594_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0104418103448-91_85-276&501_461&56
  After [8]: datasets/CCPD2019/ccpd_base/0104418103448-91_85-276&501_461&569-465&565_283&553_
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1157.jpg
  After [9]: datasets/CRPD_all/CRPD_single/val/images/64_1157.jpg

### proxy_support.csv

- total_lines=1300, abs=1300 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000098848.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000098848.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000498845.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000498845.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000064222.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000064222.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000197919.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000197919.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000391493.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000391493.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000454809.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000454809.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000315525.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000315525.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000271876.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000271876.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000115134.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000115134.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000431909.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000431909.jpg

### proxy_green_bridge.csv

- total_lines=500, abs=500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p29_u5b81/
  After [0]: datasets/green_exact_quad_synthetic_v1/images/train/p29_u5b81/genx-0-49&73_985&3
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p05_u664b/
  After [1]: datasets/green_exact_quad_synthetic_v1/images/train/p05_u664b/genx-0-49&68_987&3
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p25_u85cf/
  After [2]: datasets/green_exact_quad_synthetic_v1/images/train/p25_u85cf/genx-0-76&62_1022&
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p16_u8c6b/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p16_u8c6b/genx-0-52&84_1011&
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p14_u8d63/
  After [4]: datasets/green_exact_quad_synthetic_v1/images/train/p14_u8d63/genx-0-66&78_995&3
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p04_u5180/
  After [5]: datasets/green_exact_quad_synthetic_v1/images/train/p04_u5180/genx-0-64&63_1022&
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p05_u664b/
  After [6]: datasets/green_exact_quad_synthetic_v1/images/train/p05_u664b/genx-0-51&62_985&3
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p19_u7ca4/
  After [7]: datasets/green_exact_quad_synthetic_v1/images/train/p19_u7ca4/genx-0-65&81_1035&
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p05_u664b/
  After [8]: datasets/green_exact_quad_synthetic_v1/images/train/p05_u664b/genx-0-51&71_1003&
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p29_u5b81/
  After [9]: datasets/green_exact_quad_synthetic_v1/images/train/p29_u5b81/genx-0-69&67_1031&

### proxy_green_real_foundation.csv

- total_lines=105, abs=105 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/303865017361111111-102_105-17
  After [0]: datasets/CCPD2020/ccpd_green/val/303865017361111111-102_105-176&453_450&595-450&
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0270290798611-94_98-222&434_4
  After [1]: datasets/CCPD2020/ccpd_green/val/0270290798611-94_98-222&434_487&537-487&537_222
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/30234375-97_105-176&435_416&5
  After [2]: datasets/CCPD2020/ccpd_green/val/30234375-97_105-176&435_416&534-416&534_200&503
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0166579861111-95_271-260&406_
  After [3]: datasets/CCPD2020/ccpd_green/val/0166579861111-95_271-260&406_462&489-462&489_26
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0279296875-96_249-185&456_445
  After [4]: datasets/CCPD2020/ccpd_green/val/0279296875-96_249-185&456_445&564-445&564_203&5
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0362413194444-90_258-137&456_
  After [5]: datasets/CCPD2020/ccpd_green/val/0362413194444-90_258-137&456_471&565-450&548_13
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/020902777777777777-92_254-228
  After [6]: datasets/CCPD2020/ccpd_green/val/020902777777777777-92_254-228&435_452&529-452&5
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0273285590278-94_252-244&425_
  After [7]: datasets/CCPD2020/ccpd_green/val/0273285590278-94_252-244&425_501&532-501&532_26
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307772352430555556-95_90-133&
  After [8]: datasets/CCPD2020/ccpd_green/val/307772352430555556-95_90-133&420_627&578-618&57
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0275954861111-93_96-226&464_5
  After [9]: datasets/CCPD2020/ccpd_green/val/0275954861111-93_96-226&464_515&560-515&560_235

## manifests/curriculum_gray3_stagea_v3_realprimary/

### train_A0.csv

- total_lines=62575, abs=62575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0227574233717-88_90-147&305_405&41
  After [0]: datasets/CCPD2019/ccpd_base/0227574233717-88_90-147&305_405&411-412&397_139&414_
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/44_0937.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/44_0937.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000356751.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000356751.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/train/images/62_2505.jpg
  After [3]: datasets/CRPD_all/CRPD_double/train/images/62_2505.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/train/simple/p26_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/train/simple/p26_u9655/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/40_0777.jpg
  After [5]: datasets/CRPD_all/CRPD_single/train/images/40_0777.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/train/images/54_0154.jpg
  After [6]: datasets/CRPD_all/CRPD_double/train/images/54_0154.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0282315613027-90_91-263&530_611&62
  After [7]: datasets/CCPD2019/ccpd_base/0282315613027-90_91-263&530_611&621-616&622_266&628_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266570881226-84_89-204&402_476&51
  After [8]: datasets/CCPD2019/ccpd_base/0266570881226-84_89-204&402_476&512-475&485_226&528_
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_multi/train/images/65_0822.jpg
  After [9]: datasets/CRPD_all/CRPD_multi/train/images/65_0822.jpg

### train_A1.csv

- total_lines=59575, abs=59575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/35_0652.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/35_0652.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/40_0307.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/40_0307.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/0203949652778-91_265-227&39
  After [2]: datasets/CCPD2020/ccpd_green/train/0203949652778-91_265-227&399_481&480-481&480_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/train/images/44_0535.jpg
  After [3]: datasets/CRPD_all/CRPD_double/train/images/44_0535.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000406596.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000406596.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0277370689655-90_87-215&473_531&56
  After [5]: datasets/CCPD2019/ccpd_base/0277370689655-90_87-215&473_531&568-543&566_231&587_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/42_0005.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/42_0005.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/66_0542.jpg
  After [7]: datasets/CRPD_all/CRPD_single/train/images/66_0542.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_0528.jpg
  After [8]: datasets/CRPD_all/CRPD_single/train/images/36_0528.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/47_0048.jpg
  After [9]: datasets/CRPD_all/CRPD_single/train/images/47_0048.jpg

### train_A1B.csv

- total_lines=64375, abs=64375 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/0106966145833-91_268-233&46
  After [0]: datasets/CCPD2020/ccpd_green/train/0106966145833-91_268-233&467_419&525-418&525_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/0146126302083-89_255-284&44
  After [1]: datasets/CCPD2020/ccpd_green/train/0146126302083-89_255-284&446_485&519-485&519_
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (565).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (565).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/56_0195.jpg
  After [3]: datasets/CRPD_all/CRPD_single/train/images/56_0195.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/train/simple/p00_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/train/simple/p00_u4eac/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0188362068966-87_90-326&499_551&58
  After [5]: datasets/CCPD2019/ccpd_base/0188362068966-87_90-326&499_551&589-550&567_323&587_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/39_0443.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/39_0443.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0150862068966-85_95-315&452_504&52
  After [7]: datasets/CCPD2019/ccpd_base/0150862068966-85_95-315&452_504&528-512&518_317&542_
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/train/simple/p03_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/train/simple/p03_u6e1d/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0232076149426-90_83-184&439_482&53
  After [9]: datasets/CCPD2019/ccpd_base/0232076149426-90_83-184&439_482&536-491&538_216&527_

### val_A0.csv

- total_lines=9901, abs=9901 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0562.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0562.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000293074.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000293074.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/039542624521-93_77-234&460_584&563
  After [3]: datasets/CCPD2019/ccpd_base/039542624521-93_77-234&460_584&563-594&591_225&556_2
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0187.jpg
  After [4]: datasets/CRPD_all/CRPD_double/val/images/43_0187.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_multi/val/images/54_0460.jpg
  After [5]: datasets/CRPD_all/CRPD_multi/val/images/54_0460.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0517.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0517.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/47_0831.jpg
  After [7]: datasets/CRPD_all/CRPD_double/val/images/47_0831.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0979.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/37_0979.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0290.jpg
  After [9]: datasets/CRPD_all/CRPD_single/val/images/38_0290.jpg

### val_A1.csv

- total_lines=9901, abs=9901 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0562.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0562.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000293074.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000293074.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/039542624521-93_77-234&460_584&563
  After [3]: datasets/CCPD2019/ccpd_base/039542624521-93_77-234&460_584&563-594&591_225&556_2
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0187.jpg
  After [4]: datasets/CRPD_all/CRPD_double/val/images/43_0187.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_multi/val/images/54_0460.jpg
  After [5]: datasets/CRPD_all/CRPD_multi/val/images/54_0460.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0517.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0517.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/47_0831.jpg
  After [7]: datasets/CRPD_all/CRPD_double/val/images/47_0831.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0979.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/37_0979.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0290.jpg
  After [9]: datasets/CRPD_all/CRPD_single/val/images/38_0290.jpg

### val_A1B.csv

- total_lines=9901, abs=9901 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0562.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0562.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000293074.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000293074.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/039542624521-93_77-234&460_584&563
  After [3]: datasets/CCPD2019/ccpd_base/039542624521-93_77-234&460_584&563-594&591_225&556_2
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0187.jpg
  After [4]: datasets/CRPD_all/CRPD_double/val/images/43_0187.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_multi/val/images/54_0460.jpg
  After [5]: datasets/CRPD_all/CRPD_multi/val/images/54_0460.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0517.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0517.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/47_0831.jpg
  After [7]: datasets/CRPD_all/CRPD_double/val/images/47_0831.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0979.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/37_0979.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0290.jpg
  After [9]: datasets/CRPD_all/CRPD_single/val/images/38_0290.jpg

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy/

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [0]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p16
  Before[1]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [1]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p01
  Before[2]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [2]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p13
  Before[3]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [3]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p14
  Before[4]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [4]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p28
  Before[5]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [5]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p18
  Before[6]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [6]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p18
  Before[7]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [7]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p19
  Before[8]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [8]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p25
  Before[9]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [9]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p07

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### train_B1A_C_train_v4e3_ccpdboard_eval_original.csv

- total_lines=65575, abs=65575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_B1A_C_original_eval.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u8c6b/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u95fd/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u8d63/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u9752/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u7ca4/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u85cf/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u8fbd/edgefit-tier3

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original/

### train_B1A_D_extreme900_v4e3_ccpdboard_eval_original.csv

- total_lines=66175, abs=66175 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### val_B1A_D_original_eval.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u8c6b/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u95fd/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u8d63/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u9752/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u7ca4/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u85cf/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u8fbd/edgefit-tier3

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy/

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [0]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p16
  Before[1]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [1]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p01
  Before[2]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [2]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p13
  Before[3]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [3]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p14
  Before[4]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [4]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p28
  Before[5]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [5]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p18
  Before[6]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [6]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p18
  Before[7]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [7]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p19
  Before[8]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [8]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p25
  Before[9]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [9]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p07

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/

### train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv

- total_lines=65575, abs=65575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### val_B1A_E1_original_eval.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u8c6b/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u95fd/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u8d63/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u9752/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u7ca4/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u85cf/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u8fbd/edgefit-tier3

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/pro
  After [0]: tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy/high/E1mod-167&
  Before[1]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/pro
  After [1]: tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy/high/E1mod-154&
  Before[2]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/pro
  After [2]: tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy/high/E1mod-111&
  Before[3]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/pro
  After [3]: tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy/high/E1mod-176&
  Before[4]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/pro
  After [4]: tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy/high/E1mod-125&
  Before[5]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/pro
  After [5]: tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy/high/E1mod-79&9
  Before[6]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/pro
  After [6]: tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy/low/E1mod-166&1
  Before[7]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/pro
  After [7]: tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy/high/E1mod-162&
  Before[8]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/pro
  After [8]: tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy/high/E1mod-140&
  Before[9]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/pro
  After [9]: tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy/high/E1mod-149&

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/

### train_B1A_E6A_single_axis_visible_eval_original.csv

- total_lines=65575, abs=65575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### val_B1A_E6A_original_eval.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u8c6b/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u95fd/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u8d63/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u9752/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u7ca4/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u85cf/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u8fbd/edgefit-tier3

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy/

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/im
  After [0]: tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/images/proxy/high/E6
  Before[1]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/im
  After [1]: tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/images/proxy/high/E6
  Before[2]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/im
  After [2]: tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/images/proxy/high/E6
  Before[3]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/im
  After [3]: tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/images/proxy/high/E6
  Before[4]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/im
  After [4]: tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/images/proxy/high/E6
  Before[5]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/im
  After [5]: tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/images/proxy/high/E6
  Before[6]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/im
  After [6]: tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/images/proxy/high/E6
  Before[7]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/im
  After [7]: tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/images/proxy/high/E6
  Before[8]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/im
  After [8]: tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/images/proxy/high/E6
  Before[9]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/im
  After [9]: tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427/images/proxy/high/E6

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/

### train_B1A_E6B_compound_visible_eval_original.csv

- total_lines=65575, abs=65575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### val_B1A_E6B_original_eval.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u8c6b/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u95fd/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u8d63/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u9752/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u7ca4/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u85cf/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u8fbd/edgefit-tier3

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy/

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/image
  After [0]: tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/images/proxy/high/E6Bcm
  Before[1]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/image
  After [1]: tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/images/proxy/high/E6Bcm
  Before[2]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/image
  After [2]: tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/images/proxy/high/E6Bcm
  Before[3]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/image
  After [3]: tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/images/proxy/high/E6Bcm
  Before[4]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/image
  After [4]: tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/images/proxy/high/E6Bcm
  Before[5]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/image
  After [5]: tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/images/proxy/high/E6Bcm
  Before[6]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/image
  After [6]: tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/images/proxy/low/E6Bcmp
  Before[7]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/image
  After [7]: tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/images/proxy/high/E6Bcm
  Before[8]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/image
  After [8]: tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/images/proxy/high/E6Bcm
  Before[9]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/image
  After [9]: tmp/green_extreme_stageB1A_E6B_compound_visible_20260427/images/proxy/high/E6Bcm

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original/

### train_B1A_E6_axis_dominant_perspective_eval_original.csv

- total_lines=65575, abs=65575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### val_B1A_E6_original_eval.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u8c6b/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u95fd/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u8d63/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u9752/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u7ca4/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u85cf/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u8fbd/edgefit-tier3

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy/

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_202604
  After [0]: tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427/images/proxy/hi
  Before[1]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_202604
  After [1]: tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427/images/proxy/hi
  Before[2]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_202604
  After [2]: tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427/images/proxy/hi
  Before[3]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_202604
  After [3]: tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427/images/proxy/hi
  Before[4]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_202604
  After [4]: tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427/images/proxy/hi
  Before[5]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_202604
  After [5]: tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427/images/proxy/hi
  Before[6]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_202604
  After [6]: tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427/images/proxy/lo
  Before[7]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_202604
  After [7]: tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427/images/proxy/hi
  Before[8]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_202604
  After [8]: tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427/images/proxy/hi
  Before[9]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_202604
  After [9]: tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427/images/proxy/hi

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_eval_original/

### train_B1A_E6_both_tilt_perspective_eval_original.csv

- total_lines=65575, abs=65575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### val_B1A_E6_original_eval.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u8c6b/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u95fd/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u8d63/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u9752/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u7ca4/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u85cf/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u8fbd/edgefit-tier3

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1A_E6_both_tilt_perspective_new_proxy/

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/i
  After [0]: tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/images/proxy/high/E
  Before[1]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/i
  After [1]: tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/images/proxy/high/E
  Before[2]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/i
  After [2]: tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/images/proxy/high/E
  Before[3]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/i
  After [3]: tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/images/proxy/high/E
  Before[4]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/i
  After [4]: tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/images/proxy/high/E
  Before[5]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/i
  After [5]: tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/images/proxy/high/E
  Before[6]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/i
  After [6]: tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/images/proxy/low/E6
  Before[7]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/i
  After [7]: tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/images/proxy/high/E
  Before[8]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/i
  After [8]: tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/images/proxy/high/E
  Before[9]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/i
  After [9]: tmp/green_extreme_stageB1A_E6_both_tilt_perspective_20260427/images/proxy/high/E

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/

### train_B1B_E6AB_preblur_v3_combined.csv

- total_lines=65575, abs=65575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_B1B_E6AB_combined.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u8c6b/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u95fd/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u8d63/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u9752/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u7ca4/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u85cf/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u8fbd/edgefit-tier3

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/prox
  After [0]: tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/proxy/high/E6Aaxis-302
  Before[1]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/prox
  After [1]: tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/proxy/high/E6Aaxis-438
  Before[2]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/prox
  After [2]: tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/proxy/high/E6Aaxis-390
  Before[3]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/prox
  After [3]: tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/proxy/high/E6Aaxis-308
  Before[4]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/prox
  After [4]: tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/proxy/high/E6Aaxis-415
  Before[5]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/prox
  After [5]: tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/proxy/high/E6Aaxis-318
  Before[6]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/prox
  After [6]: tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/proxy/high/E6Aaxis-280
  Before[7]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/prox
  After [7]: tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/proxy/high/E6Aaxis-284
  Before[8]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/prox
  After [8]: tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/proxy/high/E6Aaxis-405
  Before[9]: /home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/prox
  After [9]: tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/images/proxy/high/E6Aaxis-326

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_obbquad/

### train_B2C_paradigm3_obbquad.csv

- total_lines=67975, abs=67975 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_B2C_paradigm3_obbquad.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

## manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_softfreeze/

### train_B2C_paradigm3_softfreeze.csv

- total_lines=67975, abs=67975 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_B2C_paradigm3_softfreeze.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

## manifests/curriculum_gray3_stageb_v1_B2D_paradigm3_progress/

### train_B2D_paradigm3_progress.csv

- total_lines=70675, abs=70675 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_B2D_paradigm3_progress.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

## manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/

### train_pose_quad.csv

- total_lines=68065, abs=68065 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_pose_quad.csv

- total_lines=10645, abs=10645 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### train_replace_extreme_as_test.csv

- total_lines=2790, abs=2790 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/0241276041667-1
  After [0]: datasets/ccpd2020_replace_pose_v3/images/train/0241276041667-100_256-258&418_462
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/039128352490421
  After [1]: datasets/ccpd2020_replace_pose_v3/images/train/03912835249042146-90_262-202&523_
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/051494252873563
  After [2]: datasets/ccpd2020_replace_pose_v3/images/train/05149425287356322-115_231-300&403
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/033845785440613
  After [3]: datasets/ccpd2020_replace_pose_v3/images/train/03384578544061303-90_230-166&478_
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/0362782118056-7
  After [4]: datasets/ccpd2020_replace_pose_v3/images/train/0362782118056-71_127-318&462_547&
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/036027298850574
  After [5]: datasets/ccpd2020_replace_pose_v3/images/train/036027298850574714-90_263-200&513
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/050056273946360
  After [6]: datasets/ccpd2020_replace_pose_v3/images/train/050056273946360155-93_225-132&490
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/060798611111111
  After [7]: datasets/ccpd2020_replace_pose_v3/images/train/06079861111111111-91_242-112&513_
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/055050287356321
  After [8]: datasets/ccpd2020_replace_pose_v3/images/train/055050287356321836-122_300-190&40
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/040330459770114
  After [9]: datasets/ccpd2020_replace_pose_v3/images/train/04033045977011494-91_226-137&515_

### train_replace_extreme.csv

- total_lines=2790, abs=2790 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/0241276041667-1
  After [0]: datasets/ccpd2020_replace_pose_v3/images/train/0241276041667-100_256-258&418_462
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/039128352490421
  After [1]: datasets/ccpd2020_replace_pose_v3/images/train/03912835249042146-90_262-202&523_
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/051494252873563
  After [2]: datasets/ccpd2020_replace_pose_v3/images/train/05149425287356322-115_231-300&403
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/033845785440613
  After [3]: datasets/ccpd2020_replace_pose_v3/images/train/03384578544061303-90_230-166&478_
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/0362782118056-7
  After [4]: datasets/ccpd2020_replace_pose_v3/images/train/0362782118056-71_127-318&462_547&
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/036027298850574
  After [5]: datasets/ccpd2020_replace_pose_v3/images/train/036027298850574714-90_263-200&513
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/050056273946360
  After [6]: datasets/ccpd2020_replace_pose_v3/images/train/050056273946360155-93_225-132&490
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/060798611111111
  After [7]: datasets/ccpd2020_replace_pose_v3/images/train/06079861111111111-91_242-112&513_
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/055050287356321
  After [8]: datasets/ccpd2020_replace_pose_v3/images/train/055050287356321836-122_300-190&40
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/040330459770114
  After [9]: datasets/ccpd2020_replace_pose_v3/images/train/04033045977011494-91_226-137&515_

### val_replace_extreme.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/03625718390804598
  After [0]: datasets/ccpd2020_replace_pose_v3/images/val/03625718390804598-84_225-147&494_45
  Before[1]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/04622844827586207
  After [1]: datasets/ccpd2020_replace_pose_v3/images/val/04622844827586207-90_228-159&467_58
  Before[2]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/01020833333333333
  After [2]: datasets/ccpd2020_replace_pose_v3/images/val/010208333333333333-87_251-336&496_4
  Before[3]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/03235991379310345
  After [3]: datasets/ccpd2020_replace_pose_v3/images/val/03235991379310345-90_245-191&523_54
  Before[4]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/04471384099616858
  After [4]: datasets/ccpd2020_replace_pose_v3/images/val/04471384099616858-94_226-221&494_60
  Before[5]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/03625718390804598
  After [5]: datasets/ccpd2020_replace_pose_v3/images/val/03625718390804598-84_225-147&494_45
  Before[6]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/04166666666666666
  After [6]: datasets/ccpd2020_replace_pose_v3/images/val/041666666666666664-90_263-127&511_4
  Before[7]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/02487428160919540
  After [7]: datasets/ccpd2020_replace_pose_v3/images/val/024874281609195404-88_229-207&489_4
  Before[8]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/05005627394636015
  After [8]: datasets/ccpd2020_replace_pose_v3/images/val/050056273946360155-93_225-132&490_5
  Before[9]: /home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/val/10128591954022989
  After [9]: datasets/ccpd2020_replace_pose_v3/images/val/10128591954022989-94_219-48&425_661

## manifests/curriculum_gray3_stageb_v1_difficulty/

### train_B1A.csv

- total_lines=65575, abs=65575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_B1A.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u8c6b/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u95fd/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u8d63/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u9752/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u7ca4/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u85cf/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u8fbd/edgefit-tier3

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_difficulty_extreme_ccpdboard_v4e3/

### train_B1A_extreme_ccpdboard_v4e3.csv

- total_lines=65575, abs=65575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_B1A_extreme_ccpdboard_v4e3.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [0]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p16
  Before[1]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [1]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p01
  Before[2]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [2]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p13
  Before[3]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [3]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p14
  Before[4]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [4]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p28
  Before[5]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [5]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p18
  Before[6]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [6]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p18
  Before[7]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [7]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p19
  Before[8]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [8]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p25
  Before[9]: /home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/boar
  After [9]: tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail/p07

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original/

### proxy_blue_ccpd2019_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&65
  After [0]: datasets/CCPD2019/ccpd_base/0127478448276-88_92-287&582_478&652-469&632_287&654_
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&59
  After [1]: datasets/CCPD2019/ccpd_base/0240517241379-90_90-223&503_482&598-485&593_223&598_
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&59
  After [2]: datasets/CCPD2019/ccpd_base/0140086206897-90_87-239&518_456&597-450&603_236&599_
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&5
  After [3]: datasets/CCPD2019/ccpd_base/0207435344828-103_78-315&438_512&557-520&563_317&510
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549
  After [4]: datasets/CCPD2019/ccpd_base/020474137931-94_80-220&450_469&549-476&551_233&526_2
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&45
  After [5]: datasets/CCPD2019/ccpd_base/0157986111111-90_90-260&367_482&453-486&452_258&445_
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&60
  After [6]: datasets/CCPD2019/ccpd_base/0456477490421-85_94-286&471_625&605-620&572_298&609_
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570
  After [7]: datasets/CCPD2019/ccpd_base/026400862069-88_96-298&457_540&570-528&550_315&569_3
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&60
  After [8]: datasets/CCPD2019/ccpd_base/0158045977012-90_90-247&526_481&609-488&598_250&602_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&56
  After [9]: datasets/CCPD2019/ccpd_base/0165193965517-91_82-224&490_460&568-446&566_232&558_

### proxy_blue_crpd_real.csv

- total_lines=2000, abs=2000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  After [0]: datasets/CRPD_all/CRPD_single/val/images/64_1081.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  After [1]: datasets/CRPD_all/CRPD_single/val/images/37_0918.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  After [2]: datasets/CRPD_all/CRPD_single/val/images/64_0778.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  After [3]: datasets/CRPD_all/CRPD_double/val/images/50_0909.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0401.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  After [5]: datasets/CRPD_all/CRPD_double/val/images/43_0139.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  After [6]: datasets/CRPD_all/CRPD_single/val/images/38_0515.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  After [7]: datasets/CRPD_all/CRPD_single/val/images/64_0339.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  After [8]: datasets/CRPD_all/CRPD_single/val/images/51_0282.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg
  After [9]: datasets/CRPD_all/CRPD_double/val/images/55_0343.jpg

### proxy_green_ccpd2020_real.csv

- total_lines=1001, abs=1001 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_
  After [0]: datasets/CCPD2020/ccpd_green/val/0250162760417-92_267-151&426_416&522-416&522_15
  Before[1]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-21
  After [1]: datasets/CCPD2020/ccpd_green/val/305987413194444444-103_245-217&422_527&617-527&
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-15
  After [2]: datasets/CCPD2020/ccpd_green/val/3051605902777777775-87_101-155&404_565&531-553&
  Before[3]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-14
  After [3]: datasets/CCPD2020/ccpd_green/val/104609917534722222-103_106-147&411_440&570-440&
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_
  After [4]: datasets/CCPD2020/ccpd_green/val/0206597222222-92_260-230&423_468&511-468&511_23
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_
  After [5]: datasets/CCPD2020/ccpd_green/val/0250802951389-94_271-239&406_493&505-491&505_23
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_5
  After [6]: datasets/CCPD2020/ccpd_green/val/0658745659722-92_89-107&442_574&584-573&584_123
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_46
  After [7]: datasets/CCPD2020/ccpd_green/val/03009765625-81_253-255&427_462&574-454&516_255&
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_
  After [8]: datasets/CCPD2020/ccpd_green/val/0247287326389-89_261-197&438_462&532-449&514_19
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_
  After [9]: datasets/CCPD2020/ccpd_green/val/0267230902778-88_267-235&413_497&516-493&490_23

### proxy_green_nonanhui_template_synth.csv

- total_lines=1500, abs=1500 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-53&68_975&326
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000400213.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000462440.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000300388.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000491347.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000103054.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d
  After [6]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p23_u8d35/edgefit-tier3-1
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p21_u743c/genx-0-81&87_1017&47
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000451303.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u84
  After [9]: datasets/green_edgefit_tier3_full_v2/images/val/simple/p06_u8499/edgefit-tier3-1

### proxy_support_cblprd.csv

- total_lines=1200, abs=1200 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  After [0]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478662.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  After [1]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000049629.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  After [2]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000478880.jpg
  Before[3]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  After [3]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000067510.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  After [4]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000354698.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000092466.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  After [6]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000296766.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  After [7]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000414013.jpg
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000126810.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg
  After [9]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000245155.jpg

### train_B1A_train_v4e3_ccpdboard_eval_original.csv

- total_lines=65575, abs=65575 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  After [0]: datasets/CRPD_all/CRPD_single/train/images/65_0498.jpg
  Before[1]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  After [1]: datasets/CRPD_all/CRPD_single/train/images/36_1059.jpg
  Before[2]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  After [2]: datasets/CRPD_all/CRPD_single/train/images/1 (44).jpg
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/
  After [3]: datasets/green_exact_quad_synthetic_v1/images/train/p01_u6caa/genx-0-53&88_995&4
  Before[4]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&54
  After [4]: datasets/CCPD2019/ccpd_base/0182579022988-93_77-251&458_482&542-480&549_263&530_
  Before[5]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&56
  After [5]: datasets/CCPD2019/ccpd_base/0221336206897-91_81-203&456_443&563-456&561_205&546_
  Before[6]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  After [6]: datasets/CRPD_all/CRPD_single/train/images/53_0516.jpg
  Before[7]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&57
  After [7]: datasets/CCPD2019/ccpd_base/0199928160919-90_91-238&475_523&571-522&565_234&574_
  Before[8]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&65
  After [8]: datasets/CCPD2019/ccpd_base/0289583333333-93_78-196&552_491&653-495&666_204&638_
  Before[9]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503
  After [9]: datasets/CCPD2020/ccpd_green/train/01953125-91_263-253&464_503&542-503&542_259&5

### val_B1A_original_eval.csv

- total_lines=10335, abs=10335 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&5
  After [0]: datasets/CCPD2019/ccpd_base/00886015325671-90_92-206&535_363&593-367&596_206&595
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p07_u8fbd/genx-0-51&65_1007&32
  Before[2]: /home/wzzz/LPRNet/datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&54
  After [2]: datasets/CCPD2019/ccpd_base/0266642720307-90_89-252&449_527&545-535&536_261&547_
  Before[3]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  After [3]: datasets/CRPD_all/CRPD_single/val/images/48_1176.jpg
  Before[4]: /home/wzzz/LPRNet/datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  After [4]: datasets/CRPD_all/CRPD_single/val/images/64_0653.jpg
  Before[5]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  After [5]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000270520.jpg
  Before[6]: /home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115
  After [6]: datasets/CCPD2020/ccpd_green/val/307942708333333333-86_262-115&456_603&620-597&5
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4ea
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p00_u4eac/edgefit-tier3-21
  Before[8]: /home/wzzz/LPRNet/datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  After [8]: datasets/CBLPRD-330k_v1/CBLPRD-330k/000273307.jpg
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p08_u5409/genx-0-51&64_1000&32

### proxy_green_edgefit_extreme.csv

- total_lines=124, abs=124 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p16_u8c6b/edgefit-tier3
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p01_u6caa/edgefit-tier3
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p13_u95fd/edgefit-tier3
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p14_u8d63/edgefit-tier3
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p28_u9752/edgefit-tier3
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p18_u6e58/edgefit-tier3
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p19_u7ca4/edgefit-tier3
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p25_u85cf/edgefit-tier3
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/extreme/p07_u8fbd/edgefit-tier3

### proxy_green_edgefit_hard.csv

- total_lines=310, abs=310 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca
  After [0]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p19_u7ca4/edgefit-tier3-12
  Before[1]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u849
  After [1]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p06_u8499/edgefit-tier3-11
  Before[2]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c8
  After [2]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p15_u9c81/edgefit-tier3-13
  Before[3]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u518
  After [3]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p04_u5180/edgefit-tier3-13
  Before[4]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d6
  After [4]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p14_u8d63/edgefit-tier3-21
  Before[5]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95f
  After [5]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p13_u95fd/edgefit-tier3-14
  Before[6]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [6]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-15
  Before[7]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1
  After [7]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p03_u6e1d/edgefit-tier3-7&
  Before[8]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u910
  After [8]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p17_u9102/edgefit-tier3-14
  Before[9]: /home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fb
  After [9]: datasets/green_edgefit_tier3_full_v2/images/test/hard/p07_u8fbd/edgefit-tier3-9&

### proxy_green_bridge_exactquad.csv

- total_lines=800, abs=800 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [0]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-48&70_1007&32
  Before[1]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/ge
  After [1]: datasets/green_exact_quad_synthetic_v1/images/val/p09_u9ed1/genx-0-75&66_1025&45
  Before[2]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/ge
  After [2]: datasets/green_exact_quad_synthetic_v1/images/val/p18_u6e58/genx-0-89&89_1018&43
  Before[3]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [3]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-51&76_996&392
  Before[4]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/ge
  After [4]: datasets/green_exact_quad_synthetic_v1/images/val/p29_u5b81/genx-0-71&68_1014&45
  Before[5]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/ge
  After [5]: datasets/green_exact_quad_synthetic_v1/images/val/p24_u4e91/genx-0-52&79_986&327
  Before[6]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [6]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-59&92_1024&35
  Before[7]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [7]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-49&81_992&456
  Before[8]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/ge
  After [8]: datasets/green_exact_quad_synthetic_v1/images/val/p06_u8499/genx-0-49&69_991&327
  Before[9]: /home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/ge
  After [9]: datasets/green_exact_quad_synthetic_v1/images/val/p11_u6d59/genx-0-48&69_977&327

## manifests/firstchar_batch1/

### D1_firstchar_manifest_green8_only_v1_train.csv

- total_lines=138850, abs=13933 rel=124917
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### D2_firstchar_manifest_green8_normal7_v1_train.csv

- total_lines=208049, abs=13933 rel=194116
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

### D3_firstchar_manifest_green8_normal7_selectedspecial_v1_train.csv

- total_lines=266702, abs=13933 rel=252769
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

## manifests/firstchar_tiny_gray_alldata_v1/

### train.csv

- total_lines=208049, abs=13933 rel=194116
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

## manifests/firstchar_tiny_gray_green8only_v1/

### train.csv

- total_lines=138850, abs=13933 rel=124917
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&52
  After [0]: CCPD2020/ccpd_green/train/00360785590278-91_265-311&485_406&524-406&524_313&520_
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548
  After [1]: CCPD2020/ccpd_green/train/00373372395833-90_96-276&514_387&548-387&548_276&547_2
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&47
  After [2]: CCPD2020/ccpd_green/train/00378472222222-90_268-291&442_400&477-396&476_292&477_
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&46
  After [3]: CCPD2020/ccpd_green/train/00395833333333-90_268-243&427_357&462-357&460_243&462_
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_3
  After [4]: CCPD2020/ccpd_green/train/0040581597222222225-91_260-248&491_358&528-358&527_253
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-4
  After [5]: CCPD2020/ccpd_green/train/00408203125-84_269-303&357_402&399-402&388_306&399_303
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&49
  After [6]: CCPD2020/ccpd_green/train/00425347222222-89_267-242&459_354&497-354&497_244&497_
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478
  After [7]: CCPD2020/ccpd_green/train/00435329861111-90_90-320&441_438&478-437&478_320&472_3
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&52
  After [8]: CCPD2020/ccpd_green/train/004375-93_265-261&490_373&529-369&529_261&524_265&490_
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&53
  After [9]: CCPD2020/ccpd_green/train/00439019097222-90_264-255&501_374&538-370&538_255&536_

## manifests/province_degrade_train_v1/

### train_province_degrade_v1.csv

- total_lines=10000, abs=10000 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/province_degrade_train_v1/images/train/03356800766283
  After [0]: datasets/province_degrade_train_v1/images/train/03356800766283525-90_265-98&516_
  Before[1]: /home/wzzz/LPRNet/datasets/province_degrade_train_v1/images/train/03602729885057
  After [1]: datasets/province_degrade_train_v1/images/train/036027298850574714-90_263-200&51
  Before[2]: /home/wzzz/LPRNet/datasets/province_degrade_train_v1/images/train/00298850574712
  After [2]: datasets/province_degrade_train_v1/images/train/002988505747126437-88_90-149&574
  Before[3]: /home/wzzz/LPRNet/datasets/province_degrade_train_v1/images/train/00815134099616
  After [3]: datasets/province_degrade_train_v1/images/train/008151340996168582-90_268-272&49
  Before[4]: /home/wzzz/LPRNet/datasets/province_degrade_train_v1/images/train/01023706896551
  After [4]: datasets/province_degrade_train_v1/images/train/010237068965517241-89_267-258&51
  Before[5]: /home/wzzz/LPRNet/datasets/province_degrade_train_v1/images/train/04599137931034
  After [5]: datasets/province_degrade_train_v1/images/train/045991379310344825-91_245-119&46
  Before[6]: /home/wzzz/LPRNet/datasets/province_degrade_train_v1/images/train/05300526819923
  After [6]: datasets/province_degrade_train_v1/images/train/053005268199233714-91_258-123&50
  Before[7]: /home/wzzz/LPRNet/datasets/province_degrade_train_v1/images/train/00366379310344
  After [7]: datasets/province_degrade_train_v1/images/train/003663793103448276-90_264-304&48
  Before[8]: /home/wzzz/LPRNet/datasets/province_degrade_train_v1/images/train/00718390804597
  After [8]: datasets/province_degrade_train_v1/images/train/007183908045977011-88_260-236&53
  Before[9]: /home/wzzz/LPRNet/datasets/province_degrade_train_v1/images/train/03602729885057
  After [9]: datasets/province_degrade_train_v1/images/train/036027298850574714-90_263-200&51

## manifests/province_stress_pose_val_v1/

### province_stress_pose_val_v1.csv

- total_lines=1240, abs=1240 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/datasets/province_stress_pose_val_v1/images/test/0360272988505
  After [0]: datasets/province_stress_pose_val_v1/images/test/036027298850574714-90_263-200&5
  Before[1]: /home/wzzz/LPRNet/datasets/province_stress_pose_val_v1/images/test/0422413793103
  After [1]: datasets/province_stress_pose_val_v1/images/test/04224137931034483-92_237-136&50
  Before[2]: /home/wzzz/LPRNet/datasets/province_stress_pose_val_v1/images/test/0360272988505
  After [2]: datasets/province_stress_pose_val_v1/images/test/036027298850574714-90_263-200&5
  Before[3]: /home/wzzz/LPRNet/datasets/province_stress_pose_val_v1/images/test/0422413793103
  After [3]: datasets/province_stress_pose_val_v1/images/test/04224137931034483-92_237-136&50
  Before[4]: /home/wzzz/LPRNet/datasets/province_stress_pose_val_v1/images/test/0360272988505
  After [4]: datasets/province_stress_pose_val_v1/images/test/036027298850574714-90_263-200&5
  Before[5]: /home/wzzz/LPRNet/datasets/province_stress_pose_val_v1/images/test/0422413793103
  After [5]: datasets/province_stress_pose_val_v1/images/test/04224137931034483-92_237-136&50
  Before[6]: /home/wzzz/LPRNet/datasets/province_stress_pose_val_v1/images/test/0360272988505
  After [6]: datasets/province_stress_pose_val_v1/images/test/036027298850574714-90_263-200&5
  Before[7]: /home/wzzz/LPRNet/datasets/province_stress_pose_val_v1/images/test/0422413793103
  After [7]: datasets/province_stress_pose_val_v1/images/test/04224137931034483-92_237-136&50
  Before[8]: /home/wzzz/LPRNet/datasets/province_stress_pose_val_v1/images/test/0360272988505
  After [8]: datasets/province_stress_pose_val_v1/images/test/036027298850574714-90_263-200&5
  Before[9]: /home/wzzz/LPRNet/datasets/province_stress_pose_val_v1/images/test/0422413793103
  After [9]: datasets/province_stress_pose_val_v1/images/test/04224137931034483-92_237-136&50

## manifests/subsets_e3_analysis_20260412/

### green8_test_all.csv

- total_lines=6082, abs=6082 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

### green8_test_real_only.csv

- total_lines=5006, abs=5006 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/test/0014128352490421455-90_90-212&467_271
  After [0]: CCPD2020/ccpd_green/test/0014128352490421455-90_90-212&467_271&489-271&489_212&4
  Before[1]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/test/0016594827586206896-90_90-341&550_407
  After [1]: CCPD2020/ccpd_green/test/0016594827586206896-90_90-341&550_407&573-407&573_341&5
  Before[2]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/test/0017959770114942528-89_265-240&542_31
  After [2]: CCPD2020/ccpd_green/test/0017959770114942528-89_265-240&542_315&564-313&563_240&
  Before[3]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/test/0021264367816091955-90_267-311&542_38
  After [3]: CCPD2020/ccpd_green/test/0021264367816091955-90_267-311&542_385&569-385&567_312&
  Before[4]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/test/0021264367816091955-92_264-333&534_40
  After [4]: CCPD2020/ccpd_green/test/0021264367816091955-92_264-333&534_407&560-405&560_334&
  Before[5]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/test/0021479885057471265-90_262-315&502_39
  After [5]: CCPD2020/ccpd_green/test/0021479885057471265-90_262-315&502_393&527-393&527_318&
  Before[6]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/test/002454501915708812-91_264-310&567_392
  After [6]: CCPD2020/ccpd_green/test/002454501915708812-91_264-310&567_392&595-391&594_310&5
  Before[7]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/test/0026041666666666665-91_265-360&505_43
  After [7]: CCPD2020/ccpd_green/test/0026041666666666665-91_265-360&505_435&537-435&537_362&
  Before[8]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/test/0026484674329501916-93_90-283&513_362
  After [8]: CCPD2020/ccpd_green/test/0026484674329501916-93_90-283&513_362&543-360&543_283&5
  Before[9]: /home/wzzz/LPRNet/CCPD2020/ccpd_green/test/00272389846743295-90_90-371&537_436&5
  After [9]: CCPD2020/ccpd_green/test/00272389846743295-90_90-371&537_436&575-436&571_371&575

### green8_test_synth_only.csv

- total_lines=1076, abs=1076 rel=0
- converted_paths_in_sample=10
  Before[0]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&
  After [0]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-48&69_985&327-48&69_9
  Before[1]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [1]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&58_1006&327-49&72_
  Before[2]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&
  After [2]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-49&72_994&327-49&72_9
  Before[3]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [3]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&61_1004&322-51&88_
  Before[4]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [4]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_994&327-52&69_9
  Before[5]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [5]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&68_995&323-89&68_9
  Before[6]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [6]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&69_1003&327-51&70_
  Before[7]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [7]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&70_985&327-65&84_9
  Before[8]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [8]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&72_998&332-68&73_9
  Before[9]: /home/wzzz/LPRNet/green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&
  After [9]: green_exact_quad_synthetic_v1/images/test/p24_u4e91/genx-0-50&74_1001&392-51&134

---

*DRY-RUN 完成，未执行任何写入操作*