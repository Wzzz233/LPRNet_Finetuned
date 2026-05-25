# Manifest 路径修复规划 (Rebase Plan)

生成时间: 2026-05-07 (v2 - 修正版)
项目根目录: /home/wzzz/LPRNet

---

## 1. 总览

| 指标 | 值 |
|------|-----|
| Manifest 总数 | 440 |
| 可自动转换 (auto-rebase) | 334 |
| 可自动转换 + 需验证 | 2 |
| 需人工审查 | 40 |
| 无需处理 | 64 |
| 绝对路径总数 | 754,479 |
| 无效路径 (rebase 前) | 53,580 |
| 无效路径 (rebase 后) | 49,289 |
| 可自动恢复的路径 | 4,291 |

## 2. 转换规则

```
原始:  /home/wzzz/LPRNet/{prefix}/{subdir}/{filename}
新:    {detected_root}/{subdir}/{filename}
       (detected_root 相对于 /home/wzzz/LPRNet)
统一 dataset_root = /home/wzzz/LPRNet
```

### Canonical Root 映射表

| 原始前缀 | Canonical 根 | 类型 | Canonical 存在 | 对应 Manifest 数 |
|---------|-------------|------|---------------|----------------|
| `CCPD2019/` | `datasets/CCPD2019/` | ✅ | 338 |
| `CCPD2020/` | `datasets/CCPD2020/` | ✅ | 338 |
| `CRPD_all/` | `datasets/CRPD_all/` | ✅ | 338 |
| `CBLPRD-330k_v1/` | `datasets/CBLPRD-330k_v1/` | ✅ | 338 |
| `CRPD_raw_ccpd_board_v1/` | `datasets/CRPD_raw_ccpd_board_v1/` | ✅ | 338 |
| `green_exact_quad_synthetic_v1/` | `datasets/green_exact_quad_synthetic_v1/` | ✅ | 338 |
| `green_edgefit_v3_allprov/` | `datasets/green_edgefit_v3_allprov/ (symlink缺失)` | ✅ | 338 |
| `green_edgefit_tier3_full_v3_su_conservative/` | `datasets/green_edgefit_tier3_full_v3_su_conservative/ (symlink缺失)` | ✅ | 338 |
| `datasets/...` | `datasets/...` | ✅ | 338 |
| `tmp/...` | `tmp/...` | ✅ | 15 |

## 3. 无效路径分析

| 类型 | 数量 | 原因 | 修复后 |
|------|------|------|--------|
| 软链接缺失 (green_edgefit) | 4,291 | 根目录 softlink 缺失，数据在 datasets/ | ✅ 0 (rebase 后消失) |
| 文件缺失 (tmp/ 等) | ~49,289 | 数据不存在或路径错误 | ❌ 需人工确认 |
| 总计 | 53,580 | | |

## 4. 自动转换清单

### 4.1 自动转换 (clean, 334)

无需验证，直接执行。

- TV=180 | manifests/unified_manifest_green_e12_pose_replace_test.csv
- TV=170 | manifests/unified_manifest_green_e12_replace_pose_v3_append.csv
- TV=165 | manifests/curriculum_gray3_stageE_e3_control/train_e3_control.csv
- TV=165 | manifests/curriculum_gray3_stageE_e3_main/train_e3_main.csv
- TV=165 | manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/train_pose_qua
- TV=155 | manifests/curriculum_gray3/val.csv
- TV=155 | manifests/curriculum_gray3_stageE_e3_control/val_e3_control.csv
- TV=155 | manifests/curriculum_gray3_stageE_e3_main/val_e3_main.csv
- TV=155 | manifests/curriculum_gray3_stagea_redesign/val.csv
- TV=155 | manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/val_pose_quad.
- TV=150 | manifests/firstchar_tiny_gray_alldata_v1/train.csv
- TV=150 | manifests/firstchar_tiny_gray_green8only_v1/train.csv
- TV=150 | manifests/special_train.csv
- TV=150 | manifests/yellow_single_train_weighted.csv
- TV=150 | manifests/yellow_train.csv
- ... 还有 319 个

### 4.2 自动转换 + 需验证 (2)

| Manifest | 失效前 | 失效后 | 根因 |
|----------|--------|--------|------|
| manifests/unified_manifest_green_edgefit_v3_allprov.csv |  3331 |     0 | 旧软链接缺失，但 canonical datasets 路径存在，可通过 rebase 修复（需验证） |
| manifests/Archive/unified_manifest_green_specialist_off |   960 |     0 | 旧软链接缺失，但 canonical datasets 路径存在，可通过 rebase 修复（需验证） |

## 5. 需人工审查

| Manifest | 类型 | 根路径 | 失效 | 原因 |
|----------|------|--------|------|------|
| manifests/firstchar_tiny_gray3_fullcrop_bal31_v1/t | high | no_abs | 0 | Canonical 根路径不存在: None; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/firstchar_tiny_gray3_fullcrop_bal31_v1_s | high | no_abs | 0 | Canonical 根路径不存在: None; 混合路径类型需统一 |
| manifests/firstchar_tiny_gray3_fullcrop_bal31_v1_s | high | no_abs | 0 | Canonical 根路径不存在: None; 混合路径类型需统一 |
| manifests/curriculum_gray3_stageb_v1_B1A_D_extreme | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/unified_manifest_green_edgefit_v4_e2_202 | high | tmp | 3000 | 含 3000 个无效路径; 混合路径类型需统一 |
| manifests/unified_manifest_green_edgefit_v4_e3_equ | high | tmp | 3331 | 含 3331 个无效路径; 混合路径类型需统一 |
| manifests/unified_manifest_green_edgefit_v4_e3_equ | high | tmp | 4340 | 含 4340 个无效路径; 混合路径类型需统一 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_train_v | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_modera | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_singl | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compo | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_d | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_t | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_train_v4e3_cc | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/unified_manifest_green_e25a_stageA_targe | high | tmp | 3331 | 含 3331 个无效路径; 混合路径类型需统一 |
| manifests/unified_manifest_green_e25b_stageB_lite_ | high | tmp | 3331 | 含 3331 个无效路径; 混合路径类型需统一 |
| manifests/unified_manifest_green_edgefit_v4_e4_ext | high | tmp | 310 | 含 310 个无效路径; 混合路径类型需统一 |
| manifests/unified_manifest_v3_smoketest.csv | high | CCPD2019 | 3331 | 含 3331 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preb | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/unified_manifest_official_gray3_bluegree | high | CCPD2020 | 1666 | 含 1666 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/unified_manifest_official_gray3_bluegree | high | CCPD2020 | 1666 | 含 1666 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/unified_manifest_official_gray3_bluegree | high | CCPD2020 | 1666 | 含 1666 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/unified_manifest_v3_round1_green_conserv | high | CCPD2019 | 3331 | 含 3331 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/unified_manifest_v4_geom_audited.csv | high | CCPD2019 | 3331 | 含 3331 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/unified_manifest_v4_round1_green_conserv | high | CCPD2019 | 3331 | 含 3331 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_E1_modera | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_E6A_singl | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_E6B_compo | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_d | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1A_E6_both_t | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preb | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/curriculum_gray3_stageb_v1_difficulty_ex | high | datasets+tmp | 0 | 跨多根路径: datasets+tmp; Canonical 根路径不存在: None |
| manifests/cluster_special_validation_v1/cluster_sp | high | no_abs | 0 | Canonical 根路径不存在: None |
| manifests/normal7_test_only_v1.csv | high | CCPD2019 | 3331 | 含 3331 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/Archive/unified_manifest_green_balance_r | high | CCPD2020+green_exact | 0 | 跨多根路径: CCPD2020+green_exact_quad_synthetic_v1; Can |
| manifests/Archive/unified_manifest_v1.csv | high | CCPD2019 | 3331 | 含 3331 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/Archive/unified_manifest_v2_with_pseudo_ | high | CCPD2019 | 3331 | 含 3331 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |
| manifests/Archive/unified_manifest_v3.csv | high | CCPD2019 | 3331 | 含 3331 个无效路径; 大文件需完整扫描; 混合路径类型需统一 |

## 6. 优先修复顺序

1. 自动转换 (clean, 334 manifests) — 立即执行
2. 自动转换 + 需验证 (2 manifests) — 执行后验证
3. 跨多根路径 (datasets+tmp) — 需要决定统一到哪个根
4. 大文件 + 混合路径 — 需要完整扫描确认
5. Archive — 确认是否保留

---
*由修正版 manifest_rebase_plan.json 自动生成*

# detected_root vs training_dataset_root: 区别说明

## 定义

| 字段 | 角色 | 含义 |
|------|------|------|
| `detected_root` | Rebase 元数据 | 描述 manifest 中路径的第一级数据源目录 |
| `training_dataset_root` | 训练配置 | 应传递给训练脚本的 dataset_root 参数 |

## 为什么不同


`detected_root` 描述的是路径的来源（数据在哪个数据集目录下），而 `training_dataset_root` 是训练时
用于拼接路径的基目录。两者含义不同：

| Manifest 路径 | detected_root | training_dataset_root |
|-------------|---------------|----------------------|
| `datasets/CBLPRD-330k_v1/...` | `datasets` | `/home/wzzz/LPRNet` |
| `CCPD2020/ccpd_green/...` | `datasets/CCPD2020` | `/home/wzzz/LPRNet` |
| `green_edgefit_v3_allprov/...` | `datasets/green_edgefit_v3_allprov` | `/home/wzzz/LPRNet/datasets` |

## 为什么 training_dataset_root = PROJECT_ROOT


Rebase 去掉了 `/home/wzzz/LPRNet/` 前缀。转换后路径已经是相对于 PROJECT_ROOT 的路径。
所以 training_dataset_root 应设为 PROJECT_ROOT。


## 现有训练脚本的兼容性


当前 `train_LPRNet.py` 不支持 `--dataset_root` 参数。`UnifiedManifestDataset` 直接使用
manifest 中的 `img_path` 调用 `cv2.imread()`，路径解析依赖于 CWD。
以原始训练启动方式为例（`cd src/training && python train_LPRNet.py`）：

```
CWD = /home/wzzz/LPRNet/src/training/
img_path in manifest = "CCPD2020/ccpd_green/train/xxx.jpg"
cv2.imread("CCPD2020/ccpd_green/train/xxx.jpg")
  → 实际查找: /home/wzzz/LPRNet/src/training/CCPD2020/ccpd_green/train/xxx.jpg
  → NOT FOUND
```

**结论：当前 rebased manifest 无法直接在现有训练脚本中使用。**

## 修复方案


### 方案 A：添加 --dataset_root 参数到 train_LPRNet.py（推荐）

```python
# 在 UnifiedManifestDataset.__init__ 中添加
self.dataset_root = dataset_root  # /home/wzzz/LPRNet

# 在 __getitem__ 中
img_path = os.path.join(self.dataset_root, row['img_path'])
```

这是最干净的方案。只需修改 `src/load_data.py` 约 3 行和 `src/training/train_LPRNet.py` 约 5 行。

### 方案 B：从 PROJECT_ROOT 运行训练脚本

```bash
cd /home/wzzz/LPRNet
python src/training/train_LPRNet.py ...
```

零代码修改，但需要更新所有实验启动脚本。并且 green_edgefit 仍需要特殊处理。

### 方案 C：保持绝对路径

不推荐。这会回到旧路径炸弹的问题。

### 方案 D：创建软链接（仅对 green_edgefit）

```bash
ln -s datasets/green_edgefit_v3_allprov green_edgefit_v3_allprov
```

使 PROJECT_ROOT + green_edgefit_v3_allprov/... 可以访问。
可配合方案 B 使用。

---
Generated by manifest_rebase_plan.md v2