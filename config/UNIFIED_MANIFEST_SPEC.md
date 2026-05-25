# 统一 Manifest 方案

## 1. 目标

在不跑偏原计划的前提下，把当前分散在多个 txt 文件里的数据，统一成一个可扩展 manifest，明确记录：

- 这条样本属于什么牌型族
- 是否是真实图 / 合成图 / 板端 dump
- 是否需要倾斜增强
- 是否具备 bbox / quad
- 能否直接走透视链
- 应该走哪一类预处理入口

这一步是 unified v1 前的数据层基建。

---

## 2. 当前结论

必须承认一个现实：

- CCPD 类数据：可从文件名解析 bbox/quad，可直接走 `ccpd_board` 透视/裁剪链
- 非 CCPD 只有车牌号标签的数据：没有 bbox/quad，不能直接复用 `obb_warp` / quad 透视流程
- 板端 dump：已经是最终 OCR 输入图，不需要再做几何裁剪

因此 manifest 必须把“能否走透视链”编码进去。

---

## 3. CSV 字段

当前定版字段如下：

- `img_path`
- `img_rel_path`
- `dataset_name`
- `split`
- `text`
- `plate_len`
- `family`
- `sub_type`
- `source`
- `is_real`
- `need_tilt_aug`
- `preprocess_group`
- `has_bbox`
- `has_quad`
- `can_parse_ccpd_geom`
- `can_perspective`
- `bbox_source`
- `quad_source`
- `ocr_channel_order`
- `ocr_crop_mode`
- `ocr_resize_mode`
- `ocr_resize_kernel`
- `ocr_preproc`
- `ocr_min_occ_ratio`
- `ocr_quad_pad_ratio`

---

## 4. 关键字段解释

### family
当前先保留三类：
- `normal7`
- `green8`
- `special`

### sub_type
当前可先用：
- `blue`
- `green`
- 后续再扩黄牌、警牌、使馆等

### preprocess_group
这是这次新增的关键字段。

只允许以下三类：

#### `ccpd_board`
用于：
- 能从 CCPD 文件名解析 bbox / quad 的样本

特点：
- 可以走 `match` / `obb_warp`
- 可以做更贴近板端的几何裁剪

#### `plain_plate`
用于：
- 只有整牌图 + 车牌文本，没有框 / 四点的样本

特点：
- 不能直接透视裁剪
- 只能按整牌图直接 resize + normalize 送识别器

#### `board_dump`
用于：
- 板端直接 dump 出来的 OCR 图

特点：
- 已经是最终输入图
- 不再做几何裁剪

### can_perspective
- `1`：可以直接走透视链
- `0`：不可以

当前规则：
- `ccpd_board` 且有 quad -> 1
- 其它 -> 0

---

## 5. 当前默认映射

### CCPD2019 / CCPD2019 hard tilt
- family = `normal7`
- sub_type = `blue`
- source = `real`
- preprocess_group = `ccpd_board`
- need_tilt_aug = `1`

### CCPD2020 green
- family = `green8`
- sub_type = `green`
- source = `real`
- preprocess_group = `ccpd_board`
- need_tilt_aug = `1`

### targeted_green_missing_18
- family = `green8`
- sub_type = `green`
- source = `synthetic_full`
- preprocess_group = `plain_plate`
- need_tilt_aug = `1`

### board dump
- family = 先按 `normal7`
- sub_type = 暂按 `blue`
- source = `board_dump`
- preprocess_group = `board_dump`
- need_tilt_aug = `0`

---

## 6. 已完成落地文件

### 生成脚本
- `/home/wzzz/LPRNet/build_unified_manifest.py`

### 数据入口支持
- `/home/wzzz/LPRNet/load_data.py`
  - 新增 `UnifiedManifestDataset`

### 测试
- `/home/wzzz/LPRNet/tests/test_unified_manifest.py`

---

## 7. 下一步用途

manifest 完成后，下一步不是直接上 unified 多头结构，而是先把训练入口改成：

- 可以从 manifest 读样本
- 并根据 `preprocess_group` 选择：
  - `ccpd_board`
  - `plain_plate`
  - `board_dump`

等数据入口稳定后，再上 unified v1：
- `shared_backbone`
- `type_head`
- `normal7_head`
- `green8_head`
