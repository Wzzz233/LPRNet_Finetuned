# unified manifest v2（加入 pseudo_geom）说明

## 1. 产物

### 新 manifest
- `/home/wzzz/LPRNet/manifests/unified_manifest_v2_with_pseudo_geom.csv`

### 摘要
- `/home/wzzz/LPRNet/manifests/unified_manifest_v2_with_pseudo_geom.summary.json`

---

## 2. 本次新增内容

把 `/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1/success_records.csv` 中成功补框的非 CCPD 样本，并入 unified manifest。

新增样本数：`9654`

其中：
- `targeted_green_missing_18_pseudo_geom` = `8176`
- `git_plate_pseudo_geom` = `1478`

按 family：
- `green8` = `8176`
- `normal7` = `1478`

---

## 3. 并入后的字段约定

对于这批自动补框成功样本，统一写为：

- `source = pseudo_geom`
- `is_real = 0`
- `preprocess_group = ccpd_board`
- `has_bbox = 1`
- `has_quad = 1`
- `can_parse_ccpd_geom = 1`
- `can_perspective = 1`
- `bbox_source = detector_obb`
- `quad_source = detector_obb`
- `ocr_channel_order = bgr`
- `ocr_crop_mode = obb_warp`

这表示：
- 它们已经具备几何裁剪和透视能力
- 但它们的几何信息来自检测器，不是原始真标注

---

## 4. 为什么叫 pseudo_geom，而不是直接当真几何

原因不是说它“没检测出来”，而是说：

### 4.1 几何来源不是人工/原始标注，而是模型推断
也就是说：
- 这四个角点不是数据集原生给出的
- 而是检测器根据图像内容预测出来的

只要是模型预测出来的几何，就应该和原始真标注区分开。

### 4.2 模型输出可能近似正确，但不是可证明真值
例如：
- 框大一点、小一点
- 四点顺序略偏
- 角点稍微歪一点
- 某些困难样本框得不够紧

这些误差不一定会让样本完全失效，
但足以说明它和 CCPD 文件名自带的真几何不能完全等价。

### 4.3 训练时需要给它保留“低一档可信度”的身份
把它标成 `pseudo_geom` 的目的，是为了后续训练时保留调节空间，例如：
- 是否降低采样权重
- 是否只在后期引入
- 是否只用于某些 family

如果现在直接把它伪装成 `real geom`，后面就丢失了控制能力。

所以：
- `pseudo` 不等于“假的不能用”
- 它的准确含义是“由模型推断得到，而不是真值来源”

---

## 5. 当前并入后的总体变化

### 原 manifest
- `453927` 条

### 新增 pseudo_geom
- `9654` 条

### manifest v2 总数
- `463581` 条

### preprocess_group 变化
- `ccpd_board` = `453680`
- `plain_plate` = `9900`
- `board_dump` = `1`

说明：
- 这批成功补框样本，已经把一部分原先只能走 `plain_plate` 的非 CCPD 数据，提升成了可走 `ccpd_board` 几何链的数据资产
