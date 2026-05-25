# 非 CCPD 自动 OBB 补框状态

## 1. 目标

把原本没有 bbox / quad 的非 CCPD 车牌数据，尽可能补成带几何信息的“伪 CCPD 样本”，用于后续统一 manifest 和统一训练。

核心原则：
- 成功样本：转成带 bbox / quad 的 CCPD式文件名，并按 family 分族归档
- 失败样本：单独分流，不混进成功样本池
- 这批几何标注属于 `pseudo_geom`，不是 CCPD 真几何

---

## 2. 当前已落地产物

### 检测器
- 权重：`/home/wzzz/LPRNet/external_detectors/obb_best.pt`
- 类型：YOLOv8 OBB

### 执行脚本
- `/home/wzzz/LPRNet/auto_label_nonccpd_obb.py`

### 输出目录
- `/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1`

---

## 3. 输出结构

### 成功样本
- `nonccpd_obb_autolabel_v1/success/<family>/<dataset>/<split>/...jpg`

成功样本文件名已写入：
- bbox
- quad
- plate text
- 原始来源标识

命名示例：
- `autoobb-0-28&5_251&67-250&67_251&9_30&5_29&63-川EFC9322-0-0-targeted_green_missing_18_train_chuan_gen-train-12_8_948_272-0-0-000000.jpg`

### 失败样本
- `nonccpd_obb_autolabel_v1/failed/<family>/<dataset>/<split>/...jpg`

### 记录文件
- 成功清单：`/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1/success_records.csv`
- 失败清单：`/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1/failed_records.csv`
- 汇总：`/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1/summary.json`

---

## 4. 当前处理范围

### A. targeted_green_missing_18
- train = 9000
- val = 900
- family = `green8`

### B. git_plate/val/val_verify
- val = 2014
- family = `normal7`

总样本数：`11914`

---

## 5. 当前结果

### targeted_green_missing_18 / train / green8
- total = `9000`
- success = `7436`
- fail = `1564`

### targeted_green_missing_18 / val / green8
- total = `900`
- success = `740`
- fail = `160`

### git_plate / val / normal7
- total = `2014`
- success = `1478`
- fail = `536`

### 总计
- total = `11914`
- success = `9654`
- fail = `2260`

---

## 6. 分族标签文件

### green8
- `/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1/labels/green8/train.txt`
- `/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1/labels/green8/val.txt`

### normal7
- `/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1/labels/normal7/val.txt`

当前这批数据里未生成 `special` 标签文件。

---

## 7. 后续并入 unified manifest 的约定

这批成功样本并入 unified manifest 时，建议固定使用：

- `source = pseudo_geom`
- `is_real = 0`（几何为伪标注；若后续需要拆图像真实度与几何真实度，可再细分）
- `preprocess_group = ccpd_board`
- `has_bbox = 1`
- `has_quad = 1`
- `can_parse_ccpd_geom = 1`
- `can_perspective = 1`
- `bbox_source = detector_obb`
- `quad_source = detector_obb`

注意：
- 它们可以走几何裁剪链
- 但不能与 CCPD 真几何样本等价看待
- 训练时权重不应默认高于 CCPD 真几何数据

---

## 8. 当前结论

这次自动补框已经不是试验想法，而是已正式落地的一层数据资产：
- green8 获得一批可走几何链的伪 CCPD 样本
- normal7 获得一批可走几何链的伪 CCPD 样本
- 失败样本被单独隔离，避免污染训练

这一步可直接作为 unified manifest 扩展数据源使用。
