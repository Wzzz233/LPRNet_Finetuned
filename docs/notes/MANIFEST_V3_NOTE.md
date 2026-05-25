# unified manifest v3 说明

## 1. 新增数据源

### A. CRPD_CCPD_STRICT_YOLO_v2
来源：
- `/home/wzzz/LPRNet/CRPD_CCPD_STRICT_YOLO_v2/mapping.csv`
- `/home/wzzz/LPRNet/CRPD_CCPD_STRICT_YOLO_v2/plate_crops/...`

接入方式：
- 直接按结构化 crop 数据并入
- 使用文件名中的 bbox / quad
- 归类为 `ccpd_board`

当前统计：
- 总数：`43365`
- split：train=`31201`，val=`7800`，test=`4364`
- family：全部归到 `normal7`
- sub_type：`blue`

### B. CBLPRD-330k_v1
来源：
- `/home/wzzz/LPRNet/CBLPRD-330k_v1/train.txt`
- `/home/wzzz/LPRNet/CBLPRD-330k_v1/val.txt`
- 图片目录：`/home/wzzz/LPRNet/CBLPRD-330k_v1/CBLPRD-330k`

说明：
- 这次不直接生吃全部样本
- 先过滤当前字符表不支持的车牌文本
- 再对剩余样本用 `obb_best.pt` 做 OBB 补框
- 成功样本并入 `pseudo_geom`
- 失败样本按 `plain_plate` 并入

字符表不支持而被跳过的尾字符主要有：
- `学`
- `挂`
- `使`
- `领`
- `临`
- `澳`
- `港`

因此当前 manifest v3 还没有把这些字符类型并进去。

---

## 2. CBLPRD 当前接入结果

### 输入支持情况
- train + val 总计：`342110`
- 当前字符表支持：`296109`
- 因字符不支持/缺图等跳过：`46001`

### OBB 补框结果（仅对支持字符集样本）
- 成功：`90501`
- 失败：`205608`

### pseudo_geom 部分
数据集名：`cblprd_pseudo_geom`
- family：
  - `normal7` = `34604`
  - `green8` = `29518`
  - `special` = `26379`

### plain_plate 部分
数据集名：`cblprd_plain`
- family：
  - `normal7` = `44356`
  - `green8` = `102057`
  - `special` = `59195`

说明：
- 蓝牌 non-CCPD 中大量失败样本仍然保留为 `plain_plate`，这符合“正面牌也可直接训练 plain 路线”的判断
- CBLPRD 的 special 已开始实质进入 manifest，但受当前字符表限制，仍不是全量 special

---

## 3. manifest v3 路径

- `/home/wzzz/LPRNet/manifests/unified_manifest_v3.csv`
- `/home/wzzz/LPRNet/manifests/unified_manifest_v3.summary.json`

总数：`803055`

---

## 4. family 现状

当前 manifest v3 中已经不止两个 family：
- `normal7`
- `green8`
- `special`

其中 `special` 已包含这些 sub_type：
- `yellow_single`
- `yellow_double`
- `tractor_green`
- `black`

但要注意：
- 带 `学/挂/使/领/临/澳/港` 等字符的样本，当前字符表还不支持
- 所以 special 现在只是“部分接入”，不是完全体

---

## 5. train_LPRNet.py manifest 入口更新

已新增更顺手的单文件入口：
- `--manifest`

当使用：
- `--data_mode manifest --manifest /home/wzzz/LPRNet/manifests/unified_manifest_v3.csv`

会自动把同一个 manifest 同时用作：
- train_manifest
- test_manifest

这样不需要再手动重复传两次。
