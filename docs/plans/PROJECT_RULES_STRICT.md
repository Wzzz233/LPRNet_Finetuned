# LPR 项目严格规则（已落地执行版）

## 1. bbox / quad / 透视参数

### 必须满足
- 真实图片必须使用真实 bbox / quad。
- 当前主线默认透视链路固定为最新纠正后的 exact-quad / 板端一致 warp。
- 生成图片进入主训练前，必须确认已经经过 `obb_best.pt`（YOLOv8 OBB）或等效真实 OBB 标注流程。
- 风格转换后的图片进入主训练前，必须重新跑一次 OBB，重新得到 bbox / quad。
- 不允许把风格转换前的框直接继承到风格转换后的图片。
- 不允许使用全图固定框、占位框、统一 bbox、统一透视参数冒充真实标注。
- 必须保留 OBB 推理输出位置、重建后数据集位置、抽查记录位置。

### 当前工程硬规则
- `source=pseudo_geom` 时：
  - `bbox_source` 必须是 `detector_obb` 或 `mapping_csv`
  - `quad_source` 必须是 `detector_obb` 或 `mapping_csv`
- `preprocess_group=ccpd_board` 时：
  - `has_bbox=1`
  - `has_quad=1`
  - `can_parse_ccpd_geom=1`
  - `can_perspective=1`
- 带有 style-transfer / stylized / translated / cycle / realmix / fastcut 等迹象的数据，如果要进主训练，必须重新 OBB 标注；当前检查器会把未使用 `detector_obb` 的这类样本报错。

---

## 2. 预处理与加载链路一致性

### 板端一致固定参数
当前主线固定要求：
- `ocr_crop_mode=obb_warp`
- `ocr_channel_order=bgr`
- `ocr_resize_mode=letterbox`
- `ocr_resize_kernel=nn`
- `ocr_preproc=none`
- `ocr_min_occ_ratio=0.90`
- `ocr_quad_pad_ratio=0.0`

### 当前已落地的强制措施
- `train_LPRNet.py` 启动时会强制检查这组参数；不一致直接报错。
- `eval_lpr_detailed.py` 启动时也会强制检查这组参数；不一致直接报错。
- manifest 中 `ccpd_board` / board warp 样本必须写成这组固定参数。
- `build_manifest_v3.py` 与 `append_nonccpd_pseudo_geom_to_manifest.py` 已统一改为通过共享策略模块写入这组固定参数，避免脚本各写一套。

---

## 3. 训练前必须做的实际检查

### 必须做
- 必须实际用 loader 读取样本，不能只假设路径正确。
- 必须确认 loader 解出的标签和数据集标签一致。
- 必须确认实际输入张量尺寸正确。
- 如果只是为了兼容 loader 改文件名，必须明确说明这是兼容动作，不得当作真实重标注。

### 当前可直接使用的检查工具
1. `check_loader_alignment.py`
   - 用于抽查 loader 是否真的能读样本、标签是否一致、输入尺寸是否正确。

2. `verify_lpr_pipeline_rules.py`
   - 用于项目级检查 manifest：
     - 预处理参数是否严格符合板端一致口径
     - pseudo_geom / style-transfer-like 样本是否满足规则
     - bbox / quad 元信息是否符合约束
     - 是否存在缺失文件
     - 随机抽样检查几何是否不是同一组固定值

---

## 4. 当前共享策略模块

文件：`/home/wzzz/LPRNet/lpr_pipeline_policy.py`

作用：
- 固定板端一致预处理参数
- 提供 `apply_board_params()`，给 manifest 构建脚本统一写参数
- 提供 `validate_manifest_row()`，给检查脚本统一验规则
- 提供 style-transfer / pseudo_geom / board_pipeline 的规则校验

---

## 5. 当前推荐执行顺序

1. 构建或更新 manifest
2. 运行 `verify_lpr_pipeline_rules.py`
3. 运行 `check_loader_alignment.py`
4. 再进入训练或评估

如果第 2 或第 3 步不过，不允许继续往下跑
