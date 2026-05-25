# 下一步执行计划（严格承接既定路线）

## 已完成的三步

### 第 1 步：统一 manifest
已完成。

产物：
- `/home/wzzz/LPRNet/build_unified_manifest.py`
- `/home/wzzz/LPRNet/manifests/unified_manifest_v1.csv`
- `/home/wzzz/LPRNet/manifests/unified_manifest_v1.summary.json`
- `/home/wzzz/LPRNet/UNIFIED_MANIFEST_SPEC.md`

关键点：
- 已显式编码 `preprocess_group`
- 已显式编码 `has_bbox / has_quad / can_perspective`
- 已把 `targeted_green_missing_18` 正式归为 `plain_plate`
- 没有把非 CCPD 数据误判成可走透视链

### 第 2 步：冻结 baseline
已完成。

产物：
- `/home/wzzz/LPRNet/baselines/blue_expert_official.json`
- `/home/wzzz/LPRNet/baselines/tilt_ocr_obbwarp_v7_stagec_best.json`
- `/home/wzzz/LPRNet/BASELINE_FREEZE.md`

关键点：
- 历史蓝牌专家已冻结
- 当前最佳 stageC 已冻结
- 当前最佳 stageC 的 normal/hard 指标已经写入基线文件

### 第 3 步：基于 manifest 改造训练入口
已完成第一阶段准备。

产物：
- `load_data.py` 新增 `UnifiedManifestDataset`
- `train_LPRNet.py` 新增 `data_mode=manifest`
- 可通过 manifest 直接读 mixed preprocess 数据

当前边界：
- pseudo anchor 仍只支持 `ccpd_board`
- 这符合当前现实，不属于跑偏

---

## 按原计划，下一步该做什么

现在不应该直接上 special，也不应该跳去别的路线。

### 下一阶段：统一模型 v1 的最小实现准备
严格按原计划，接下来应该做：

1. 基于 manifest 拆出 unified v1 的 train / val 子清单
   - normal7
   - green8
   - 暂不引入 special head

2. 设计 unified v1 的最小结构草图
   - `shared_backbone`
   - `type_head`
   - `normal7_head`
   - `green8_head`

3. 冻结第一版 unified v1 的验收口径
   - 蓝牌 normal exact 不可明显退化
   - hard tilt 不能比当前最优 stageC 大幅退步
   - 绿牌必须单独评估
   - 非 CCPD plain_plate 数据不能误套 ccpd_board 评估口径

---

## 当前不应做的事

1. 不要把非 CCPD plain_plate 数据硬改成 obb_warp 流程
2. 不要跳过 unified v1，直接上大而全 special 统一模型
3. 不要把历史蓝牌专家和当前 stageC 最优 baseline 混成一个对照物

---

## 推荐紧接着执行的两个具体任务

### 任务 A
做 unified v1 的 manifest 子集构建脚本：
- train_normal7.csv
- train_green8.csv
- val_normal7.csv
- val_green8.csv

### 任务 B
写 unified v1 的模型结构草图和训练接口草图。

这两个任务完成后，才是正式进入“统一模型第一版实现”。
