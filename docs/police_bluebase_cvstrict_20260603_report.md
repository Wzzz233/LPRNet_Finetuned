# Police BlueBase CVStrict — 最终报告

## 实验概述

**目标**: 使用蓝牌专家权重 v7_stageC_Final 作为 police OCR 的 backbone 基底，通过三阶段训练（冻结 backbone→半放开→全量微调）来构建警牌 OCR。

**日期**: 2026-06-03

## 蓝牌权重选择

从 4 个候选蓝牌模型中选定了 `experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth`：
- 68 类标准 CHARS 输出（31 provinces + 10 digits + 24 letters + I/O/-）
- 最优蓝牌识别 backbone（tilt/db/challenge 上 60.6%）
- 结构：62 层 state_dict，container.0.weight shape=[68, 516, 1, 1]

## Init Checkpoint

继承 54/62 层，8 层因 shape 差异（68→67 类）随机初始化：
- backbone.0-19 ✅ 继承
- backbone.20-21 ➡ 随机初始化（channel 数从 68→67）
- container.0 ➡ 随机初始化（515→516 输入特征变化）
- BN running stats ✅ 继承

## 训练过程

| 阶段 | Epoch | LR | Backbone 状态 | val_clean |
|:----|:-----:|:--:|:------------:|:---------:|
| A | 20 | 1e-3 | 冻结 (54/62层继承) | 35.9% |
| B | 20 | 5e-4 | 放开 backbone.16-21 (6层) | 71.2% |
| C | 30 | 1e-4 | 全量微调 (所有层) | **97.6%** |

## 完整评估

| 评估集 | 指标 | Stage C 结果 | police v2 baseline | 对比 |
|--------|------|:-----------:|:------------------:|:----:|
| val_clean (500) | exact | **97.60%** | 99.48% | -1.88pp |
| val_hard (500) | exact | **94.20%** | 98.26% | -4.06pp |
| val_clean | province | **97.60%** | 99.81% | -2.21pp |
| val_hard | province | **96.00%** | 98.65% | -2.65pp |
| 真实 mgC0001J (40) | province=蒙 | **100%** | 0% (全→青) | **+100pp** |
| 真实 mgC0001J (40) | exact=蒙C0001警 | **50%** | 0% | **+50pp** |

## 关键结论

1. **蓝牌 backbone 完美修复蒙→青混淆**：真实 dump 上 province=蒙 100%。不需要 sidecar 融合，不需要条件替换。

2. **v2 模型 vs bluebase 的核心差异**：
   - v2 用 official warm start（CCPD 预训练），从通用车牌特征出发
   - bluebase 用蓝牌专家权重，已经学会了蓝色车牌上的 31 省字形
   - 蓝牌权重中的省字特征迁移到了警牌，而 v2 的 official warm 没有（需要从 0 学）

3. **合成指标低于 v2**（val_clean -1.88pp, val_hard -4.06pp）：可能是 bluebase 的 backbone 在蓝牌数据上过拟合，需要更多警牌训练来调整。

4. **Body 0↔U 混淆仍存在**：真实 dump 上 20/40 的 exact 失败全是 body 问题（C0U01警 vs C0001警），这是 v2 也有的独立问题。

## 是否推荐继续？

**推荐。** BlueBase 路线在真实域省份识别上明显优于 v2 baseline（100% vs 0%）。以下是具体建议：

### 下一步最小行动项（按优先级）

1. **收集真实警牌 ocrin 图继续训练**：用 20-30 张不同省的真实警牌 ocrin 图做额外 fine-tune，预期可以抹平合成指标差距。

2. **更长的阶段 C 训练**：当前 30 epoch 后 loss 还未完全收敛。再跑 20 epoch 可能提升 1-2pp。

3. **val_hard 专项提升**：当前 hard 94.2% vs clean 97.6%，差距 3.4pp。可以加 hard 增强训练。

4. **Body 0↔U 问题**：这是一个跨模型问题，v2 和 bluebase 都有。可以考虑 character-level 辅助 loss。

### 是否推荐导出 RKNN？

**暂不。** 合成指标还没完全追上 v2 baseline。建议先跑 1（收集真实 ocrin 图继续训练），等合成指标到 99% 之后再考虑导出。

## 文件清单

| 文件 | 路径 |
|------|------|
| Init checkpoint | `experiments/police_bluebase_cvstrict_20260603/init_from_bluebase_police_keys.pth` |
| Stage A best | `experiments/police_bluebase_cvstrict_20260603/stageA/best_LPRNet_model.pth` |
| Stage B best | `experiments/police_bluebase_cvstrict_20260603/stageB/best_LPRNet_model.pth` |
| Stage C best | `experiments/police_bluebase_cvstrict_20260603/stageC/best_LPRNet_model.pth` |
| Stage C final | `experiments/police_bluebase_cvstrict_20260603/stageC/Final_LPRNet_model.pth` |
| 训练 log | `experiments/police_bluebase_cvstrict_20260603/stageC/train.log` |
| 候选评估 | `experiments/police_bluebase_cvstrict_20260603/blue_candidate_eval.json` |
| 计划 | `docs/police_bluebase_cvstrict_20260603_plan.md` |
