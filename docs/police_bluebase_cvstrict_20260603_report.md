# Police BlueBase CVStrict — 最终报告（修订版）

## Part A: BlueBase + special_v2 三阶段训练（已完成）

使用 `manifests_rebased/special_split_v2_20260601/train_police.csv`（旧 v2 合成数据）的三阶段训练结果。

| 阶段 | Epoch | Backbone | val_clean exact | val_clean province | val_hard exact | real 蒙 province |
|:----|:-----:|:--------:|:---------------:|:------------------:|:--------------:|:----------------:|
| A | 20 | 冻结 | 35.9% | — | — | — |
| B | 20 | backbone.16-21 | 71.2% | — | — | — |
| C | 30 | 全量 | **97.35%** | **98.06%** | **95.29%** | **100%** |

**关键发现**：蓝牌 backbone 继承修复了蒙→青混淆。真实 dump 上 province=蒙 100%，不需要 sidecar。

**注意**：这个结果用的是 `special_v2` 的旧合成数据，不是新的 strict CV 数据。

## Part B: Strict CV 替换集（新增）

### 数据生成

| 属性 | Smoke 值 | 正式目标 |
|------|:--------:|:--------:|
| 源图池 | `pose_quads.jsonl`（200K） | 同左 |
| 每省 train | **50** | 500 |
| 每省 val_clean | **10** | 50 |
| 每省 val_hard | **10** | 50 |
| 总图数 | **2,170** | 21,700 |
| CV 特征 | 9 类（亮度/模糊/噪声/JPEG/曝光/饱和度/对比度/清晰度） | 同左 |
| 格式 | province+letter+4alnum+警 | 同左 |
| I/O | 0 | 0 |
| 31 省均衡 | ✅ | ✅ |

### Smoke 训练状态

- Stage A（frozen backbone, 5 epoch）：🔄 运行中（epoch 1, loss 14→11）
- 使用 `init_from_bluebase_police_keys.pth` 为 warm start
- 新 manifest 加载正常，police keys 正常

### 正式训练命令

见 `experiments/police_bluebase_cvstrict_20260603/run_strictcv_training.sh`

## 结论

1. **BlueBase + special_v2 三阶段训练：成立。** 97.35% val_clean，真实蒙 100%。建议继续完整 strict CV 训练。

2. **Strict CV 数据：初步通过。** smoke 数据 2,170 张已生成并通过格式检查。正式 21,700 张数据待正式训练。

3. **下一步**：
   - 等 smoke 训练完成 → 确认 manifest 可训练
   - 用 `run_strictcv_training.sh` 跑正式 strict CV 三阶段训练
   - 收集更多真实 police ocrin 图做最终验证
