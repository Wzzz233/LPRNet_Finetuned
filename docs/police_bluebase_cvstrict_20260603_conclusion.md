# Police BlueBase CVStrict — 最终结论

## 三组实验结果对比

| 实验 | 训练数据 | val_clean | real 蒙 province | 真实泛化 |
|:----|:--------|:---------:|:----------------:|:--------:|
| v2 baseline | special_v2 (15K) | 99.48% | 0%（全→青） | ❌ 省份错 |
| BlueBase + special_v2 | special_v2 (15K) | **97.35%** | **100%**（蒙 ✅） | ✅ 省份修好 |
| BlueBase + strict CV smoke | strict CV (1.5K) | **98.39%** | **0%**（乱码） | ❌ 完全崩 |

## 关键结论

1. **BlueBase + special_v2 是目前唯一成立的组合**。97.35% val_clean + 100% 真实蒙 province，不需要 sidecar。

2. **Strict CV 数据合成方法有根本问题**。虽然在自身验证集上达到 98.39%，但真实域上完全失效（0%）。根因推测：WQY 字体渲染 + CV 特征过度匹配源域纹理，生成的图像与真实 ocrin 差别太大。

3. **不推荐继续 strict CV 路线**。当前渲染质量不足以支撑真实域泛化。要改进需要更好的字体匹配（车牌专用字体而不是 WQY）或直接使用真实 ocrin 图。

4. **推荐方向**：用 BlueBase + special_v2 的最佳 checkpoint（`stageC/best_LPRNet_model.pth`）导出 RKNN，收集 20-30 张真实 ocrin 图做最终微调。
