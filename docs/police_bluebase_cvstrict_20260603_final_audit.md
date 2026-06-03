# Police BlueBase CVStrict — Final Audit

## 三路线最终对比

| 维度 | v2 official baseline | **BlueBase + special_v2** | BlueBase + strict CV smoke |
|:----|:-------------------:|:-------------------------:|:--------------------------:|
| Checkpoint | `police_v2_fullft_officialwarm_20260601/` | **`police_bluebase_cvstrict_20260603/stageC/`** | `police_bluebase_cvstrict_20260603/strictcv_stageC/` |
| Warm start | official (CCPD pretrain) | **蓝牌 v7_stageC_Final** | 蓝牌 v7_stageC_Final |
| 训练数据 | special_v2 (15500) | special_v2 (15500) | strict CV smoke (1550) |
| val_clean exact | **99.48%** | 97.35% | 98.39% |
| val_hard exact | **98.26%** | 95.29% | 69.03% |
| **真实 mgC0001J province=蒙** | **0%**（全→青） | **100%** | **0%**（乱码） |
| 真实 mgC0001J exact | 0% | **50%** | 0% |
| 是否需要 sidecar | **是**（蒙→青需 sidecar） | **否**（蓝背 innate） | — |
| 导出候选？ | ❌ | **✅** | ❌ |

## 最终 Verdict

**唯一导出候选：`experiments/police_bluebase_cvstrict_20260603/stageC/best_LPRNet_model.pth`**

| 项目 | 通过？ | 备注 |
|------|:------:|------|
| 合成 val_clean >= 97% | ✅ 97.35% | 低于 v2 的 99.48%，但 province fix 是 trade-off |
| 真实 dump province=蒙 | ✅ 100% | 核心修复：青→蒙 |
| 0 I/O | ✅ | |
| 0 length errors (val_clean) | ✅ | |
| 0 jing@pos1 | ✅ | |
| 真实 dump exact | ⚠️ 50% | 剩余 50% 是 body 0↔U 混淆（独立问题） |

## 明确限制

1. **真实验证范围**：只覆盖了 `蒙C0001警` 这一组 dump（40 帧，同一辆车/场景）。其他省份、其他车辆未验证。
2. **Province 修复成立**：蓝牌 backbone 继承的 province 识别能力从 "青" 修复为 "蒙"，sidecar 不再需要。
3. **Full exact 仍只有 50%**：body 0↔U 混淆（`C0U01警` vs `C0001警`）是独立问题，需要后续微调或板端规则处理。
4. **Strict CV 结论**：当前 smoke 渲染实现不泛化到真实域（0%），但不要扩大为 "所有 strict CV 思路永久无效"。改进渲染（车牌专用字体 / 真实 ocrin 混合训练）可能改变结论。

## 导出建议

1. **先用 ONNX 导出复测**（export_checklist.md 第 2 项）
2. **确认 FixedNorm 不引入动态 Reshape**（之前 embassy 踩过坑）
3. **RKNN 转换后在 simulator 上过 40 帧真实 dump**
4. **板端 A/B 比较**：v2 official vs BlueBase + special_v2
