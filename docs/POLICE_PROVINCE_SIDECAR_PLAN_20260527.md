# Police Province Sidecar 设计计划 — 2026-05-27

## 背景

Police 主 OCR 训练在 per-sample normalization 修复后进行，结果在 tail 警与 province 之间存在系统性冲突。

### Sweep 结论

| aux weight | Full | Province | Letter | Mid4 | Tail 警 | LenMatch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 (B) | 60.00% | 64.52% | 97.74% | 87.74% | **100.00%** | 88.71% |
| 0.05 | 61.94% | 68.06% | 97.74% | 91.67% | 92.90% | 89.03% |
| 0.10 | 64.84% | 69.68% | 98.39% | 92.87% | 93.23% | 89.68% |
| 0.20 | 63.55% | 68.71% | 97.42% | 91.75% | 93.87% | 89.03% |

**结论**：first-char aux loss 系统性破坏 tail 警（100% → ~93%），
所有权重都 fail tail≥98% criterion。停止主 OCR aux 路线。

### 根本原因

警察号牌的特殊格式（最后一位固定为"警"）意味着 CTC 解码器
在序列末尾必须精确输出"警"字。first-char aux loss 增加了一个
与 CTC 主 loss 竞争的训练信号，干扰了尾部解码的稳定性。

## Sidecar 设计目标

**只识别 police 省份首字，不干扰主 OCR body 和末尾"警"。**

| 要求 | 说明 |
|------|------|
| 输入 | police OCR crop 或整牌 quad warp（与主 OCR 同源） |
| 输出 | 31 省份 one-hot |
| 精度目标 | province accuracy ≥ 85% |
| 不修改 | OCR 输出第 1 位 ~ 末尾"警" |
| 不训练 | 主 OCR 模型权重冻结 |

## 模型架构建议

```
Input: 3×H×W (full plate quad warp or OCR crop, gray3 or RGB)
  │
  ▼
ResNet18 (或更小的 ResNet10/MobileNetV3)
  │
  ▼
Linear(512 → 31)
  │
  ▼
Softmax → 31 省份概率
```

### 尺寸选择

| 方案 | 输入尺寸 | 优势 | 劣势 |
|------|:---:|------|------|
| OCR crop | 24×94 | 与主 OCR 同源，共享预处理链 | 视野窄，可能缺上下文 |
| Full plate quad warp | 72×224 | 有更多空间上下文 | 需要额外 warp 步骤 |

推荐先用 full plate quad warp (72×224, gray3)，因为绿牌首字 sidecar
已在这条路线取得成功（`experiments/routeA_prime_quadwarp_20260512/`）。

## 数据

| 数据集 | 路径 | 样本数 |
|------|------|:--:|
| Train | `manifests_rebased/special_split_20260526/train_police_only.csv` | 3,720 |
| Val | `manifests_rebased/special_split_20260526/val_police_only.csv` | 310 |

数据源：CCPD2019 base + pose quad cvreplace，省份均衡生成（每省 ~120 train / 10 val）。

### 标签生成

从 `text` 字段提取第 0 位字符作为省份标签：
```
藏VV3FM警 → label=藏 (index 0-30)
```

31 省份映射已在 `keys/police_keys.txt` 中定义（前 31 行为省份）。

## 评估指标

| 指标 | 计算方式 | 目标 |
|------|------|:---:|
| Province accuracy | sidecar 首字正确率 / 310 | ≥ 85% |
| Fused full exact | (OCR 第0位替换为 sidecar 第0位) 全串正确率 / 310 | ≥ B baseline 60% |
| Tail invariance | sidecar 融合前后 tail 警 accuracy 不变 | = 100% |
| False overwrite | sidecar 改正了原本正确的首字 / 310 | 尽量少 |

### 不要仅关注 province accuracy 单项

必须同时验证 fused full exact 和 tail invariance。
首字侧路的价值在融合后体现，单独看 province accuracy 会高估收益。

## 融合策略

```
if plate_routed_as_police:
    ocr_pred = main_ocr(image)          # 主 OCR 输出全串
    prov_pred = sidecar(image)          # sidecar 输出省份
    prov_conf = max(softmax(prov_pred)) # 省份置信度

    if prov_conf >= threshold:          # 高置信才替换
        ocr_pred[0] = sidecar_province
    # 第 1 位到末尾"警"永远不改
```

### 安全措施

1. **最低置信阈值**：sidecar softmax max < threshold 时不替换（建议初始 0.80）
2. **Multi-frame voting**：板端多帧一致时替换，不一致时保留 OCR 原输出
3. **格式校验**：只替换 police 格式（7 位，末位=警）的样本
4. **Source check**：仅当板端颜色路由已将 plate 标记为 police 时才启用 sidecar

## 板端接入前置条件

当前 ARM **不支持** police/embassy 分离路由。主 OCR 模型通过
`--ocr-special-model` 使用通用的 special 模型（包含所有特殊牌型）。

板端接入 police sidecar 需要：
1. ARM UNKNOWN 二级路由区分 police vs embassy
2. 新增 `--police-sidecar-model` 参数
3. 触发条件：颜色分类为 UNKNOWN → 二级路由 → police → sidecar

这些 ARM 改动**不在本计划范围内**，仅记录为前置条件。

## 训练计划（暂不执行）

以下仅为参考，不在本次工作中执行：

```
Warm start: ImageNet pretrained ResNet18 (或随机初始化)
Train samples: 3,720 (31 省 × 120)
Val samples: 310 (31 省 × 10)
Epochs: 30-50
Optimizer: Adam LR=1e-3
Loss: CrossEntropy
Augmentation: gray3, mild brightness jitter
```

## 相关实验参考

绿牌 province sidecar 成功案例：
- 路线：`experiments/routeA_prime_quadwarp_20260512/`
- 报告：`experiments/routeA_prime_quadwarp_20260512/ROUTEA_PRIME_SUMMARY.md`
- 关键结论：224×72 + gray3 + ResNet18 可行；94×24 小模型不可行

## 禁止事项

- 不要修改主 OCR 输出第 1 位到末尾"警"
- 不要在 sidecar 训练中 unfreeze 主 OCR backbone
- 不要混入 embassy 数据
- 不要用 CCPD2020 base 数据（police 数据是 CCPD2019 base + cvreplace）

## 相关文档

- 拆分方案：[SPECIAL_POLICE_EMBASSY_SPLIT_20260526.md](SPECIAL_POLICE_EMBASSY_SPLIT_20260526.md)
- 绿牌首字参考：[Route A' Summary](../experiments/routeA_prime_quadwarp_20260512/ROUTEA_PRIME_SUMMARY.md)
