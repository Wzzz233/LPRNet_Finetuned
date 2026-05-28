# Police / Embassy 拆分方案 — 2026-05-26

## 为什么不混训

police 和 embassy 外观域差异过大：

| 属性 | Police | Embassy |
|------|--------|---------|
| 底板颜色 | 白色 | 黑色 |
| 字符颜色 | 黑色（首位/警字红色） | 白色（使字红色） |
| 格式 | 省 + 字母 + 4位字母数字 + 警 | 使 + 6位数字 |
| 字符集 | 66 chars (31省 + 10数字 + 24字母 + 警) | 11 chars (0-9 + 使) |
| class_num | 67 | 12 |
| ARM 颜色分类结果 | UNKNOWN (99%) | UNKNOWN (98.5%) |

混训一个 72-key special OCR 会强制两个差异极大的任务共享同一个输出空间，
导致容量竞争和非法字符污染（如 police 输出中出现 港/领/澳/使）。

**禁止训练 police+embassy 混合模型**。禁止将 yellow_single 加入 police/embassy 训练。

## 各自数据规模

| 数据集 | Police | Embassy |
|--------|:------:|:-------:|
| Train | 3,720 | 3,000 |
| Val | 310 | 300 |
| 数据来源 | CCPD2019 base + pose quad cvreplace | 同 |
| Manifest 目录 | `manifests_rebased/special_split_20260526/` | 同 |

## 各自 Keys 文件

| | Police | Embassy |
|------|--------|---------|
| 文件 | `keys/police_keys.txt` | `keys/embassy_keys.txt` |
| 字符数 | 66 | 11 |
| class_num | 67 | 12 |

police_keys.txt 内容：31 省 + 0-9 + A-Z(无I/O) + 警
embassy_keys.txt 内容：0-9 + 使

## Smoke 起点权重

两者都使用 `experiments/special_special_v2/best_LPRNet_model.pth` 的 backbone warm start。

该权重 class_num=73 (72-key special_keys)，与 police(67) 和 embassy(12) 的 class_num 不匹配。
训练代码 `train_LPRNet.py` 的 shape-aware 加载逻辑 (line 1314) 自动跳过不匹配的 head 层，
backbone 0-19 层 (54 layers) 正常加载，其余层随机初始化。

```
Police: backbone=54 loaded, head=0 → 34,572 params from scratch
Embassy: backbone=54 loaded, head=0 → 5,532 params from scratch
```

## Smoke 结果 (freeze_backbone, 100-300step)

| | Police (100-step) | Police (300-step) | Embassy (100-step) |
|------|:---:|:---:|:---:|
| Accuracy | 0% | 0.32% (1/310) | 0% |
| 非法字符 | 无 | 无 | 无 |

## Overfit64 诊断 (全 backbone unfreeze, Adam LR=1e-3)

**结论：链路完全正确，模型可以 100% 学会这两个任务。**

| 指标 | Police (class_num=67) | Embassy (class_num=12) |
|------|:------:|:------:|
| Step 0 | 0% | 0% |
| Step 40 | **92.2%** | 53.1% |
| Step 60 | **100%** | **100%** |

这证明：decode/keys/label/crop 链路无 bug。前期 probe 失败纯属训练策略问题。

## Embassy A/B/C 策略对照 (special_special_v2 warm, 1000-step)

| 策略 | Trainable | Accuracy | Len Err |
|------|:--------:|:--------:|:-------:|
| A: head-only | 5,532 | **0%** | 76.3% |
| B: unfreeze backbone.20+21 | ~38K | **65.7%** | 32.7% |
| C: unfreeze backbone.18-21 | ~76K | **65.7%** | 32.7% |

**结论**：head-only (A) 永远不收敛。backbone.20+21 (B) 已足够。B/C checkpoint SHA256 相同，无法有效证明 C 有额外收益；当前按更小策略 B 优先。

## Police A/B/C 策略对照 (special_special_v2 warm, 1000-step)

（待 ablations 完成填入）

## Police 黄牌 warm start 对照

黄牌 `yellow_single_v2_weighted_phase2/best` 仅作为 police B 策略对照。
黄牌起点不含警字但含 I/O/学/挂，class_num=70 vs police=67。

**黄牌不进入 police 训练数据**，仅作 warm start 对比。

（待 ablations 完成填入）

## 正式训练候选策略

### Embassy 正式策略 ✅ FixedNorm 重训完成，达到导出前验收阈值

| 指标 | 旧 (buggy eval) | 旧 (fixed eval) | **FixedNorm 重训** |
|------|:---:|:---:|:---:|
| Checkpoint | embassy_formal_20260526 | 同左 | **embassy_formal_fixednorm_20260526** |
| Accuracy | 95.00% (BUG) | 78.67% | **90.33% (271/300)** |
| Best epoch | 25 | — | **14** |
| Threshold | ❌ 虚高 | 未达标 | **✅ >=90% 导出前验收通过** |

```
Warm start:      special_special_v2/best (backbone only)
Keys:            keys/embassy_keys.txt (class_num=12)
Trainable:       backbone.20, backbone.21, head
Aux:             --first_char_aux_weight 0
LR schedule:     --lr_schedule 20,40,60
Max steps:       3000
Max epoch:       15
Code:            FIXED per-sample normalization
```
#### Embassy ONNX 导出 (2026-05-27) ✅

| Artifact | Path | SHA256 |
|------|------|------|
| ONNX (opset 18) | `artifacts/embassy_LPRNet_fixednorm_20260526.onnx` | d18a53a47bfabdc1fc3d86435cedc84c2a4b3c7fe0074b2a5e4124d2ba7683b5 |
| ONNX (opset 11, legacy) | `artifacts/embassy_LPRNet_fixednorm_20260526_op11.onnx` | 59941f49930d7005cbb13a0c696ac2566d874ba6f1508f6964f8a70dfc2cd52c |

ONNX/PyTorch Decode Consistency: **300/300 predictions match** (both opsets).
  Max numerical diff: 2.29e-04 (above 1e-4 strict threshold, but does not affect CTC greedy decode).
  Incremental check CSVs: `artifacts/embassy_LPRNet_fixednorm_20260526_onnx_decode_check_{10,50,100,300}.csv`

RKNN Conversion: **COMPLETED (2026-05-27)** ✅ — rknn-toolkit2 2.3.2 on WSL2.

| RKNN | Path | SHA256 | Size | ONNX source | Build warnings |
|------|------|--------|:--:|------|:--:|
| fp16 (primary) | `artifacts/embassy_LPRNet_fixednorm_20260526_fp16.rknn` | `4159d2ede625426cdc09df98be16584366681b7d400b23d92240854be2d9cce9` | 745 KB | op11 | 5× "Unkown op target: 0" (non-blocking) |
| fp16 (fallback) | `artifacts/embassy_LPRNet_fixednorm_20260526_fp16_op18.rknn` | `91dfb7a6020249f36023c1364a14d4489681bde0fd82dcc0cebda1129909f3ee` | 742 KB | op18 | same 5 warnings |

Conversion params: `target_platform=rk3568, do_quantization=False, mean_values/255=[0,0,0], std_values/255=[1,1,1]`

Simulator: **NOT RUN** — requires ADB-connected RK3568 board (`init_runtime(target='rk3568')` → "no devices/emulators found").
  Board-side decode verification still required before deployment.

Export Report: updated in `artifacts/embassy_LPRNet_fixednorm_20260526_export_report.txt`
RKNN Handoff: `artifacts/embassy_LPRNet_fixednorm_20260526_rknn_handoff.md`

### Police 正式策略 ❌ FixedNorm 重训未达标，需要重新设计策略

| 指标 | 旧 (buggy training) | **FixedNorm 重训** |
|------|:---:|:---:|
| Checkpoint | police_probe_20260526_3k | **police_formal_fixednorm_20260526** |
| Accuracy | 79.03% (fixed eval) | **60.00% (186/310)** |
| Best epoch | — | **31** |
| Threshold | 接近80% | **❌ 远低于80%** |

```
Warm start:      special_special_v2/best (backbone only)
Keys:            keys/police_keys.txt (class_num=67)
Trainable:       backbone.20, backbone.21, head
Aux:             --first_char_aux_weight 0
LR schedule:     --lr_schedule 20,40,60
Max steps:       3000
Max epoch:       60
Code:            FIXED per-sample normalization
```

**错误分析 (124 errors):**
- First-char (province) errors: 110 (88.7%) ← 主导
- Tail (警) misses: 25/310 in current full-val eval; mostly length truncation
- Length errors: 35/124
- 省份混淆分散 (无单一主导): 粤→青 3/10, 吉→闽 2/10, etc.
- 成因推测: frozen backbone (layer 0-19) 在 buggy 代码下训练, 其输出的特征分布
  为 batch-dependent normalization 校准, 而 per-sample 训练改变了归一化尺度,
  导致 backbone.20-21 难以适应。Embassy 字符集小 (11 chars) 受影响较小。

#### Police FixedNorm 诊断

**B 策略位置错误统计 (3k steps, 60.00%)**

| Segment | Accuracy |
|------|:---:|
| Province (char0) | **64.52%** ← 瓶颈 |
| Letter (char1) | 97.74% |
| Middle 4 chars | 87.74% |
| Tail 警 | 91.94% (285/310, corrected full-val eval) |
| Length match | 88.71% |

省份混淆高度分散：97 对混淆，无单一主导（最大：粤→青 3/10）。省份首字识别率仅 64.52% 是唯一瓶颈。

CSV: `experiments/police_formal_fixednorm_20260526/error_breakdown_by_position.csv`
混淆矩阵: `experiments/police_formal_fixednorm_20260526/province_confusion.csv`

**C1/C2 策略对照 (1000-step probes)**

| Strategy | Full | Province | Tail | LenErr |
|------|:---:|:---:|:---:|:---:|
| B (baseline 3k) | 60.00% | 64.52% | 91.94% (285/310) | 35 |
| C1 (unfreeze 18-21, 1k) | 43.23% | 50.00% | 100% | 60 |
| C2 (first-char aux 0.2, 1k) | 42.26% | 55.16% | 100% | 53 |

- C1/C2 在 1k 步时均未超越 B 策略 3k 的 60%，但 1k 步对比不公平。
- C2 省份 55.16% 相比 B 在相似步数的 ~32% 有提升，first-char aux 可能有效。
- 两者都远低于旧 buggy probe 的 79.03%，说明 per-sample 归一化训练
  与 batch-dependent 预训练 backbone 存在系统性兼容问题。

**C2 3k 延长验证 (2026-05-27)**

| Strategy | Full | Province | Letter | Mid4 | Tail 警 | LenMatch |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| B (baseline 3k) | 60.00% | 64.52% | 97.74% | 87.74% | 100.00% | 88.71% |
| **C2 3k (first-char aux 0.2)** | **63.55%** | **68.71%** | 97.42% | 91.75% | 93.87% | 89.03% |

```
Checkpoint:  experiments/police_fixednorm_probe_firstaux02_3k/best_LPRNet_model.pth
Best epoch:  46
Trainable:   backbone.20, backbone.21, head (freeze_backbone=true)
Aux:         --first_char_aux_weight 0.2
Max steps:   3000 (explicit --max_epoch 80)
Code:        FIXED per-sample normalization
```

**结论：C2 3k 超过 B baseline，但后续 sweep 发现主 OCR 仍无法同时满足 province 和 tail 阈值，aux 路线已停止。见下节。**

但是：
- Tail 警仍低于 98% 部署阈值，主 OCR 不能单独进入导出。
- Province 68.71% 仍远低于可用水平 (<80%)，但相比 B 的 64.52% 显著改善。
- 省份混淆仍然高度分散（最大 2 票），无单一主导错误。

**Aux weight sweep 结果 (2026-05-27) — STOP aux 路线** ❌

| aux weight | Full | Province | Letter | Mid4 | Tail 警 | LenMatch | 目录 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|------|
| 0.00 (B) | 60.00% | 64.52% | 97.74% | 87.74% | **91.94% (285/310)** | 88.71% | `police_formal_fixednorm_20260526` |
| 0.05 | 61.94% | 68.06% | 97.74% | 91.67% | 92.90% | 89.03% | `police_fixednorm_probe_firstaux005_3k` |
| 0.10 | 64.84% | 69.68% | 98.39% | 92.87% | 93.23% | 89.68% | `police_fixednorm_probe_firstaux010_3k` |
| 0.20 | 63.55% | 68.71% | 97.42% | 91.75% | 93.87% | 89.03% | `police_fixednorm_probe_firstaux02_3k` |

**结论：ALL aux weights fail tail 警 ≥ 98% criterion，province 仍低于可用水平。停止 aux 路线。**

规律：
- Province 随 aux weight 略有提升（+4-5pp），但仍不足以支撑 police 主 OCR 单独部署。
- Tail 警在所有主 OCR 候选中都低于 98% 部署阈值。
- aux=0.10 是 full/province 局部最优，但仍不可导出。
- 继续调 first-char aux 不解决根问题，下一步应使用独立 province sidecar。

**Police 下一步推荐：**
1. Province sidecar 网络（独立 ResNet18 做首字分类）— Phase 1 已通过 clean synthetic / hard synthetic 审计
2. 板端 UNKNOWN 二级路由后接入 sidecar
3. 真实板端 police 图像验证 sidecar 鲁棒性
4. 不再继续 first-char aux 路线

## Police Province Sidecar Phase 1 审计 (2026-05-28)

Sidecar 已完成第一阶段验证：224×72 全牌图 + ResNet18 只识别第 0 位省份。当前结论是 sidecar 路线成立，但还没有完成真实板端验证。

| 检查 | 结果 |
|------|------|
| 数据 | Train 3,720 / Val 310，31 省均衡；train/val base image、文本、warped basename 无重叠 |
| 标准 val | gray3 pretrained / color pretrained / gray3 random 均达到 100% province acc |
| Hard holdout | 186 张不同生成批次，gray3 pretrained 186/186 = 100% |
| 匿名路径 | 图片改名为 `000000.png` 后仍 310/310 = 100% |
| Mask left 25% | 10/310 = 3.23%，遮住省份后接近随机 |
| Mask right 75% | 310/310 = 100%，只保留左侧省份区域仍可识别 |
| Fusion | 主 OCR 60.00% → sidecar 替换首字后 86.77%，changed wrong = 0 |
| Tail | 285/310 → 285/310，sidecar 不改变 tail |

结论：没有发现数据泄露；模型确实读取图像左侧省份字符。100% 的原因是 clean synthetic 任务本身简单，不代表板端完成。下一步必须用真实板端 police 图像验证噪声、模糊、定位误差下的鲁棒性。

产物：

```text
scripts/train_police_province_sidecar.py
datasets/police_province_sidecar_20260528/fullplate_224x72/
manifests_rebased/police_province_sidecar_20260528/
experiments/police_province_sidecar_20260528/
```

## ⚠️ Batch-Dependent Normalization Bug (2026-05-26)

### 发现

训练 eval 报告 embassy 95% 但单张验收仅 78.67%，排查发现根因为：

`src/LPRNet.py` 第72行 (及 LPRNet_multihead.py, export_onnx_rknn_*.py):
```python
f_mean = torch.mean(f_pow)  # 跨 B*C*H*W 全局均值 — batch-size 依赖
```

当 batch=256 时归一化因子与 batch=1 不同，导致同一样本在 batch 和单张模式下输出不一致。

### 修复

5个文件改为 per-sample 归一化：
```python
f_mean = torch.mean(f_pow.view(f_pow.size(0), -1), dim=1, keepdim=True).view(f_pow.size(0), 1, 1, 1)
f = torch.div(f, f_mean.clamp_min(1e-12))
```

修复文件：
- `src/LPRNet.py`
- `src/LPRNet_multihead.py`
- `src/export/export_onnx_rknn_compatible.py`
- `src/export/export_onnx_rknn_multihead.py`
- `src/utils/verify_export_consistency.py`

### 影响

| 场景 | 是否受影响 |
|------|:------:|
| batch>1 离线 PyTorch eval | ❌ 所有历史数字不可靠，需修复后复评 |
| 单张推理 (batch=1) | ✅ 不受影响 (batch=1时等价于per-sample) |
| ARM 板端 / RKNN 推理 | ✅ 单张推理不受影响 |
| 训练过程 | ⚠️ 模型权重在 batch-dependent 下训练，修复代码后重训可能获得更好单张精度 |

### 验证

回归测试 `scripts/test_batch_invariance.py` — 3 test suites all PASS:
- LPRNet single-head: batch=1 vs batch=3/256 — PASS
- LPRNetMultiHead baseline: family outputs — PASS
- LPRNetMultiHead with pos0 + adapters — PASS

Embassy 验证: batch=1 vs batch=256, 300/300 predictions agree.

## 训练可行性状态

| 任务 | 状态 |
|------|:----:|
| Embassy FixedNorm 重训 | ✅ 完成 (90.33%, 导出前最终验收已完成) |
| Embassy ONNX 导出 | ✅ 完成 (op18 + op11, 300/300 decode consistent) |
| Embassy RKNN 转换 | ✅ 完成 ([handoff](../artifacts/embassy_LPRNet_fixednorm_20260526_rknn_handoff.md)) |
| Police FixedNorm 重训 (B, baseline) | ❌ 未达标 (60.00%, <80%) |
| Police C2 first-char aux 3k | ✅ 完成 (63.55%, 超过 B) |
| Police aux weight sweep (0.05/0.10/0.20) | ❌ 停止 — province 和 tail 均未达部署阈值 |
| Police 主 OCR | ⏸️ 暂停，主 OCR 不继续训练 |
| Police province sidecar | ✅ Phase 1 审计通过 ([sidecar plan](POLICE_PROVINCE_SIDECAR_PLAN_20260527.md))；待真实板端 police 图验证 |
| Police 旧 probe | 📋 仅参考 (79.03% fixed-eval, buggy training) |
| LPRNet 归一化修复 | ✅ 5文件已修复 + 回归测试 PASS (9/9) |
| Police+embassy 混合训练 | ❌ 永远禁止 |
| Export ONNX/RKNN | ✅ Embassy ONNX/RKNN done; board/simulator decode check pending |

## 关于历史 batch eval 数字

**明确规则**：所有 batch>1 的离线 PyTorch 评估报告（train_log 中的 test acc、
Greedy_Decode_Eval 输出等）在修复前的代码上不可作为验收依据。
板端/RKNN 单张推理不因此直接作废。

后续离线评估必须确保 batch 不影响单样本输出（已通过修复保证）。

## 训练与部署强制规则

1. **禁止 batch-dependent normalization 作为训练策略**：训练、导出、ONNX、RKNN 必须全程使用 per-sample normalization。
2. **禁止"训练 batch-dependent、导出 per-sample"方案**：这会导致训练分布与推理分布不一致，产生不可预期的性能退化。
3. **已有模型复评规则**：任何 batch>1 离线 PyTorch 评估在修复前代码上的数字不可作为验收依据。
4. **Police 当前状态**：主 OCR 训练暂停，aux 路线已停止。province sidecar Phase 1 审计通过，但真实板端 police 图验证前不得宣称部署完成（[设计文档](POLICE_PROVINCE_SIDECAR_PLAN_20260527.md)）。
