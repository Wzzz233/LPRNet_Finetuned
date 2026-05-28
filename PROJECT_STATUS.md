# LPRNet 工作区状态

> 最后更新: 2026-05-28（特殊牌 police/embassy 已拆分；LPRNet FixedNorm 已修复；Embassy ONNX/RKNN 已完成，Police 主 OCR 暂停转 sidecar）

---

## 当前状态总览

| 项目 | 状态 |
|------|------|
| 数据集目录 | ✅ CCPD2020、生成数据、原始 CCPD2019、special cvreplace 数据仍在位 |
| 训练代码 (src/) | ✅ 已修复 LPRNet batch-dependent normalization；训练/导出统一使用 per-sample normalization |
| manifests/ (旧绝对路径) | ✅ legacy 目录保留，仍可向后兼容 |
| manifests_rebased/ | ✅ 已新增 `green_v5_final_20260509/`、`ccpd2020_green_real_20260509/`、`green_backbone_mix_realprimary_20260510/`、`a_ablation_20260510/`、`a_ratio_sweep_20260510/`、`b_lite_probe_20260510/`、`routeA_firstchar_r50_20260512/`、`routeA_prime_quadwarp_20260512/`、`yellow_phase1_20260515/`、`special_split_20260526/` |
| experiments/ | ✅ 已新增 `green_ccpd2019_v5_final_20260509/`、`mix_source_audit_20260510/`、`green_backbone_mix_*_20260510/`、`a_ablation_*_20260510/`、`a_ratio_*_20260510/`、`b_prime_*_20260510/`、`b_lite_*_20260510/`、`routeA_firstchar_r50_20260512/`、`routeA_prime_quadwarp_20260512/`、`routeA_nextstage_20260512/`、`routeA_epochcurve_20260512/`、`fc224_board_eval_20260512/`、`yellow_phase1_20260515/`、`embassy_formal_fixednorm_20260526/`、`police_*fixednorm*_20260526/` |
| 根目录外围文件 | ✅ 已整洁化 |
| Model Zoo 专家包 | ⚠️ 目录结构完整，但当前仍是 lightweight index；四个 `checkpoints/` 目录都为空 |
| Manifest Rebase | ✅ 已完成（336 份） |
| 训练脚本 dataset_root 支持 | ✅ 已添加 |
| 训练链路验证 | ✅ yellow 系列通过 |
| **LPRNet FixedNorm** | ✅ 已完成 — 单头、多头、导出脚本统一 per-sample normalization；`scripts/test_batch_invariance.py` 通过 |
| **普通单排黄色车牌 Phase 1** | ✅ 已收敛并已上板验证；ARM 已有按颜色路由到 yellow OCR 的分流逻辑 |
| **特殊牌 cvreplace 全量生成** | ✅ 已完成 — 14,150 张；已拆为 `yellow_single` 路由审计、`police`、`embassy` 三路 |
| **Embassy 专家** | ✅ FixedNorm 重训 90.33%；ONNX op11/op18 300/300 decode consistent；RKNN fp16 已转换，待板端/simulator decode check |
| **Police 主 OCR** | ⏸️ 暂停 — FixedNorm 主 OCR 最高 64.84%，first-char aux 会系统性伤 tail“警”；下一步转 province sidecar |
| **蓝牌 CCPD2019 posquad v1** | ✅ 完成 — 旧蓝牌 53.0% → **59.5%** (+6.5pp) |
| **蓝牌 posquad v2 hardmine** | ✅ 完成 — v1 59.5% → **60.6%** (+1.1pp) |
| **蓝牌退化检查** | ✅ 无退化 — simple/val/hard 均提升 |
| **绿牌 CV-Replace 探索** | ✅ 完成 — 从 old_green 14.5% → **v4b_clean 41.3%** (+26.8pp) |
| **绿牌 v5 多股真实+合成探索** | ✅ 完成 — v5_sweet_spot (iter 8000): cvr_val 42.7% / gv_wan 61.3% / nw_p1st 80.6% |
| **绿牌域差距诊断 Q1/Q2/Q3** | ✅ 完成 — 池化 5.8K 真实 vs 2.7K 合成样本 |
| **绿牌 CCPD2020 分布审计** | ✅ 完成 — val/test 主流干净，5-10% 极端样本 |
| **绿牌三模型角色定位** | ✅ 历史结论已归档 — 仅适用于 CCPD2020 / 合成 proxy，不再作为当前主线判断依据 |
| **真实板端 OCRIN 审计** | ✅ 已完成 — 旧 val / 合成 proxy 明显高估真实板端表现 |
| **A/B 路线审计** | ✅ 已完成 — A 当前有效；B 当前 stage1 产物不应继续接 stage2 |
| **A 系列数据作用拆解** | ✅ 已完成 — replace 主导 tilt，real 主导 static/province 稳定性 |
| **A ratio sweep** | ✅ 已完成 — `R50` 是首个通过当前 post-province sweet-spot 定义的比例点 |
| **B' / B-lite 尝试** | ✅ 已完成 — 三种 B 线变体均失败，当前应封存 B 线 |
| **真实极端场景鲁棒性验证** | ⚠️ 仅完成小规模板端 probe；更大规模真实极端集仍缺失 |
| **Province-Split / 首字侧路 Route A** | ✅ 第一阶段完成 — `94x24` tiny 路线已证伪，`224x72 + gray3 + ResNet18` 路线已在控制板端集上跑通 |
| **Route A 公平审计** | ✅ 完成 — `G0_baseline_repro/best.pt` 当前是最强审计候选（`dump2=90%` / `dump=72%` / `province_stress=98.2%`） |
| **Route A epoch 曲线审计** | ✅ 完成 — `dump2` 峰值在 `epoch 3`，`dump / province_stress` 更偏 `epoch 15`，不存在单一全局最优 epoch |
| **Province-Split 当前主线状态** | ⏸️ 已完成第一阶段验证；当前绿牌效果已满足阶段目标，sidecar 保留为可选增强/诊断项，不再作为继续烧训练预算的理由 |
| **ARM 固定 224x72 首字 dump** | ✅ 已接入并推送 — 可直接输出 `fc224_*.ppm` 做板端对齐审计 |
| **绿牌当前部署主线** | ✅ 冻结维护 — `R50` 主 OCR 为当前推荐主模型；`prov_deg` 仅作为旧稳定回退；首字 sidecar 可选 |
| Manifest/数据集清理 | 📋 候选清单已生成，未执行 |
| 回滚方案 | ✅ 可用 |

---

## 2026-05-28 增量更新（特殊牌拆分、FixedNorm、Embassy RKNN）

这一节优先于下方 2026-05-17 的特殊牌生成链路描述。旧章节仍保留作为生成链路来源，但当前执行口径以本节为准。

### 1. LPRNet batch-dependent normalization 已修复

已确认旧 LPRNet forward 中存在 batch-size 依赖归一化：

```python
f_mean = torch.mean(f_pow)
```

该写法会跨 `B*C*H*W` 求全局均值，导致同一张图在 `batch=1` 和 `batch>1` 离线评估时输出不同。结论：

- 修复前所有 `batch>1` 离线 PyTorch eval 数字不可直接作为验收依据。
- ARM / RKNN 单张推理不因此自动作废，因为 `batch=1` 时旧写法等价于 per-sample mean。
- 后续训练、导出、ONNX、RKNN 必须统一使用 per-sample normalization，禁止“训练 batch-dependent、导出 per-sample”的混合方案。

已修复文件：

```text
src/LPRNet.py
src/LPRNet_multihead.py
src/export/export_onnx_rknn_compatible.py
src/export/export_onnx_rknn_multihead.py
src/utils/verify_export_consistency.py
```

回归测试：

```text
scripts/test_batch_invariance.py
```

当前结果：single-head、multihead、multihead+aux 均通过 batch invariance 检查。

### 2. 特殊牌数据和任务拆分

特殊牌不再走 police+embassy 混合 OCR，也不再把普通黄牌混入 special OCR。

当前正式拆分：

| 任务 | 用途 | Train | Val | Keys |
|------|------|---:|---:|------|
| `yellow_single` | 仅用于颜色路由 sanity check；普通黄牌仍走 yellow OCR | 6,200 | 620 | `keys/yellow_keys.txt` |
| `police` | 警牌 OCR / 后续 province sidecar | 3,720 | 310 | `keys/police_keys.txt` |
| `embassy` | 使馆牌 OCR | 3,000 | 300 | `keys/embassy_keys.txt` |

拆分 manifest：

```text
manifests_rebased/special_split_20260526/
```

权威说明文档：

```text
docs/SPECIAL_POLICE_EMBASSY_SPLIT_20260526.md
```

### 3. Embassy 当前状态

Embassy 专家已完成 FixedNorm 重训、ONNX 导出、RKNN fp16 转换。

关键指标：

| 项目 | 当前值 |
|------|------|
| Best checkpoint | `experiments/embassy_formal_fixednorm_20260526/best_LPRNet_model.pth` |
| FixedNorm val | 271/300 = **90.33%** |
| Batch consistency | 300/300 predictions agree |
| ONNX/PyTorch decode consistency | 300/300 predictions match (op11 + op18) |
| RKNN toolkit | `/root/miniconda3/envs/rknn_env`, rknn-toolkit2 2.3.2 |
| RKNN target | `rk3568`, fp16, `do_quantization=False` |

Artifacts:

```text
artifacts/embassy_LPRNet_fixednorm_20260526.onnx
artifacts/embassy_LPRNet_fixednorm_20260526_op11.onnx
artifacts/embassy_LPRNet_fixednorm_20260526_fp16.rknn
artifacts/embassy_LPRNet_fixednorm_20260526_fp16_op18.rknn
artifacts/embassy_LPRNet_fixednorm_20260526_export_report.txt
artifacts/embassy_LPRNet_fixednorm_20260526_rknn_handoff.md
```

RKNN sha256:

| Artifact | SHA256 |
|------|------|
| `embassy_LPRNet_fixednorm_20260526_fp16.rknn` | `4159d2ede625426cdc09df98be16584366681b7d400b23d92240854be2d9cce9` |
| `embassy_LPRNet_fixednorm_20260526_fp16_op18.rknn` | `91dfb7a6020249f36023c1364a14d4489681bde0fd82dcc0cebda1129909f3ee` |

限制：

- Simulator / 板端 decode check 尚未完成；当前没有 ADB 连接的 RK3568 板端。
- Embassy 模型不能替换通用 `--ocr-special-model`。
- 后续必须先实现 UNKNOWN 二级路由，再将 embassy 作为独立专家接入。
- 板端部署前至少做 50-sample simulator 或板端 decode check。

### 4. Police 当前状态

Police 主 OCR 当前暂停。FixedNorm 后主 OCR 的主要瓶颈是省份首字。

关键结果：

| 路线 | Full | Province | Tail 警 | 结论 |
|------|:--:|:--:|:--:|------|
| B baseline (`aux=0`) | 60.00% | 64.52% | 100.00% | tail 稳定，但首字不足 |
| C2 (`aux=0.20`) | 63.55% | 68.71% | 93.87% | full/province 提升，但 tail 退化 |
| `aux=0.10` | 64.84% | 69.68% | 93.23% | full/province 局部最优，但 tail 不达标 |

结论：

- 所有 first-char aux 权重都会系统性伤害末尾“警”。
- Police 主 OCR 不继续加步数，不导出 RKNN。
- 下一步改为独立 province sidecar，只修第 0 位，不碰第 1 位到末尾“警”。

Sidecar 计划：

```text
docs/POLICE_PROVINCE_SIDECAR_PLAN_20260527.md
```

### 5. ARM / 板端口径

ARM 已有颜色分流：

- `GREEN` -> green OCR
- `YELLOW` -> yellow OCR
- `UNKNOWN` -> special OCR
- 默认 -> blue OCR

但当前 ARM 仍只有单一 `--ocr-special-model` 槽位。Police / Embassy 双专家部署前需要 UNKNOWN 二级路由：

- police 白底 -> police OCR 或 police province sidecar + police OCR
- embassy 黑底 -> embassy OCR

本轮未改 ARM。

---

## 2026-05-17 增量更新（Yellow Phase 1 收敛）

这一节记录普通单排黄色车牌第一阶段探索结论。当前执行口径是：先把最接近蓝牌迁移能力、最容易板端闭环的普通单排黄牌做好；不要让稀有牌型或结构差异更大的牌型拖累主线。

### 1. 第一阶段目标边界

当前 Phase 1 只做：

- 普通单排黄色车牌

当前 Phase 1 明确不做：

- 挂车牌
- 学牌
- 双层黄牌
- 小摩托黄牌
- 警用号牌
- 使馆号牌
- 领馆号牌

这些牌型后续可以另开阶段，但不再混入普通单排黄牌主线。

### 2. 当前推荐黄牌模型

当前普通单排黄牌离线候选仍使用旧黄牌模型：

```text
experiments/yellow_single_v2_weighted_phase2/best_LPRNet_model.pth
```

已有板端/中间 artifact：

```text
artifacts/yellow_LPRNet_v5_phase2.onnx
artifacts/yellow_LPRNet_v5_phase2_fp16.rknn
```

注意：上述 artifact 已存在，但仍需后续确认是否与当前推荐 checkpoint 严格对应。

当前 clean 普通黄牌离线指标：

| 测试集 | accuracy |
|------|:--:|
| `val_yellow_normal_real` | 67.18% |
| `test_yellow_normal_real` | 62.65% |
| `val_yellow_normal_single` | 90.09% |

### 3. 有效数据集

Yellow Phase 1 清洗后保留的普通单排黄牌 manifest：

| Manifest | 行数 | 说明 |
|------|---:|------|
| `manifests_rebased/yellow_phase1_20260515/train_yellow_normal_clean.csv` | 86,783 | 从 `yellow_train_weighted.csv` 过滤 8,292 条“学”牌 |
| `manifests_rebased/yellow_phase1_20260515/val_yellow_normal_real.csv` | 1,420 | 真实单排黄牌验证集，无挂/学 |
| `manifests_rebased/yellow_phase1_20260515/test_yellow_normal_real.csv` | 249 | 真实单排黄牌测试集，无挂/学 |
| `manifests_rebased/yellow_phase1_20260515/val_yellow_normal_single.csv` | 2,130 | CBLPRD 单排黄牌验证集，已去学 |

### 4. 为什么不采用新训练模型

`normal_yellow_stabilize_v1` 没有超过旧黄牌模型：

| 测试集 | 旧模型 | `normal_yellow_stabilize_v1` | 变化 |
|------|:--:|:--:|:--:|
| real val | 67.18% | 67.04% | -0.14pp |
| real test | 62.65% | 61.85% | -0.80pp |
| CBLPRD single val | 90.09% | 90.09% | 持平 |

因此普通单排黄牌 Phase 1 继续推荐 `experiments/yellow_single_v2_weighted_phase2/best_LPRNet_model.pth`，不切换到新训练模型。

### 5. 挂车牌探索结论

挂车牌暂不纳入第一阶段。已验证的失败路线：

1. 主 OCR 加挂训练并解冻 backbone：
   - 模型开始输出“挂”，但普通黄牌 `real_val` 从 67.18% 降到约 60.14%。
2. 低 backbone LR 短程探针：
   - 没找到 `real_val >= 65%` 且 `tail_gua_acc > 0` 的折中 checkpoint。
3. 冻结 backbone、只训练 head：
   - 普通黄牌可训，但“挂”学不动。
4. tail sidecar：
   - “挂”recall 高，但真实黄牌误报率约 2%-4%，超过目标。
5. sidecar + append/replace fusion：
   - 无法修复挂牌 OCR body 崩溃，`test_gua` 无有效提升。

当前结论：

- 挂车牌不能继续污染普通黄牌主线。
- 后续若要做挂车牌，应另开 Phase 2，并重新定义数据、模型结构和板端融合策略。

### 6. 当前下一步

推荐后续顺序：

1. 做黄色车牌颜色分流接入。
2. 校验 `artifacts/yellow_LPRNet_v5_phase2_fp16.rknn` 是否与推荐 checkpoint 对齐。
3. 板端采集真实普通单排黄牌样本验证。
4. 挂车牌另开二阶段，不再与 Phase 1 混训。

---

## 2026-05-17 增量更新（特殊车牌替换生成链路已固化）

这一节记录当前已经落地的“特殊车牌替换到 CCPD2019 base 上”的数据生成链路。该链路当前只用于**数据生成与 QA**，尚未进入训练主线。

### 1. 当前目标边界

当前特殊车牌链路只覆盖三类：

- 单排黄牌 `yellow_single`
- 警用号牌 `police`
- 使馆号牌 `embassy`

当前明确不做：

- 双层黄牌
- 小摩托黄牌
- 挂车牌
- 学牌
- 领馆号牌

### 2. 生成项目位置

当前特殊车牌替换生成项目位于 Windows 工作区：

```text
C:\Users\Wzzz2\OneDrive\Desktop\test\special_gen\chinese_license_plate_generator
```

WSL 对应路径：

```text
/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/special_gen/chinese_license_plate_generator
```

本轮新增/固化的脚本：

```text
lib_special_plate_renderer.py
special_ccpd2019_base_step1_generate.py
```

### 3. 源数据与几何来源

当前替换链路不是直接在干净模板上导出，而是**以 CCPD2019 base 真图为底图做 CV replace**。

几何来源文件：

```text
/home/wzzz/LPRNet/datasets/ccpd2019_base_posquads_20260509/pose_quads.jsonl
```

该文件逐行提供：

- `img_path`
- `text`
- `gt_quad`
- `pose_quad`

其中当前主链路使用 `pose_quad` 作为贴回和 QA warp 的统一透视依据。

### 4. 当前生成链路

当前生成过程固定为：

1. 从 `pose_quads.jsonl` 读取 CCPD2019 base 样本。
2. 按固定随机种子切分 source pool：
   - 前 `val_source_count=2000` 张作为 `val`
   - 其余作为 `train`
3. 按牌型家族构造目标文本：
   - `yellow_single`: `省 + 字母 + 5位字母数字`
   - `police`: `省 + 字母 + 4位字母数字 + 警`
   - `embassy`: `使 + 6位数字`
4. 用 `SpecialPlateRenderer` 渲染规范牌面。
5. 从原始 CCPD2019 base 图上按 `pose_quad` 抽取原车牌区域，作为风格参考 patch。
6. 对渲染牌面做外观迁移：
   - 保留目标牌种的主颜色与字符结构
   - 用 source patch 的亮度分布和低频明暗起伏迁移到新牌面
   - 再施加轻度模糊、噪声和 JPEG 压缩，使其更接近板端处理后的观感
7. 将迁移后的特殊车牌按 `pose_quad` 透视贴回原始整图。
8. 输出整图、manifest、QA 图和统计信息。

注意：

- QA 顶部图现在展示的是**贴回整图后再按 `pose_quad` 反拉正得到的生成结果**，不是干净模板。
- QA 底部图展示原始蓝牌 `pose_quad` warp，便于直接比较“替换后 vs 原图”。

### 5. 当前图像处理口径

为了避免“模板味太重”，当前链路已固定以下处理原则：

- 先做 `L` 通道主导的亮度风格迁移
- 再叠加 source patch 的低频亮度变化
- 若生成结果比 source 明显更锐，则追加轻度高斯模糊
- 若 source 噪声更强，则补少量噪声
- 最后走一轮轻度 JPEG 压缩

当前 QA 预览额外做了轻微退化，口径是：

- `GaussianBlur((3,3), sigma=0.45)`
- `JPEG quality=84`

这一步只用于 QA 观感，不单独改变 manifest 结构。

### 6. 当前输出目录

当前默认输出目录：

```text
datasets/special_ccpd2019_base_cvreplace_v1_20260517/
```

其中包含：

- `train/` 生成训练图
- `val/` 生成验证图
- `qa/` QA 预览图
- `generation_meta.json` 生成统计

当前 rebased manifest 输出目录：

```text
manifests_rebased/special_ccpd2019_base_cvreplace_v1_20260517/
```

### 7. Manifest 产生方法

当前 manifest 由 `special_ccpd2019_base_step1_generate.py` 直接生成，固定分为两份：

- `train_special_base_cvreplace.csv`
- `val_special_base_cvreplace.csv`

字段结构固定为：

- `img_path`
- `text`
- `family`
- `source`
- `split`
- `preprocess_group`
- `has_quad`
- `can_parse_ccpd_geom`
- `can_perspective`
- `quad_source`
- `bbox_source`
- `quad_1x` ~ `quad_4y`
- `ocr_crop_mode`
- `ocr_resize_mode`
- `ocr_resize_kernel`
- `ocr_preproc`
- `ocr_channel_order`
- `ocr_quad_pad_ratio`

当前记录口径：

- `img_path` 为 rebased 相对路径
- `text` 为替换后的新车牌文本
- `family` 标记三种牌型家族
- `split` 标记 `train/val`
- quad 坐标写入生成后整图中的最终四点
- OCR 相关字段直接写入后续训练所需的 crop / resize / preproc 元信息

### 8. Train / Val 样本分配方法

当前样本配额由脚本参数控制。

正式模式默认：

- `train-yellow-per-province = 200`
- `val-yellow-per-province = 20`
- `train-police-per-province = 120`
- `val-police-per-province = 10`
- `train-embassy-total = 3000`
- `val-embassy-total = 300`

Smoke 模式当前用于链路验证，固定缩小为：

- 单排黄牌：每省 `train=3`，`val=1`
- 警牌：每省 `train=2`，`val=1`
- 使馆：`train=20`，`val=10`

source 图使用“先打乱、再循环取样”的调度方式，避免某一小段 source 图被连续重复使用。

### 9. 当前 QA / Smoke 验证口径

当前最小可复现实验命令：

```bash
cd /mnt/c/Users/Wzzz2/OneDrive/Desktop/test/special_gen/chinese_license_plate_generator
python3 special_ccpd2019_base_step1_generate.py --smoke
```

当前 smoke 已验证通过，输出包括：

- train manifest
- val manifest
- QA 图
- `generation_meta.json`

最近一次 smoke 结果确认：

- `train`: 175 条数据
- `val`: 72 条数据
- QA 图已按新口径刷新

### 10. 当前状态结论

当前“特殊车牌替换到 CCPD2019 base”的第一步数据链路已完成固化，状态为：

- ✅ 已能稳定生成单排黄牌 / 警牌 / 使馆牌三类数据
- ✅ 已能输出 train/val rebased manifests
- ✅ 已能输出贴回整图后的 QA 对照图
- ✅ 已完成从“过干净模板观感”到“更接近板端观感”的处理收敛
- ⏸️ 当前只固化到数据生成阶段，尚未进入训练与板端部署阶段

---

## 2026-05-14 增量更新（绿牌阶段冻结）

这一节优先于下方 2026-05-13 / 2026-05-10 绿牌探索叙事。下方历史章节保留作为路线依据，但当前执行口径以本节为准。

### 1. 当前绿牌效果已满足阶段目标

- 当前板端绿牌效果已达到可接受状态，用户确认“目前的绿牌效果很满意”。
- 因此绿牌方向不再继续扩训练分支，不再继续烧 `A ratio`、`B/B'/B-lite` 或新的 end-to-end 单体绿牌专家。
- 后续绿牌工作只做：
  - 部署维护
  - 小范围回归验证
  - 必要时的模型导出 / 参数修正
  - 真实问题复现后的定点修复

### 2. 当前推荐绿牌部署件

| 角色 | 当前推荐 | 路径 |
|------|------|------|
| 绿牌主 OCR | `R50` 主模型，负责后 7 位、长度稳定性、倾斜鲁棒性 | `artifacts/r50_green_ocr/R50_green_multihead_no_rknnpre_rk3568_fp16.rknn` |
| 绿牌旧回退 | `prov_deg` / old_green，保留作稳定回滚 | `experiments/green_e12_province_degrade_unfreeze/prov_deg_fp16_no_rknnpre.rknn` |
| 首字 sidecar | 可选增强 / 诊断项，不作为当前必须闭环项 | `artifacts/green_firstchar_sidecar/green_firstchar_G0_refit_best_rk3568_fp16.rknn` |
| 首字 sidecar 备用 | 静态板端局部候选 | `artifacts/green_firstchar_sidecar/green_firstchar_G0_refit_epoch003_rk3568_fp16.rknn` |

当前板端建议优先以 `R50` 主 OCR 作为绿牌模型替换 `prov_deg`：

```bash
--ocr-green-model /userdata/model/R50_green_no_rknnpre_fp16.rknn
```

首字 sidecar 先保持可选。若启用，建议继续使用多帧投票保护：

```bash
--green-firstchar-min-votes 5
--green-firstchar-min-share 0.60
```

### 3. 当前冻结规则

- 不再继续扩大 `A ratio sweep`。
- 不再重开旧 `B / B' / B-lite`。
- 不再为了修首字去破坏 `R50` 已经获得的后 7 位 / tilt 能力。
- 不再以 CCPD2020 `green_val` 或合成 proxy 单独决定生产模型。
- 若未来绿牌重新开工，必须先有新的真实板端失败样本，并能说明当前 `R50` 部署件在哪类场景稳定失败。

---

## 2026-05-13 增量更新（首字 / Province-Split / 绿牌专家）

这一节优先于下方 2026-05-10 主线总结。下方历史章节仍保留，但若有冲突，以本节为准。

### 1. Province-Split 已从“候选想法”推进到“第一阶段已完成”

- **Route A 小模型路线（`94x24` 独立省份分类器）已证伪**：
  - `A1/A2/A3` 全部塌缩为单一省份
  - `A4` 即使改成 full-crop 也没有形成可用板端表现
- 当前有效路线是 **Route A' / quad-warp fullplate 路线**：
  - 输入：`fullplate 224x72`
  - 预处理：`gray3`
  - 模型：`ResNet18`
  - 数据：按省份均衡的 `91,512` 张整牌 quad-warp 图
- 第一阶段正式成功点：
  - `experiments/routeA_prime_quadwarp_20260512/B3_fullplate_gray3_224x72_bal31/best.pt`
  - 结论报告：`experiments/routeA_prime_quadwarp_20260512/ROUTEA_PRIME_SUMMARY.md`

### 2. 省份第一位当前最强候选不是 B3，而是经过公平审计后的 G0

- Route A Next Stage 已完成公平复评：
  - `experiments/routeA_nextstage_20260512/FAIR_AUDIT_REPORT.md`
- 当前最强审计候选是：
  - `experiments/routeA_nextstage_20260512/G0_baseline_repro/best.pt`
- 统一口径下的关键指标：
  - `dump2 first-char = 90.0%`
  - `dump first-char = 72.0%`
  - `province_stress macro = 98.2%`
  - `fused exact = 65.0%`
- 当前可成立的管理结论：
  - `G1`（boardlike aug）无效且更差
  - `G2`（real upweight）无效且更差
  - `G3`（两者叠加）最差
  - 因此当前**没有证据**支持继续在这条省份 sidecar 上加 boardlike aug 或强 real 上采样

### 3. Checkpoint 选择机制问题已查清；“epoch 1 最强”叙事已撤回

- 旧 `B3` 的 `best.pt` 选择逻辑有 bug，导致 `best.pt` 长期停在 `epoch 1`
- 新的 epoch 曲线审计已完成：
  - `experiments/routeA_epochcurve_20260512/EPOCH_CURVE_REPORT.md`
- 已确认的规律：
  - `dump2` 峰值在 `epoch 3`（最高 100%）
  - `dump / province_stress` 在 `epoch 15` 附近更优
  - 不存在能同时最优静态板端、倾斜板端、stress 的单一 checkpoint
- 当前实际使用建议应按任务拆分：
  - **静态板端优先**：保留 `epoch_003.pt` 作为候选
  - **综合平衡优先**：保留 `G0_baseline_repro/best.pt` / `epoch 15` 一类候选

### 4. 新的 ARM 固定 224x72 dump 已经接入，但“固定 warp”本身不是最终答案

- ARM 端已新增首字专用固定输出：
  - `fc224_*.ppm`
- 代码已在 `VHDL_Project/ARM` 分支提交：
  - commit `beab86c`
- 对应板端对齐评测产物：
  - `experiments/fc224_board_eval_20260512/board_eval_fc224_vs_crop_summary.json`
  - `experiments/fc224_board_eval_20260512/epoch_sweep_fc224_vs_crop_summary.json`
- 当前增量结论：
  - `fc224` 明确比旧 `ocrin 94x24` 更接近训练语境
  - 但在新增真实 dump（`fc_dump=京AD06088`, `fc_dump2=苏BF01111`）上，**固定 `224x72` warp 本身还不能统一解决两组样本**
  - 例如：
    - `G0 best` 在 `fc_dump` 仅 `37.5%`
    - 同模型在 `fc_dump2` 为 `76.7%`
    - `epoch_003` 可把 `fc_dump` 拉到 `75%`，但会把 `fc_dump2` 拉低到 `27.9%`
- 因此当前不能把“加一条固定 warp 输出”误判为生产闭环；它更多是**把板端输入与训练输入对齐、用于定位问题来源**的一步

### 5. 绿牌专家总体训练最新主线

- **主 OCR / 后 7 位 / tilt 专家仍然是 `R50` 路线**
  - 这部分结论没有被首字 sidecar 路线推翻
- **Province-Split 的正确理解**不是再做一个新的单体 end-to-end 绿牌专家，而是：
  - `R50` 继续负责后 7 位、长度稳定性、倾斜鲁棒性
  - 单独引入省份第一位 sidecar 做首字修复
- 这也意味着：
  - `A ratio sweep` 不应继续扩点
  - `B / B' / B-lite` 继续封存
  - 当前新增训练预算应优先投入到**真实板端首字 sidecar 验证**和**更大真实 dump 重采样**，而不是继续拧单体绿牌大模型

### 6. 当前首字/Province-Split 权威入口

| 用途 | 路径 |
|------|------|
| Route A 失败路线总表 | `experiments/routeA_firstchar_r50_20260512/ROUTEA_SUMMARY.md` |
| Route A' 成功路线总表 | `experiments/routeA_prime_quadwarp_20260512/ROUTEA_PRIME_SUMMARY.md` |
| 公平审计 | `experiments/routeA_nextstage_20260512/FAIR_AUDIT_REPORT.md` |
| 训练动力学 / epoch 曲线 | `experiments/routeA_epochcurve_20260512/EPOCH_CURVE_REPORT.md` |
| 新板端 `fc224` 对齐评测 | `experiments/fc224_board_eval_20260512/board_eval_fc224_vs_crop_summary.json` |
| 新板端 epoch 全扫 | `experiments/fc224_board_eval_20260512/epoch_sweep_fc224_vs_crop_summary.json` |

---

## 2026-05-10 当前有效结论（优先于下方历史实验叙事）

这一节是**当前主线**。若与下方 v4/v5 历史章节有冲突，以本节为准。

### 1. 真实板端小样本已经推翻“只看 green_val / 合成 proxy”的判断方式

- `pos_ocr_dump` 的有效倾斜测试段是 **frames 11-40**（`seg_B_tilt`），不是全 50 帧。
- `pos_ocr_dump_2` 是 **static_board_ocrin_control**，不是倾斜集。
- `ocrin_*.ppm` 已经是最终 OCR 输入，不需要再额外预处理。
- 结论：CCPD2020 `green_val` 与合成 `cvr_val` 只能做内部 proxy，**不能再直接外推成真实板端能力**。

### 2. 当前 A/B 审计结论

#### A（green refine 对照）

- 目录：`experiments/green_backbone_mix_branchA_refine_only_20260510/`
- 训练产物可用；`train_summary.json` 缺失属于 provenance defect，不是训练 crash。
- board-optimal checkpoint: `best_LPRNet_model.pth`
- 板端关键指标：
  - `seg_B_tilt pp_char = 47.1%`
  - `seg_B_tilt len_err = 63.3%`
  - `static_board_ocrin_control pp_exact = 52.5%`
- 相对 `old_green`：
  - tilt 字符级更强（`+9.0pp`）
  - 倾斜下长度坍缩显著减少（`-23.4pp len_err`）
- 当前结论：**A 是目前已验证的最强 exploratory branch。**

#### B stage1（mixed-real backbone stage1）

- 目录：`experiments/green_backbone_mix_branchB_stage1_backbone_mix_20260510/`
- `iter_2000` / `iter_4000` 已确认与 A 对应文件 md5 完全相同，属于后拷贝污染，不可用于选点。
- 仅 `best` / `Final` / `last` 三个 checkpoint 有效，且三者板端表现一致：
  - `seg_B_tilt pp_char = 35.7%`
  - `seg_B_tilt len_err = 90.0%`
  - `static_board_ocrin_control pp_exact = 2.5%`
- 当前唯一有效结论：
  - **`DO_NOT_OPEN_B_STAGE2_FROM_CURRENT_B_STAGE1_RUN`**
  - 这是否决“用这次 B stage1 产物继续开 B stage2”，**不是**彻底证伪 mixed-real-backbone 路线本身。

### 3. A 系列数据作用拆解（size-matched ablation，97,767 rows）

已完成三个分支：

| 分支 | board-optimal checkpoint | seg_B pp_char | seg_B len_err | static pp_exact | 主要含义 |
|------|------|:--:|:--:|:--:|------|
| `A_real_only_matched` | `best_LPRNet_model.pth` | 40.5% | 80.0% | **77.5%** | real 主导 static / province / sequence stability |
| `A_replace_only_matched` | `LPRNet__iteration_6000.pth` | 44.8% | 80.0% | **0.0%** | replace 主导 tilt，但 static 完全崩 |
| `A_mix_rebuild` | `LPRNet__iteration_2000.pth` | **46.2%** | **76.7%** | 37.5% | 保留大部分 tilt 收益，并部分挽回 static |

当前可成立的解释：

- **replace 数据是 tilt robustness 的主驱动**
- **real 数据是 static 绿牌稳定性、省份识别、序列结构稳定性的主驱动**
- 当前 `mix` 更像**偏 replace 的折中**，不是强 additivity

### 4. A ratio sweep 定案：`R50` 是当前最佳折中点，但只解决了“后 7 位 + tilt”，没有解决首字

统一协议下的 7 个比例点结论已经完成。当前最重要的结论不是“曲线是否平滑”，而是：

- `R50`（50% real / 50% replace）是**第一个**满足当前 sweet-spot 判定标准的比例点：
  - `seg_B_tilt pp_char = 51.9%`
  - `seg_B_tilt len_err = 70.0%`
  - `static_board_ocrin_control post-province pp_exact = 67.5%`
- 但 `R50` 在 `dump2` 上的 **full exact 仍是 0.0%**
- 省份位偏置仍重：`皖` 仍是主导错误首字

因此，当前可以成立的更精确结论是：

- `R50` 已明显优于 `old_green`，如果你关心的是：
  - 倾斜鲁棒性
  - 后 7 位稳定性
  - 序列长度不崩
- 但 `R50` **没有**把首字/省份位修好
- 所以“继续只调 real/replace 比例”已经接近收益递减区

### 5. green_val 绝对值不再作为这轮 A 系列的主裁判

- `A_mix_rebuild` 没有复现原 A 的 `green_val best = 36.37%`，只达到 `19.7%`
- 已审计：
  - manifest 字节级一致
  - 参数逐项一致
  - 预训练权重一致
  - loss 轨迹几乎一致
- 当前最有力解释是 **CUDA CTC backward non-determinism**
- 因此：
  - `green_val` 仍可作参考
  - 但**本轮 A 配比/消融的主判断必须落在 board-centric checkpoint scan 上**

### 6. B 线当前应封存：旧 B、B'、B-lite 都没有修好首字，且会伤 green8 主体能力

当前已经失败的 B 线尝试包括：

1. **旧 B stage1**
   - 有效 checkpoint 板端已死，不应继续开旧 B stage2
2. **B' Stage 1**
   - `full-backbone + 90% normal7` 导致 green8 guard 直接 collapse
3. **B-lite balanced probe**
   - 从 `R50` 起跑、轻度开放 `backbone.18/19/20`
   - 结果：
     - tilt 明显退化（`seg_B pp_char 51.9% -> 40.5%`）
     - `province_top1_acc` 无改善
     - `皖` 偏置无改善
     - `full exact` 无改善

当前可成立的管理结论：

- **B 线在当前代码路径和当前数据条件下应封存**
- 这不是理论上永久封杀任何 backbone 想法
- 但在没有新的真实非皖 green8 数据、或没有先验证“province repair 机制确实作用到 green8 主 decoder”之前，不应继续烧 B 线

### 7. 当前主线

- **保留 `R50` 作为当前最佳绿牌基线**
- **封存 B 线**
- 当前下一步不再是继续拧 end-to-end 比例，也不是继续 mixed-backbone
- 当前最值得探索的新方向是：
  - **Province-Split / 首字拆分**
  - 即：让 `R50` 继续负责后 7 位与 tilt，单独建模首字/省份位

### 8. 权威入口（后续任何人都应先读这些文件）

| 用途 | 路径 |
|------|------|
| 当前总状态 | `PROJECT_STATUS.md` 当前页首 |
| canonical board scan 协议 | `experiments/mix_source_audit_20260510/board_scan_protocol_reference.md` |
| A ratio sweep 一致性修正 | `experiments/mix_source_audit_20260510/a_ratio_consistency_audit.md` |
| R50 最终总结 | `experiments/mix_source_audit_20260510/a_ratio_r50_summary.md` |
| B' 设计边界 | `experiments/mix_source_audit_20260510/b_prime_design_audit.md` |
| B-lite probe 结果 | `experiments/mix_source_audit_20260510/b_lite_probe_summary.md` |

---

## Model Zoo 当前封装状态

`model_zoo/` 目录结构完整（`checkpoints/` `board_artifacts/` `configs/` `keys/` `manifests/` `logs/`），
但当前仍是 **lightweight index**：四个专家目录的 `checkpoints/` 都为空，`board_artifacts/` 下也只有 README 引用原始路径，
并不是自包含权重包。`lineage.json` 记录的是当前引用关系，而不是“权重已复制到包内”。

| 专家目录 | checkpoints/ 内容 | lineage.json 引用 | 待更新 |
|------|-----------|--------|--------|
| `yellow_from_blue_expert/` | **空** | yellow_LPRNet_v5_phase2 | 若需要离线交付，仍需复制 checkpoint / board artifact |
| `special_plate_expert/` | **空** | special_LPRNet_fp16 | 待 smoke test；若需交付仍需复制权重 |
| `blue_plate_expert/` | **空** | old `tilt_ocr_obbwarp_v7` + `LPRNet_stage3_rk3568_fp16_more_trained.rknn` | posquad v2 best 待导出 |
| `green_plate_expert/` | **空** | `green_e12_province_degrade_unfreeze` (old_green) | 当前实际部署口径已更新为 `R50` 主 OCR；model_zoo 仍未封装，后续如需交付再复制 board artifact |

**候选封装权重**（均在 experiments/ 下，未移入 model_zoo）：
- `experiments/green_e12_province_degrade_unfreeze/best_LPRNet_model.pth` (old_green，最干净真实皖)
- `experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v4b_clean_20260509/best_LPRNet_model.pth` (v4b_clean，真实非皖最佳)
- `experiments/green_ccpd2019_v5_final_20260509/v5_sweet_spot_iter8000_LPRNet_model.pth` (v5_sweet_spot，合成鲁棒性最佳)
- `experiments/a_ratio_r50_20260510/best_LPRNet_model.pth` (`R50`，当前绿牌主 OCR / 后 7 位 / tilt 主候选)
- `artifacts/r50_green_ocr/R50_green_multihead_no_rknnpre_rk3568_fp16.rknn` (`R50` 已导出的当前板端推荐绿牌主 OCR)
- `experiments/routeA_nextstage_20260512/G0_baseline_repro/best.pt` (当前最强首字 / 省份 sidecar 候选)
- `experiments/routeA_epochcurve_20260512/G0_refit/checkpoints/epoch_003.pt` (`dump2` 静态板端优先时的首字 sidecar 候选)
- `experiments/blue_ccpd2019_posquad_v2_hardmine_20260508/best_LPRNet_model.pth` (蓝牌 posquad v2 best)

---

## 蓝牌 Posquad 训练实验索引

### v1 — 基础 Posequad 微调

**目标**: 用 YOLOv8n-pose 对 CCPD2019 tilt/db/challenge 重新标注四点，用 pose quad 训练蓝牌 LPRNet。

**数据量**: pose 标注 90,258/90,351 (99.9%) · train 86,566 · test 3,692

**训练**: 从旧蓝牌专家开始 5 epoch, LR 1e-4 → 1e-6

| Epoch | LR | Test Acc |
|-------|:--:|:--------:|
| 1 | 1e-4 | 49.2% |
| 2 | 1e-4 | 42.6% |
| 3 | 1e-5 | **58.3%** |
| 4 | 1e-6 | **59.0%** |
| 5 | 1e-6 | **59.6%** |

**关键路径**:
- `scripts/step1_pose_inference_ccpd2019.py` — 用 YOLOv8n-pose 推理 tilt/db/challenge
- `scripts/step2_build_ccpd2019_posquad_manifest.py` — 根据 eval 划分 train/test
- `scripts/step3_ccpd2019_posquad_qa.py` — 生成 contact sheet 和自动质检
- `datasets/ccpd2019_tilt_db_challenge_posquads_20260508/` — pose quads (jsonl)
- `manifests_rebased/blue_ccpd2019_tilt_db_challenge_posquad_20260508/` — train 86,566 / test 3,692
- `experiments/blue_ccpd2019_tilt_db_challenge_posquad_20260508/` — best 权重 + 评估 JSON

**诊断结论**: pose quad 质量不是瓶颈（GT quad 反而比 pose quad 低 3.5pp）。瓶颈是字符混淆（D↔0, Z↔2, B↔8）和省份不均衡（皖 87.5%）。

**评估 JSON**: `experiments/blue_ccpd2019_tilt_db_challenge_posquad_20260508/final_old_vs_new_eval.json`
**诊断 JSON**: `experiments/blue_ccpd2019_tilt_db_challenge_posquad_20260508/diagnosis.json`

### v2 — Hardmine + Province Balanced 第二轮

**目标**: 基于诊断结果做 hard mining + 省份均衡，进一步提升。

**策略**:
- Hard mining: 从 v1 训练集挖 28,032 个错例，x2 权重
- 省份均衡: 皖从 87% 降至 70%，非皖省份提升权重
- 混入普通蓝牌数据（blue_simple/val_ccpd2019）防退化
- brightness aug 20, province_balance_mode inv_sqrt

**训练**: 从 v1 best 继续 5 epoch, LR 5e-5 → 5e-7

| Epoch | LR | Hardmine Test Acc |
|-------|:--:|:-----------------:|
| 1 | 5e-5 | 41.85% |
| 2 | 5e-5 | 42.67% |
| 3 | 5e-6 | **45.51%** (best) |
| 4 | 5e-7 | ~45% |
| 5 | 5e-7 | 44.98% |

**关键路径**:
- `scripts/step5_diagnose_posquad_v1.py` — 诊断 v1: 位置/省份/置信度/几何分析
- `manifests_rebased/blue_ccpd2019_posquad_v2_hardmine_20260508/` — train 71,549 / test 3,766
- `experiments/blue_ccpd2019_posquad_v2_hardmine_20260508/` — best 权重 + 评估 JSON

**评估 JSON**: `experiments/blue_ccpd2019_posquad_v2_hardmine_20260508/final_old_v1_v2_eval.json`

---

## 绿牌 CCPD2019 CV-Replace 实验索引

### 背景

将蓝牌已成功的 CCPD2019 tilt/db/challenge + YOLOv8n-pose quad 方案复制到绿牌域。
生成绿牌替换训练数据：从 CCPD2019 原图提取车牌区域 CV 特性，生成 clean 绿牌，迁移退化特征（亮度/模糊/噪声），贴回原图。

### 关键技术决策

| 决策 | 结论 |
|------|------|
| Color transfer | **L-only** — 只迁移 L 通道亮度/对比度，A/B 保留绿牌自身色相 |
| Color guard | blue_ratio < 0.10，超标跳过 |
| Quad refiner | **放弃** — pose quad 比 GT quad 好 |
| expE (512ch head) | **不需要** — 与 expD (256ch) 无显著差异 |
| Focal CTC | **无效** — 不解决 cvr_val 平台 |
| **Data coverage** | **关键瓶颈** — 16.9% → 72.6% 覆盖率带来 +15pp |

### v2 — L-only CV Transfer 首版

| Epoch | LR | cvr_val |
|-------|:--:|:-------:|
| 1 | 5e-5 | 21.3% |
| 2-5 | — | ~25% |
| **Best** | — | **25.6%** |

生成 20,256 张，source coverage ~17%。从 old_green 起跑。
发现 **green_val 96% 皖**，需要 皖/non-皖 拆分评估。
**修正**: 颜色迁移从 LAB full match 改为 L-only。

### v3 — Source-Balanced Manifest

| Epoch | LR | cvr_val |
|-------|:--:|:-------:|
| 1 | 5e-5 | 19.1% |
| 2 | 5e-5 | 23.9% |
| **3** | **5e-6** | **25.7%** |
| 4 | 5e-7 | 25.5% |
| 5 | 5e-7 | 25.7% |

E12 real cap = 2000/prov. Source ratio: real 49%, cvr 25%, prov 13%, other 12%.
皖=4.4%, max/avg=1.42.

### Phase 2 — Soft-Unfreeze backbone.16

| Epoch | LR | cvr_val |
|-------|:--:|:-------:|
| 1 | 5e-6 | 25.4% |
| **3** | **5e-7** | **26.3%** |
| 5 | 5e-8 | 26.2% |

Marginal gain. Additional backbone unfreezing doesn't help.

### expE — 512ch Head Capacity Probe

Warmstart from expD → expE (72.2% argmax agreement). Same data, same training.

| Epoch | LR | cvr_val |
|-------|:--:|:-------:|
| 1 | 5e-6 | 25.0% (warmstart dip) |
| **3** | **5e-7** | **26.1%** |
| 5 | 5e-8 | 26.1% |

**结论**: 与 expD 无差异。Head capacity 不是瓶颈。

### Focal CTC — Anti-Collapse Attempt

`--ctc_loss_type focal --focal_ctc_alpha 0.5 --focal_ctc_gamma 2.0 --rear_seq_aux_weight 0.3`

| Epoch | cvr_val |
|-------|:-------:|
| 1 | 24.7% |
| 2 | 25.0% |
| **5 (best)** | **25.5%** |

**结论**: 与 v3 无差异。CTC collapse 不是主瓶颈（校正后 short_rate 30%，非 57.8%）。

### 数据覆盖率审计

| 指标 | 值 |
|------|:--:|
| 可用 pose sources | 90,258 |
| v3 cvreplace 使用 | 15,257 |
| **覆盖率** | **16.9%** |
| 未使用 sources | 75,001 |

**根因**: 生成脚本用 `random.choices` with replacement，大量 source 重复，83% 从未使用。

### v4 — Full Coverage Generation

**改进**: 用 unique source scheduling 替代 replacement sampling。每 source 最多用一次再循环。

| 指标 | v3 | v4 |
|------|:--:|:--:|
| 生成图片 | 20,460 | **93,836** |
| Unique sources | 15,257 | **65,493** |
| 覆盖率 | 16.9% | **72.6%** |
| Starting from | v3_best | **old_green** (clean) |

| Epoch | LR | cvr_val |
|-------|:--:|:-------:|
| 1 | 5e-5 | 28.6% |
| 2 | 5e-5 | 33.9% |
| 3 | 5e-6 | 39.3% |
| **7 (best)** | **5e-8** | **40.9%** |

**突破性结果**: cvr_val 从 25.7% → 40.9%，+15.2pp。数据覆盖率是瓶颈。
**副作用**: green_simple 56.9%, green_hard 86.5% — cvr 占比 63% 过高。

### v4b — Rebalanced Continuation

从 v4 best 继续。Manifest: cvr 60%, real 27%, prov 7%, other 6%.

| Epoch | cvr_val |
|-------|:-------:|
| 1 | 41.8% |
| **5 (best)** | **42.8%** |
| green_simple | 58.6% |
| green_hard | 87.7% |
| **nw_p1st** | **80.6%** |

所有指标改善，nw_p1st 从 74.2% → 80.6% (+6.4pp)。

### v4b Clean — 干净起跑

从 old_green 起跑，v4b manifest，10 epoch。

| Epoch | cvr_val |
|-------|:-------:|
| 1 | 28.7% |
| 2 | 36.2% |
| 5 | 41.0% |
| **9 (best)** | **41.3%** |

---

## 绿牌 v5 探索（CCPD2020 真实绿牌 + 多股 CV-Replace）

### 背景与动机

v4c probe 3-epoch 探查（A_wan15, B_wan25）发现 cvr 降比收益有限且 3 epoch 无法外推。
Supervisor 审计 v4b 训练数据发现**严重污染**：
- v4b "real" 流 41,375 行里，**green8 只有 5,200 行**
- 其余 36,175 行是 `normal7`（蓝牌）+ `special`（黄牌 + 特殊牌）和 CCPD2019 tilt 原图
- v4b 能工作，但很大程度上是在学习"多家族真实摄影特征"而非纯绿牌

### v5 数据配方（零非绿牌污染）

**只保留 4 股纯绿牌数据**：

| 流 | 行数 | 占比 | 性质 |
|:-|---:|---:|:-|
| CCPD2020 真实绿牌 train（新建 manifest）| 5,769 | 3.57% | 🟢 纯真实 |
| ccpd2020_replace_pose_v3 train | 2,790 | 1.73% | 🔴 生成（CCPD2020 bg + LAB transfer）|
| base_cvreplace train（CCPD2019 base）| 61,120 | 37.81% | 🔴 生成（LAB transfer）|
| train_cvreplace_v4 纯版（CCPD2019 tilt/db/challenge）| 91,998 | 56.90% | 🔴 生成（LAB transfer）|
| **合计** | **161,677** | — | 全 green8，零非绿牌污染 |

**关键澄清**：base/v4/pose_v3 三个"生成流"实际使用**同一套方法**（pose quad 检测 → 合成模板 + LAB 亮度迁移 → 粘贴到 quad），差别仅在源图池。真正的"真实数据"只有 CCPD2020 train 5,769 行，占 3.57%。

**稀释效应**：CCPD2020 train 96.6% 为皖，但合成流省份完美均衡（1.01x）。合并后皖占 6.55%（vs 均值 3.23%），max/min 只有 2.12x，训练开 inv_sqrt 后皖有效过采样仅 1.44x，可接受。

### v5 训练设置

严格复用 v4b_clean 配置（LR 5e-5, schedule [4,6,8], freeze backbone + trainable prefixes 18,19,20, multihead expD, green8 head only）。

- 预训练权重：`experiments/green_e12_province_degrade_unfreeze/best_LPRNet_model.pth` (old_green)
- train_manifest：`manifests_rebased/green_v5_final_20260509/train_v5_final.csv`
- test_manifest（selection proxy）：`manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv` — **⚠️ 这是 96% 皖的 val**

### v5 训练轨迹（完整 iteration 扫描）

每 2000 iter 保存一次 checkpoint。完整评估在 green_val (833 行, 802 皖 + 31 非皖)：

| iter | 估算 epoch | LR | 皖 ex | nw ex | 皖 p1 | **nw p1** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2000 | 0.8 | 5e-5 | 61.7% | 25.8% | 91.0% | 67.7% |
| 4000 | 1.6 | 5e-5 | 59.7% | 25.8% | 94.0% | 71.0% |
| 6000 | 2.4 | 5e-5 | 60.0% | 32.3% | 93.9% | 71.0% |
| **8000** | **3.2** | **5e-5** | **61.3%** | **38.7%** | **93.4%** | **★ 80.6% ★** |
| 10000 | 4.0 | 5e-6 | 61.3% | 38.7% | 93.8% | 77.4% |
| 12000+ | 4.8+ | 5e-6+ | 61.2-61.7% | 32.3% | 93.5% | 71.0% (坍塌) |
| best* | 4 | 5e-5→5e-6 | 61.7% | 32.3% | 93.9% | 71.0% |

*"best" 是 proxy_exact 在 96% 皖 val 上选出来的，选择偏向皖 — 实际为 epoch 4 附近。

**关键发现**：
- iter 8000 (epoch 3.2) 是**真正的 sweet spot**：non-皖 prov1st 峰值 80.6%
- LR drop (epoch 4 末尾) 之后模型开始过拟合 96% 皖 proxy，non-皖 prov1st 从 80.6% 跌回 71.0%
- **被 "best" 选中的 checkpoint 实际上是 wan-overfit 版本**

**修复**：将 iter 8000 固化为 `v5_sweet_spot_iter8000_LPRNet_model.pth`，作为 v5 的部署 checkpoint。

### 绿牌实验完整对比（含 v5 全面更新）

所有模型在 5 条 val 流上的 accuracy（split bug 修复后）：

| Model | cvr_v2 | base_cvr | v4_cvr | pose_v3 | gv_all | gv_wan | gv_nwan | nw_p1st |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **old_green** | 14.5% | 46.6% | 11.8% | 46.1% | **70.8%** | **71.8%** | 45.2% | 80.6% |
| v3_best | 25.7% | — | — | — | 53.8% | — | 45.2% | 77.4% |
| v4 (cvr 63%) | 40.9% | — | — | — | 38.3% | — | 48.4% | 74.2% |
| v4b_continue | 42.8% | — | — | — | 41.9% | — | 48.4% | 80.6% |
| **v4b_clean** | 41.3% | 71.7% | 37.2% | **47.4%** | 44.3% | 44.1% | **48.4%** | **83.9%** |
| v5_final (ep 4) | 43.4% | 72.7% | 40.1% | 37.1% | 60.6% | 61.7% | 32.3% | 71.0% |
| **v5_sweet_spot (iter 8000)** | **42.7%** | **71.2%** | **39.4%** | 38.4% | 60.5% | **61.3%** | 38.7% | **80.6%** |

**v5_sweet_spot vs v4b_clean 关键差异**：
- 🟢 gv_wan: +17.2pp（44.1% → 61.3%，真实皖大赢）
- 🟢 w_p1st: +30.6pp（62.8% → 93.4%）
- 🟢 cvr_v2: +1.4pp, v4_cvr: +2.2pp（合成小赢）
- 🟡 base_cvr_val: -0.5pp（持平）
- 🔴 gv_nwan: -9.7pp（48.4% → 38.7%）
- 🔴 nw_p1st: -3.3pp（83.9% → 80.6%）
- 🔴 pose_v3_val: -9.0pp（47.4% → 38.4%）

### v5 的结构性结论

1. **没有一个模型全面优于其他** — 这是一个帕累托前沿，不是单调进步
2. **v5 是皖域/长三角专家**，v4b_clean 仍是全国均衡冠军
3. 一个**有力解释**是：v4b 的"蓝牌污染"额外提供了真实摄影成像物理信号和非皖字符曝光，这和它在 pose_v3_val / gv_nwan 上较强的表现一致
4. **v5 清理污染流后 pose_v3_val 回退 9pp**，支持上述解释，但这仍是解释，不是单独证明因果的实验

### 识别的两个根本瓶颈

| # | 瓶颈 | 严重度 | 可破性 |
|:-:|:-|:-:|:-:|
| 1 | **非皖真实绿牌数据全球只有 ~500 行**（196 CCPD2020 train + 31 val + ~200 test）| 🔴🔴🔴 | 难 — 需要外部数据源 |
| 2 | **v5 缺少"真实全图 + 真实车牌"训练样本**（v5 只 5,769 行，v4b 有 ~36K 含非绿牌）| 🔴🔴 | 中 — 可用 multihead backbone 共享 |

### v5 Prep/Training/Eval 产物

**关键产出**（均在 `experiments/green_ccpd2019_v5_final_20260509/` 下）：
- `v5_sweet_spot_iter8000_LPRNet_model.pth` — 推荐部署权重（从训练中间状态提取，唯一拷贝）
- `SWEET_SPOT_README.md` — 权重使用说明
- `ckpt_diagnosis_full.json` — 15 个 checkpoint 的 wan/nw 数据
- `supervisor_eval_sweetspot_20260509.json` — 12 模型 × 5 val 流完整评估

**诊断脚本**（临时文件，仅存于 /tmp/，未纳入 repo）：
- `/tmp/domain_gap_diag.py` — Q1/Q2/Q3 域差距诊断（已执行 2026-05-10）
- `/tmp/q2_expanded_diag.py` — Q2 扩充版池化对比（已执行 2026-05-10）

**Eval 脚本**（`scripts/green_ccpd2019_final_eval.py`）变更：
- MANIFESTS 结构改为 `{name: (path, split)}` 元组，支持 per-manifest split_filter
- 当前 5 个 val 流：cvr_val (test), base_cvr_val (val), v4_cvr_val (test), pose_v3_val (val), green_val (test)
- green_simple / green_hard 已移除（本质是合成 plain_plate，与部署路径不匹配）
- 当前实验目录内已保存：`supervisor_eval_fixed_20260509.json`、`iter8000_sweetspot_eval.json`、`supervisor_eval_sweetspot_20260509.json`
- MODELS 字典包含 12 个模型，v5_sweet_spot 为最后条目
- 已备份 5 份：`bak_supervisor_20260509_172830`, `bak_supervisor_v5final_20260509_181741`,
  `bak_supervisor_v5run_20260509_184737`, `bak_supervisor_diag_20260509_193527`,
  `bak_supervisor_sweetspot_20260509_195508`

---

## 绿牌 Q2 扩充版域差距 + CCPD2020 分布审计（2026-05-10）

### Q2 扩充：池化合成 vs 池化真实

原始 Q2 只用 pose_v3_val (310 行, 皖=6) 做合成对比，皖样本太少不可信。
扩充到**池化合成 (base_cvr + v4_cvr + pose_v3 = 2,755 行)** 对比**池化真实 (green_val + CCPD2020 test = 5,839 行)**。

**池化汇总表**（关键）：

| 模型 | 组 | 真实 acc (n) | 合成 acc (n) | gap (real - synth) |
|:-|:-:|:-:|:-:|:-:|
| old_green | 皖 | **71.4%** (5637) | 24.4% (86) | **+47.0pp** |
| old_green | 非皖 | 55.0% (202) | 23.3% (2669) | +31.7pp |
| v4b_clean | 皖 | 45.3% (5637) | 51.2% (86) | -5.9pp |
| v4b_clean | 非皖 | **60.9%** (202) | 45.7% (2669) | +15.1pp |
| v5_sweet_spot | 皖 | 48.6% (5637) | **53.5%** (86) | -4.8pp |
| v5_sweet_spot | 非皖 | 37.1% (202) | **46.0%** (2669) | **-8.9pp** |

### 三模型角色定位（每人一个维度 best）

| 维度 | 最佳模型 | accuracy |
|:-|:-:|:-:|
| **真实皖**（干净场景）| **old_green** | **71.4%** |
| **真实非皖**（中国各省）| **v4b_clean** | **60.9%** |
| 合成皖 | v5_sweet_spot | 53.5% |
| 合成非皖 | v5_sweet_spot | 46.0% |
| 合成倾斜（v4_cvr_tilt）| v5_sweet_spot | 28.5% |
| 合成光照（v4_cvr_db）| v5_sweet_spot | 49.2% |
| 合成极端（v4_cvr_challenge）| v5_sweet_spot | 40.2% |

**关键发现：没有单一模型全面领先**。这是一个帕累托前沿。

### v5 过拟合合成域的证据

v5_sweet_spot 非皖 gap = -8.9pp（合成显著高于真实）。这不是"无域差距"，是**v5 过拟合合成分布**。

**一个强解释**：v5 训练见过的**真实非皖字符样本**极少：
- v5 训练真实非皖：~196 行（CCPD2020 train 的 4% 非皖）
- v4b 训练真实非皖：估算 **~14.9K 行**（来自蓝牌 / 特殊牌等真实流的额外非皖字符曝光）
- **差距 76×**，可以解释 v4b 真实非皖 60.9% vs v5 37.1% 的 23.8pp 差距，但不是唯一可能原因

v4b 的"污染流"（36K 非绿牌真实数据）很可能提供了真实摄影物理 + 非皖字符曝光；这是 v5 清理污染后丢失的一个关键信号解释。

### CCPD2020 分布审计（关键事实）

**水平倾斜 `|tilt_h - 90°|` 分布**：

| 数据集 | median | p90 | max | **>10° 倾斜比例** |
|:-|:-:|:-:|:-:|:-:|
| CCPD2020 green/**train** | 1° | 2° | 6° | **0.0%** ← 最干净 |
| CCPD2020 green/**val** | 4° | 10° | 89° | **10.0%** |
| CCPD2020 green/**test** | 1° | 7° | 88° | 4.9% |
| CCPD2019 base | 1° | 8° | 32° | 6.4% |
| **CCPD2019 tilt** | 69° | 75° | 76° | **100%** |
| **CCPD2019 challenge** | 87° | 90° | 90° | **100%** |
| **CCPD2019 db** | 87° | 90° | 90° | **100%** |

**含义**：
- CCPD2020 **train 最干净**（max 6° 倾斜）→ **训练数据没教模型处理倾斜**；这为 old_green 的部署短板提供了一个强解释
- CCPD2020 val/test 有 5-10% 极端样本，但绝大多数 (90%) 仍是干净场景
- **green_val 对鲁棒性只有弱测试力**（5-10% 样本）
- **v4_cvr_val 的 tilt/db/challenge 三个子集是 100% 极端场景**，是当前最强的鲁棒性 proxy，但仍不是实拍极端场景验证

### 鲁棒性结论（真实极端场景能力估算）

v5_sweet_spot 在**合成困难场景**上达到 v4b 或更好：
- tilt: 28.5%（old_green 只有 7.58%，**v5 领先 3.8x**）
- db: 49.2%（old_green 17.21%，领先 2.9x）
- challenge: 40.2%（old_green 10.41%，领先 3.9x）

但"合成 plate → 真实 plate" 的 -8.9pp 非皖 gap 意味着：
- v5 真实倾斜绿牌推断：~20% accuracy
- v5 真实光照绿牌推断：~40%
- v5 真实极端绿牌推断：~31%

**这是推断，缺真实极端绿牌 val 验证**。

### 基准数据全景（2026-05-10 定案）

所有模型 × 所有 val 流的最终表（完整 JSON 在 `experiments/green_ccpd2019_v5_final_20260509/supervisor_eval_sweetspot_20260509.json`）：

| Model | cvr_v2 | base_cvr | v4_cvr | pose_v3 | gv_all | gv_wan | gv_nwan | nw_p1st |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| old_green | 14.5% | 46.6% | 11.8% | 46.1% | **70.8%** | **71.8%** | 45.2% | 80.6% |
| v4b_clean | 41.3% | 71.7% | 37.2% | **47.4%** | 44.3% | 44.1% | **48.4%** | **83.9%** |
| v5_final (ep 4) | **43.4%** | **72.7%** | **40.1%** | 37.1% | 60.6% | 61.7% | 32.3% | 71.0% |
| **v5_sweet_spot** | 42.7% | 71.2% | 39.4% | 38.4% | 60.5% | 61.3% | 38.7% | 80.6% |

### Q3: v4_cvreplace 按源 BG 难度拆分（spread 大）

v4_cvreplace val 按源图 subdirectory 拆分三个子集，**对 v5_sweet_spot 表现**：

| 子集 | 源 BG 性质 | v5 accuracy |
|:-|:-|:-:|
| v4_cvr_db | CCPD2019 db（极端光照）| 49.2%（最易）|
| v4_cvr_challenge | CCPD2019 challenge（综合极端）| 40.2% |
| v4_cvr_tilt | CCPD2019 tilt（极端倾斜）| **28.5%**（最难）|
| **spread** | | **20.7pp** |

**倾斜比光照更难识别**，challenge 居中。spread 20.7pp 证明**场景本身对识别能力影响巨大**。

### 关键 JSON 产物

- `experiments/green_ccpd2019_v5_final_20260509/diag_Q123_per_prov.json` — 原始 Q1/Q2/Q3 按省份分解
- `experiments/green_ccpd2019_v5_final_20260509/diag_Q2_expanded.json` — **扩充版池化对比（权威版本）**
- `experiments/green_ccpd2019_v5_final_20260509/supervisor_eval_sweetspot_20260509.json` — 12 模型 × 5 val 完整评估
- `experiments/green_ccpd2019_v5_final_20260509/ckpt_diagnosis_full.json` — 15 个 checkpoint 的 wan/nw 轨迹

### 战略定位（基于诊断，2026-05-10）

**没有单一冠军模型**。三个模型各司其职：

| 部署场景 | 推荐模型 | 理由 |
|:-|:-:|:-|
| 皖域 / 长三角干净场景 | **old_green** | 真实皖 71.4% 最高，但倾斜崩 |
| 全国均衡（非皖多）| **v4b_clean** | 真实非皖 60.9% 最高，v5 只 37.1% |
| 极端场景鲁棒性（倾斜/光照/综合）| **v5_sweet_spot** | 在当前合成鲁棒 proxy 上最高，真实板端仍需验证 |
| 混合部署 | v4b_clean 或 v5_sweet_spot | 取决于实际场景中极端样本占比 |

### 识别的根本瓶颈（修正版）

| # | 瓶颈 | 现状 | 可破性 |
|:-:|:-|:-|:-:|
| 1 | **真实极端绿牌 val 缺失** | 现有所有真实绿牌都来自 CCPD2020（95% 干净）| 🔴 只能板端实测 |
| 2 | **非皖真实字符曝光量不足** | v5 只 196 行，v4b 经蓝牌污染有 15K 行 | 🟡 路径 Y（multihead）可解 |
| 3 | **合成→真实的方法签名 gap** | 全部生成流用同一 LAB+pose_quad 方法 | 🟡 多样化合成方法可缓解 |
| 4 | **真实绿牌天生 96% 皖** | CCPD2020 自身的地域偏见 | 🔴 需外部数据源 |

### 用户反馈的关键约束

> old_green 部署对倾斜/光照鲁棒性很糟糕，引入生成数据的目的就是补足鲁棒性，但看起来还是有很大域差距

这个反馈意味着：
1. **部署指标不是 green_val accuracy**，是"真实困难场景识别率"
2. v4_cvr tilt/db/challenge 是当前唯一的鲁棒性 proxy
3. **合成→真实 gap 是否影响部署**，只能板端测试确认

---

## 绿牌 CCPD2020 数据集结构（关键事实）

审计 `datasets/CCPD2020/` 后的确认：

| 目录 | 图片数 | 性质 |
|:-|---:|:-|
| `ccpd_green/train/` | 5,769 | 真实绿牌（当前训练在用） |
| `ccpd_green/val/` | 1,001（manifest 833）| 真实绿牌（eval proxy）|
| `ccpd_green/test/` | 5,006 | 真实绿牌（未用于训练）|

**关键特性**：
- 没有按难度分级（无 blur/tilt/challenge 子分类）— train/val/test 只是随机 split
- 全部 ~96% 皖牌（CCPD2020 数据集天然偏皖）
- 文件名自带 bbox/quad/label，可直接解析

**未充分利用**：CCPD2020 test 5,006 行若加入训练可使真实绿牌翻倍（5,769 → 10,775），但不改变 96% 皖的省份分布。

---

## 三模型对比总表（蓝牌）

### Pose-Quad 测试集（tilt + db + challenge, n=3692）

| 测试集 | 旧蓝牌专家 | v1 Posquad | v2 Hardmine | Δ v2-v1 |
|--------|:----------:|:-----------:|:-----------:|:-------:|
| **Overall** | **53.0%** | **59.5%** | **60.6%** | **+1.1pp** |
| ccpd_tilt | 57.9% | 63.1% | **65.5%** | **+2.4pp** |
| ccpd_db | 46.9% | 55.5% | **56.0%** | **+0.5pp** |
| ccpd_challenge | 53.4% | 59.2% | **60.4%** | **+1.2pp** |
| Province 1st char | 91.1% | 92.9% | **93.2%** | +0.3pp |
| Char accuracy | 87.4% | 89.7% | **90.0%** | +0.3pp |
| Short pred rate | 11.3% | 10.2% | **8.9%** | -1.3pp |

### 普通蓝牌退化检查

| 测试集 | 旧蓝牌专家 | v1 Posquad | v2 Hardmine | Δ v2-v1 |
|--------|:----------:|:-----------:|:-----------:|:-------:|
| blue_simple (n=1997) | 77.0% | 92.9% | **95.5%** | **+2.6pp** |
| val_ccpd2019_blue (n=1144) | 77.2% | 93.1% | **95.9%** | **+2.8pp** |
| blue_hard 20K | 60.7% | 63.9% | **64.8%** | **+0.9pp** |

**结论**: 蓝牌 posquad pipeline 已验证有效，推荐 RKNN 导出。

---

## Rebased Manifest 使用方式

### 训练时必须带 --dataset_root

```bash
cd /home/wzzz/LPRNet

# 普通 rebased manifest（大多数）
python src/training/train_LPRNet.py \
  --dataset_root /home/wzzz/LPRNet \
  --train_manifest manifests_rebased/xxx_train.csv \
  ...

# green_edgefit 特殊 case
python src/training/train_LPRNet.py \
  --dataset_root /home/wzzz/LPRNet/datasets \
  --train_manifest manifests_rebased/unified_manifest_green_edgefit_v3_allprov.csv \
  ...
```

蓝牌/绿牌 posquad manifest 使用 `--dataset_root /home/wzzz/LPRNet`。

### 旧 manifest 向后兼容

旧 manifest（`manifests/` 下）仍可直接使用，不传 `--dataset_root` 时
使用 CWD。但新实验应优先使用 `manifests_rebased/`。

---

## 不能删除的文件和目录

- `datasets/` — 训练依赖
- `experiments/` — 历史实验和当前专家引用
- `manifests/` — legacy reference
- `manifests_rebased/` — 当前训练主 manifest
- `model_zoo/` — 专家包索引
- `src/` — 训练代码
- `keys/` — 字符集文件
- 所有 `*.pth`、`*.pt`、`*.onnx`、`*.rknn` — 权重和板端产物
- `artifacts/` — 板端转换产物和评估报告
- **蓝牌**: `experiments/blue_ccpd2019_*/`, `manifests_rebased/blue_ccpd2019_*/`
- **绿牌**: 以下实验目录和 manifest 目录均受保护

### 绿牌关键路径保护列表

| 类型 | 路径 |
|------|------|
| Data | `datasets/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/` |
| Data | `datasets/green_ccpd2019_tilt_db_challenge_cvreplace_v4_20260508/` |
| Data | `datasets/green_ccpd2019_base_cvreplace_posquad_v1_20260509/` (v5 base 源) |
| Data | `datasets/ccpd2019_base_posquads_20260509/` (199,996 张 pose 推理 jsonl) |
| Data | `datasets/CCPD2020/ccpd_green/` (真实绿牌 train/val/test) |
| Data | `datasets/ccpd2020_replace_pose_v3/` (v5 pose_v3 源) |
| Manifest v2 | `manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/` |
| Manifest v3 | `manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v3_20260508/` |
| Manifest v4 | `manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4_20260508/` |
| Manifest v4b | `manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4b_20260509/` |
| Manifest v4c | `manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4c_20260509/` (探针，未完全训练) |
| Manifest base_cvr | `manifests_rebased/green_ccpd2019_base_cvreplace_posquad_v1_20260509/` (v5 base 流) |
| Manifest pose_v3 | `manifests_rebased/ccpd2020_replace_pose_v3/` (v5 pose 流) |
| Manifest CCPD2020 real | `manifests_rebased/ccpd2020_green_real_20260509/train_ccpd2020_green_real.csv` (v5 新建 5,769 行) |
| Manifest v5 合并 | `manifests_rebased/green_v5_final_20260509/train_v5_final.csv` (161,677 行) |
| Manifest stage1/stage2 audit | `manifests_rebased/green_backbone_mix_realprimary_20260510/` |
| Manifest A ablation | `manifests_rebased/a_ablation_20260510/` |
| Manifest A ratio sweep | `manifests_rebased/a_ratio_sweep_20260510/` |
| Manifest B-lite probe | `manifests_rebased/b_lite_probe_20260510/` |
| Exp v3 | `experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_20260508/` |
| Exp v2 | `experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/` |
| Exp phase2 | `experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_phase2_20260508/` |
| Exp expE | `experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_expE_20260508/` |
| Exp focal | `experiments/green_ccpd2019_cvr_antishort_focalctc_20260508/` |
| Exp v4 | `experiments/green_ccpd2019_cvr_v4_from_oldgreen_20260508/` |
| Exp v4b | `experiments/green_ccpd2019_cvr_v4b_continue_20260509/` |
| Exp v4b_clean | `experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v4b_clean_20260509/` |
| Exp v4c A (probe, 3 ep) | `experiments/green_ccpd2019_cvr_v4c_probe_A_wan15_20260509/` |
| Exp v4c B (probe, 3 ep) | `experiments/green_ccpd2019_cvr_v4c_probe_B_wan25_20260509/` |
| **Exp v5_final (含 sweet_spot)** | `experiments/green_ccpd2019_v5_final_20260509/` |
| Exp A branch | `experiments/green_backbone_mix_branchA_refine_only_20260510/` |
| Exp B stage1 | `experiments/green_backbone_mix_branchB_stage1_backbone_mix_20260510/` |
| Exp A ablation real_only | `experiments/a_ablation_real_only_20260510/` |
| Exp A ablation replace_only | `experiments/a_ablation_replace_only_20260510/` |
| Exp A ablation mix_rebuild | `experiments/a_ablation_mix_rebuild_20260510/` |
| Exp A ratio R10/R20/R35/R50 | `experiments/a_ratio_*_20260510/` |
| Exp B' stage1 | `experiments/b_prime_stage1_real_multidomain_20260510/` |
| Exp B-lite balanced probe | `experiments/b_lite_balanced_probe_20260510/` |
| Audit hub | `experiments/mix_source_audit_20260510/` |
| Diagnosis | `experiments/green_cvr_plateau_diagnosis_20260508/` |
| Audit | `experiments/green_cvr_v4_data_coverage_audit_20260508/` |
| Audit | `experiments/green_ccpd2019_eval_protocol_audit_20260508/` |

---

## 后续 Backlog

按优先级排列：

### P0: 当前冻结主线 — R50 主 OCR + 可选 Province-Split

目标：
- **保留 `R50` 作为后 7 位 / tilt 专家**
- **单独建模首字 / 省份位**，但当前仅作为可选增强 / 诊断项
- 避免再为了修首字去破坏 `R50` 已经拿到的倾斜能力

当前状态：
- 2026-05-14 用户已确认当前绿牌板端效果满意，绿牌训练和路线探索进入冻结维护状态
- `R50` 主 OCR 已导出：`artifacts/r50_green_ocr/R50_green_multihead_no_rknnpre_rk3568_fp16.rknn`
- `94x24` tiny 省份分类器路线已证伪
- `224x72 + gray3 + ResNet18` 的 quad-warp fullplate 路线已跑通
- 公平审计当前最强候选：`experiments/routeA_nextstage_20260512/G0_baseline_repro/best.pt`
- 但新板端 `fc224` dump 审计显示：固定 `224x72` warp 本身还不足以统一解决新增真实样本
- 首字 sidecar 当前不再作为必须闭环项；只有在真实失败样本明确指向首字问题时再启用/复测

若未来重新打开绿牌训练，启动前必须回答的核心问题：
1. 省份分类器是否与 `R50` 使用同一 OCR crop / 同一前处理
2. 省份分类数据源是否足够多省均衡
3. 最终推理如何与 `R50` 拼接，而不引入新的长度/字符副作用
4. checkpoint 选择到底按 `dump2`、按 `dump`，还是按 `province_stress`；当前三者最优 epoch 不一致
5. ARM 端固定 `224x72` 输出接入后，真实大样本 board dump 是否仍复现当前两组小样本上的冲突

### P1: 当前冻结项（不要继续烧）

1. **A ratio sweep 继续扩点**
   - 当前状态：**冻结**
   - 原因：`R50` 已过当前 sweet-spot 门槛；继续只扫比例，对首字问题帮助有限

2. **旧 B / B' / B-lite**
   - 当前状态：**封存**
   - 原因：
     - 旧 B：board-dead
     - B' Stage 1：green8 collapse
     - B-lite：province 无改善且 tilt 明显退化

3. **任何隐式“normal7 -> green8 首字迁移”训练**
   - 当前状态：**暂停**
   - 原因：当前代码路径下没有成功信号，且多次尝试都以伤 green8 主体能力告终

### P2: 与当前绿牌主线弱相关的待办

4. **蓝牌 posquad v2 → RKNN 导出 + 板端验证**
   - 当前 best: `experiments/blue_ccpd2019_posquad_v2_hardmine_20260508/best_LPRNet_model.pth`
   - 用 `rknn_env` conda 环境导出
   - 板端 `--plate-detector-type yolov8_pose_rknn`

5. **黄牌 Phase 1 板端验证**
   - 当前离线候选：`experiments/yellow_single_v2_weighted_phase2/best_LPRNet_model.pth`
   - 已有 artifact：`artifacts/yellow_LPRNet_v5_phase2_fp16.rknn`
   - 待确认 artifact 是否与推荐 checkpoint 对齐
   - 板端需接入黄色车牌颜色分流，并采集真实普通单排黄牌样本验证
   - 挂车牌、学牌、双层黄牌、小摩托黄牌不纳入 Phase 1

6. **特殊牌专家 smoke test**
   - 当前状态：非 Yellow Phase 1 主线；仅在重新打开稀有特殊牌方向时执行
   - 跑 100-step: `bash configs/rebased_experiments/special_yellow_v5_rebased_smoke.sh`

### P3: 清理类事项（非当前重点）

7. **manifest cleanup**
   - cleanup_candidate 清单在 `manifest_retention_plan.json`
   - 可移动到 `tmp/cleanup_candidates/`，不删除

8. **dataset cleanup**
   - cleanup_candidate 清单在 `dataset_retention_plan.json`
   - 约 23 个小型 QA/probe 数据集
   - **当前不建议动**

9. **完整专家包交付**
   - 需要打包时执行 build_expert_archives.py 复制 weight

### P3: 可做可不做

10. **实验目录归档**（87 个 archived_candidate，需 catalog review 后）
11. **green_edgefit 根目录软链接**（不需要，dataset_root 已解决）

---

## 主要脚本和工具路径

| 用途 | 路径 |
|------|------|
| 专家归档构建 | `tools/build_expert_archives.py` |
| 根目录整理 | `tools/cleanup/tidy_workspace_root.py` |
| Manifest rebase | `tools/rebase_manifest_paths.py` |
| Manifest 审计 | `tools/audit_manifest_paths.py` |
| 实验扫描 | `tools/scan_experiments.py` |
| 工作区迁移 | `tools/migrate_workspace.py` |
| **Pose 推理** | `scripts/step1_pose_inference_ccpd2019.py` |
| **蓝牌 Manifest 构建** | `scripts/step2_build_ccpd2019_posquad_manifest.py` |
| **蓝牌 QA** | `scripts/step3_ccpd2019_posquad_qa.py` |
| **纯评估** | `scripts/step4_eval_pure.py` / `scripts/green_ccpd2019_final_eval.py` |
| **蓝牌诊断** | `scripts/step5_diagnose_posquad_v1.py` |
| **绿牌 CV-Replace 生成 v2** | `scripts/green_ccpd2019_cvreplace_step1_generate_v2.py` |
| **绿牌 CV-Replace 生成 v4 (unique source)** | `scripts/green_ccpd2019_cvreplace_step1_generate_v4.py` |
| **绿牌 QA v2** | `scripts/green_ccpd2019_cvreplace_step2_qa_v2.py` |
| **绿牌 Manifest 构建 v2** | `scripts/green_ccpd2019_cvreplace_step3_build_manifest_v2.py` |
| **绿牌 Manifest 构建 v3** | `scripts/green_ccpd2019_cvreplace_step3_build_manifest_v3.py` |
| **绿牌 Manifest 构建 v4** | `scripts/green_ccpd2019_cvreplace_step3_build_manifest_v4.py` |
| **绿牌 Manifest 构建 v4b** | `scripts/green_ccpd2019_cvreplace_step3_build_manifest_v4b.py` |
| **绿牌 Manifest 构建 v4c** | `scripts/green_ccpd2019_cvreplace_step3_build_manifest_v4c.py` |
| **绿牌 CCPD2019 base pose 推理** | `scripts/green_ccpd2019_base_step0_pose_inference.py` |
| **绿牌 CCPD2019 base cvreplace 生成** | `scripts/green_ccpd2019_base_step1_generate.py` |
| **expD→expE 权重迁移** | `scripts/convert_green_expD_to_expE_warmstart.py` |
| **绿牌平顶诊断** | `scripts/green_cvr_plateau_diagnosis.py` |
| **数据覆盖率审计** | `scripts/green_cvr_v4_data_coverage_audit.py` |
| **评估协议审计** | `scripts/green_ccpd2019_eval_protocol_audit.py` |
| **训练脚本: v4** | `scripts/train/run_green_v4_from_oldgreen.sh` |
| **绿牌 final eval (5-stream)** | `scripts/green_ccpd2019_final_eval.py` (MODELS 含 12 模型，MANIFESTS 用 (path, split) 元组，5 条纯绿牌 val) |
| **canonical board scan（单模型/单分支）** | `scripts/board_scan_ablation_reference.py` |
| **canonical board scan（checkpoint 扫描）** | `scripts/board_scan_checkpoint_reference.py` |
| **临时域差距诊断脚本** | `/tmp/domain_gap_diag.py`、`/tmp/q2_expanded_diag.py`（2026-05-10 执行时使用；未纳入 repo，不应视为长期资产） |

---

## 回滚

- **Root tidy 回滚**: `root_tidy_rollback_plan.json`
- **Manifest rebase**: 旧 manifest 全部未改，在 `manifests/` 下
- **实验切换**: 输出到 `experiments/rebased_validation/`，旧实验未碰
- **蓝牌 posquad**: 旧蓝牌权重 `experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth` 未动
- **绿牌实验**: 所有实验目录各自独立，v4b_clean / v5_sweet_spot 均可独立回退到 old_green baseline
- **绿牌 v5 eval 脚本**: 已备份 5 份 `scripts/green_ccpd2019_final_eval.py.bak_supervisor_`（详见 v5 产出节），可任意回退历史版本
- **绿牌 v5 sweet spot checkpoint**: 仅存一份拷贝 `experiments/green_ccpd2019_v5_final_20260509/v5_sweet_spot_iter8000_LPRNet_model.pth`（从训练中间状态提取的训练权重，非训练器自动保存产物；训练器自动保存的 best/last/Final 权重仍在同级目录下）
