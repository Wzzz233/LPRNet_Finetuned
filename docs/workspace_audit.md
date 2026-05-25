# LPRNet 工作区审计报告

> 生成时间: 2026-05-07
> 审计工具: tools/audit_manifest_paths.py, tools/scan_experiments.py

---

## 1. 当前目录结构概览

```
LPRNet/                          # 根目录
├── src/                         # 源码 (2.1M)
├── datasets/                    # 数据集 (61G, 182万文件)
├── manifests/                   # manifest 文件 (14G, 440+文件)
├── experiments/                 # 历史实验 (6.2G, 359子目录)
├── runs/                        # 训练 run 记录 (4.0G)
├── generated/                   # 生成数据 (3.3G)
├── tmp/                         # 临时文件 (2.6G)
├── archive/                     # 归档 (1.2G)
├── artifacts/                   # 构建产物 (623M)
├── models/                      # 模型定义/权重 (451M)
├── reports/                     # 报告 (168M)
├── labels/                      # 标签数据 (83M)
├── scripts/                     # 脚本 (2.7M)
├── config/                      # 配置文件 (28K)
├── configs/                     # 附加配置 (8K)
├── keys/                        # OCR keys (12K)
├── docs/                        # 文档/审计报告 (新建)
├── tools/                       # 审计工具 (新建)
├── logs/                        # 日志 (24K)
├── tests/                       # 测试 (2.8M)
├── data/                        # 数据 (24K)
├── third_party/                 # 第三方库 (27M)
├── qa_*/                        # 质量检查 (多个)
├── tmp_*/                       # 临时实验 (多个)
├── green_edgefit_*              # 生成数据 (25M)
└── *.py, *.md, *.onnx           # 根目录散落文件
```

## 2. 项目主要组成部分

### 源码区域 (src/)
- `src/LPRNet.py` — LPRNet 模型定义 (3.5K)
- `src/LPRNet_multihead.py` — 多头 LPRNet (18K)
- `src/load_data.py` — 数据加载器 (37K)
- `src/training/` — 训练模块
- `src/evaluation/` — 评估模块
- `src/manifest/` — manifest 构建
- `src/export/` — 模型导出 (ONNX/RKNN)
- `src/quad_refiner/` — quad refiner 子模块
- `src/micro_rectifier/` — 微型矫正
- `src/utils/` — 工具函数
- `src/analysis/` — 分析工具

### 脚本区域 (scripts/)
- `scripts/train/` — 训练启动脚本
- `scripts/eval/` — 评估脚本
- `scripts/curriculum_gray3/` — 渐进式训练脚本
- 各种 build/manifest 生成脚本

### 数据集区域 (datasets/)
共 1,828,352 个文件，61G。包含多个数据集子目录（通过真实目录和软链接混合指向）：
- CCPD2019 (25G) — 真实车牌数据
- CCPD2020 (905M) — 真实车牌数据
- CRPD_all (19G) — 真实车牌数据
- CRPD_raw_ccpd_board_v1 (176M)
- CBLPRD-330k_v1 (2G) — 合成数据
- cblprd_obb_autolabel_v1 (100M)
- green_edgefit_* — 绿色车牌 edgefit 生成数据
- green_exact_quad_synthetic_v1 — 精确四边形合成数据

**注意**: 根目录下同时有软链接指向 datasets/ 内的子目录，形成重复入口。

### Manifest 区域 (manifests/)
共 440+ 个文件，14G。主要分支：
- `unified_manifest_*.csv` — 统一 manifest 系列（版本 v1-v4, 绿牌平衡, U1/U1B/U1C）
- `curriculum_gray3*/` — 渐进式训练的分阶段 manifest
- `ccpd2020_replace_*/` — 替换版 manifest
- `firstchar_*/` — 首字训练 manifest
- `yellow_*/` — 黄牌 manifest
- `special_*/` — 特殊牌 manifest
- `Archive/` — 归档的旧 manifest (5G)

### 历史实验区域 (experiments/)
共 359 个目录，其中 271 个被识别为疑似实验目录（177 建议保留，94 建议归档）。
大小合计 8.0G。

## 3. 源码区域分析

源码相对整洁，集中在 src/ 和 scripts/。主要文件及功能：

| 文件 | 大小 | 功能 |
|------|------|------|
| src/LPRNet.py | 3.5K | 单头 LPRNet |
| src/LPRNet_multihead.py | 18.7K | 多头 LPRNet |
| src/load_data.py | 37K | 数据加载和预处理 |
| scripts/train/ | 多层 | 训练启动脚本 |
| scripts/curriculum_gray3/ | 多层 | 渐进式训练管线 |

**问题**: 
- 部分临时脚本（tmp_*.py）散落在根目录
- `__pycache__` 目录需要清理
- scripts/ 下脚本和项目根目录下脚本有重叠

## 4. 数据集区域分析

**总大小**: 约 61G（实际数据，不含软链接重复计数）
**总文件数**: 1,828,352 个

**关键发现**:
1. 数据集通过根目录软链接 (`CCPD2019 ->, CCPD2020 ->, CRPD_all ->`) 和 datasets/ 下的真实数据双重指向
2. `datasets/` 中包含 xxx_full_v2 等非标准命名的目录（green_edgefit_tier3_full_v2 等），这其实是生成数据而非原始数据集
3. 部分数据在 `generated/`、`green_edgefit_v4_boardlike_a3000/` 等根目录级目录中
4. 数据集路径存在软链接断链风险（0 大小显示）
5. 数据集目录混合了原始数据（CCPD）和生成数据（edgefit, template synth）

**建议**:
- 重新梳理软链接，确保一致性
- 区分原始数据和生成数据
- 数据集目录暂时不动

## 5. Manifest 路径依赖分析

**核心发现: 严重路径依赖问题**

| 指标 | 值 |
|------|-----|
| 扫描文件数 | 440 |
| 采样样本行 | 1,128,482 |
| 绝对路径数 | 754,479 |
| 无效路径数 | 53,585 |
| 高风险文件 | 376 (85.5%) |
| 中风险文件 | 1 |
| 低风险文件 | 63 |

**路径类型分布**:
- 绝对路径: 754,479 — 全部绑定 `/home/wzzz/LPRNet/`
- 相对路径: 少数
- 裸文件名: 若干
- 无效路径: 53,585 — 路径指向不存在的文件

**典型绝对路径示例**:
```
/home/wzzz/LPRNet/CCPD2020/ccpd_green/train/...
/home/wzzz/LPRNet/CRPD_all/images/train/...
/home/wzzz/LPRNet/datasets/green_edgefit_allprov_v4_realistic_b/...
```

**风险等级**: 🔴 极高
- 只要工作区迁移到其他路径或机器，所有 754K 绝对路径立即失效
- 53K 无效路径表明已有部分数据被移动或删除
- 主 manifest (`unified_manifest_official_gray3_bluegreen_u1.csv`) 有 662K 行，全部绝对路径

**建议**: 
- 统一改为 `dataset_root + relative_path` 格式
- dataset_root 在训练配置中声明
- 先通过软链接兼容旧路径，再生成新 manifest

## 6. 历史实验区域分析

**总大小**: 8.0G
**实验数量**: 271 个识目录（177 建议保留, 94 建议归档）

**实验系列**:
| 系列 | 数量 | 总大小 | 说明 |
|------|------|--------|------|
| curriculum_gray3_stage* | ~30 | ~1.5G | 渐进式训练系列（stage A/B/C/D/E） |
| green_e* | ~30 | ~1G | 绿牌 E 系列实验 |
| green_h* | ~20 | ~500M | 绿牌 H 系列实验 |
| firstchar* | ~15 | ~20M | 首字实验 |
| special_yellow* | ~8 | ~70M | 黄牌/特殊牌 |
| tilt_ocr_obbwarp* | ~8 | ~500M | 倾斜实验 |
| quad_refiner* | ~10 | ~2G | 四边形优化 |
| 其他 | ~150 | ~3G | 杂项实验 |

**主要问题**:
1. 实验命名不一致，难以追踪
2. 多个实验包含大量 checkpoint（某些有 10+ 权重文件）
3. `weights`、`weights_stageA/B/C` 等命名过于泛泛
4. 备份实验和失败实验混在一起（`*_bak*`, `*_old*` 等）
5. 实验配置分散在 train_summary.json 和 training 脚本中，没有独立配置文件

**建议保留的重要实验**:
- curriculum_gray3_stageA — 渐进式训练基础
- curriculum_gray3_stageB_* — 渐进式训练阶段B系列
- green_e* — 绿牌 E 系列实验（系统化实验）
- green_h* — 绿牌 H 系列实验
- special_yellow_v* — 黄牌实验
- firstchar_batch* — 首字实验
- green_specialist_official_* — 官方绿牌专家模型

**建议归档的实验**:
- 多个 `weights`、`weights_stage*` 目录（重复的权重存放，无日志/配置）
- `*_bak*` 备份目录
- `recheck_*` 临时检查
- 实验名称包含 `_old_`、`_bak_` 的目录
- 日志为空的实验目录

## 7. 权重和日志分析

### 权重文件分布
权重文件散布在多个位置：
- `experiments/*/` — 每个实验目录中（best.pth, last.pth, Final.pth, 以及迭代检查点）
- `models/` — 模型相关文件 (451M)
- 根目录 `check0_base_optimize.onnx` — 一个基础 ONNX 权重 (44M)
- `runs/quad_refiner/` — quad refiner 的 best.pt, last.pt（130M 每个）

### 日志文件
- 主要日志在各实验目录下的 `train.log`
- `logs/` 目录只有 24K，不用于实际日志存储
- 评估结果以 `eval_val_*.json` 存储在实验目录中

**问题**: 权重没有统一管理，没有 "最佳权重归档" 机制。

## 8. 疑似重要实验列表

基于实验大小、权重数量和最佳指标，以下是最值得保留的实验：

1. **curriculum_gray3_stageA** — 渐进式训练起点 (25M, best_acc=0.54)
2. **curriculum_gray3_stageB_B2D_paradigm3_progress** — 阶段B关键实验 (75M)
3. **curriculum_gray3_stageB_v1_B1B_E6AB_preblur_v3_combined** — E6 系列 (46M)
4. **green_e1_v4_a3000** — E1 基准 (253M)
5. **green_e8c_brightness_replace** — 亮度替换实验 (20M)
6. **green_e9c_exact_template_allprov_1800** — 精确模板 (20M)
7. **special_yellow_v5** — 黄牌最优 (7M)
8. **firstchar_batch1** (840M) / **firstchar_batch2** (651M) — 首字实验
9. **green_h36** (294M) — H系列最大实验
10. **green_multihead_round1** (160M) — 多头实验

## 9. 疑似可归档内容

1. **experiments/ 中 94 个建议归档实验** (~500M) — 缺少权重/日志/配置的实验
2. **多个 `weights*` 目录** — 只有权重文件，无日志和配置
3. **`*_bak*`、`*_old*` 目录** — 明显是备份
4. **`tmp_*` 根目录目录** (2.6G) — 临时实验和输出
5. **`__pycache__` 目录** — Python 缓存
6. **`archive/`** (1.2G) — 已有归档目录但内容杂乱
7. **manifests/Archive/** (5G) — 大量旧 manifest 版本
8. **`reports/`** (168M) — 生成报告（可保留但可压缩）
9. **`generated/`** (3.3G) — 生成数据（需确定是否仍在使用）

## 10. 疑似临时文件和缓存

| 位置 | 大小 | 类型 |
|------|------|------|
| tmp/ | 2.6G | 临时实验输出 |
| tmp_refiner_*/ | ~100M | 临时 refiner 实验 |
| tmp_scripts/ | 3.0M | 临时脚本 |
| tmp_v4_probe/ | 16K | 临时探针 |
| __pycache__/ (所有) | ~2M | Python 缓存 |
| `*.Zone.Identifier` | 若干 | Windows 元数据文件 |
| 根目录 `tmp_*.py` | 若干 | 临时脚本散落根目录 |

## 11. 当前结构的主要问题

1. **🔴 绝对路径依赖**: manifest 和配置大量使用 `/home/wzzz/LPRNet/...` 绝对路径
2. **🔴 无效路径**: 53K 样本指向不存在的文件
3. **🟡 数据分散**: 数据集、生成数据、软链接形成多入口的复杂引用关系
4. **🟡 实验杂乱**: 359 个实验目录中近 100 个可归档
5. **🟡 权重分散**: 权重文件没有统一管理，每个实验目录自行存放
6. **🟡 命名不一致**: 实验名、manifest 名缺乏统一体系
7. **🟢 配置固化**: 训练参数嵌在 train_summary.json 中而非独立配置
8. **🟢 临时文件过多**: 根目录散落临时脚本和缓存目录

## 12. 最大风险点

### 🔴 风险 1: Manifest 绝对路径绑定 (极高)
如果工作区被移动到其他路径或机器：
- 所有 manifest 全部失效（754K 绝对路径）
- 训练无法启动
- 评估无法缓存
- 需要重新生成全部 manifest

### 🔴 风险 2: 数据路径一致性 (高)
- 软链接 + 真实目录双重引用
- 部分软链接断链（0 大小）
- 数据集既在 datasets/ 下，又在根目录下有软链接
- 生成数据分散在 generated/、tmp/、green_edgefit_*/

### 🟡 风险 3: 实验不可复现 (中)
- 实验配置嵌在 train_summary.json 中，而非独立配置文件
- 部分实验缺少日志和配置
- manifest 绝对路径绑定特定机器
- 数据集版本变化难以追踪

### 🟡 风险 4: 权重版本混乱 (中)
- 没有统一的最佳权重记录
- 实验目录中有 best.pth / Final.pth / last.pth 混合
- 权重和 manifest 版本无对应关系
- RKNN/ONNX 导出产物散布

## 13. 推荐的新目录结构

```
LPRNet/
├── README.md
├── PROJECT_STATUS.md
├── requirements.txt
├── src/                          # 保持不动
├── scripts/                      # 保持不动
├── configs/                      # 标准化训练配置
│   ├── base.yaml                 # 基础 LPRNet 配置
│   ├── dataset_roots.yaml        # 所有数据集根路径定义
│   └── experiments/              # 实验配置
├── data/                         # 整理后的数据引用
│   ├── symlinks/                 # 软链接统一管理
│   │   ├── CCPD2019 -> ../../datasets/CCPD2019
│   │   └── ...
│   └── README.md                 # 数据集清单
├── manifests/                    # 保持目录结构，但新增符合规范版本
│   ├── README.md                 # manifest 规范说明
│   ├── active/                   # 当前使用的 manifest
│   ├── archived/                 # 旧 manifest（从 Archive 迁移）
│   └── curriculum_*/             # 保留
├── experiments/                  # 实验目录
│   ├── active/                   # 进行中/重要实验
│   ├── archived/                 # 归档实验
│   └── README.md                 # 实验索引
├── weights/                      # 统一权重管理
│   ├── best/                     # 最佳权重
│   ├── exported/                 # ONNX/RKNN 导出
│   └── archived/                 # 旧权重
├── logs/                         # 统一日志
│   └── tensorboard/              # TensorBoard 日志
├── docs/                         # 文档
│   ├── workspace_audit.md
│   ├── experiment_index.md
│   ├── manifest_path_audit.md
│   └── dataset_notes.md
├── tools/                        # 工具脚本
│   ├── audit_manifest_paths.py
│   ├── scan_experiments.py
│   ├── migrate_workspace.py
│   └── clean_experiments.py      # 未来实现
└── tmp/                          # 临时目录（根目录 tmp 保持不动）
```

## 14. 整理优先级

### 第一阶段（当前 — 只读审计 ✅）
- [x] 生成索引文件
- [x] 审计 manifest 路径依赖
- [x] 扫描实验目录
- [x] 生成审计报告

### 第二阶段（等待确认后执行）
1. **归档实验目录** — 将 94 个建议归档的实验移到 experiments/archived/
2. **标准化 manifest** — 为关键 manifest 生成相对路径版本
3. **根目录整理** — 移动 tmp_*.py 到 tools/ 或 scripts/
4. **清理 __pycache__** — 可安全删除所有 __pycache__

### 第三阶段（低优先级）
1. **统一权重管理** — 将最佳权重集中到 weights/best/
2. **配置标准化** — 从 train_summary.json 提取到独立 yaml
3. **数据集路径统一** — 重建软链接体系
4. **完整可复现性检查** — 确保每个实验可复现

## 15. 下一步建议

1. **先阅读**: docs/manifest_path_audit.md — 了解约束最大的路径依赖问题
2. **审查实验**: docs/experiment_index.md — 确认哪些实验保留/归档
3. **确认风险**: 验证无效路径是否影响当前训练
4. **制定迁移计划**: 基于 migration_plan.json 逐项确认
5. **生成回滚方案**: 每次迁移前确保可回滚
6. **不要一步到位**: 每次整理一个维度，确认后再下一项

---

*报告由 audit_manifest_paths.py 和 scan_experiments.py 联合生成*
