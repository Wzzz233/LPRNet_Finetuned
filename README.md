# LPRNet - 车牌识别项目

基于 LPRNet 的车牌识别项目，支持蓝牌和绿牌的多牌型统一识别。

---

## 📂 目录结构

```
LPRNet/
├── README.md              # 本文件
├── .gitignore             # Git 忽略规则
│
├── config/                # 配置文件
├── datasets/              # 数据集目录
├── docs/                  # 文档目录
├── experiments/           # 实验结果目录
├── logs/                  # 日志目录
├── manifests/             # 数据清单目录
├── models/                # 模型权重目录
├── reports/               # 实验报告目录
├── runs/                  # TensorBoard 运行日志
├── scripts/               # 脚本目录
├── src/                   # 源代码目录
├── tests/                 # 测试代码目录
├── tmp_scripts/           # 临时脚本
│
├── artifacts/             # 构建产物
└── data/                  # 数据目录（内部使用）
```

---

## 目录详解

### 🔧 config/ - 配置文件
项目配置和环境设置：

| 文件 | 说明 |
|------|------|
| `environment.train.yml` | Conda 训练环境配置 |
| `requirements-train.txt` | Python 训练依赖 |
| `board_anchor_labels.txt` | 板端锚点标签 |
| `UNIFIED_MANIFEST_SPEC.md` | 统一清单规范 |
| `USE_ENV.sh` / `USE_ENV.ps1` | 环境激活脚本 |

---

### 📊 datasets/ - 数据集
所有训练和测试数据（37 个数据集）：

#### 主要数据集
- **CCPD2019/**, **CCPD2020/** - CCPD 车牌数据集
- **CRPD_all/** - CRPD 车牌数据集
- **git_plate/** - GitHub 开源车牌数据

#### 绿牌数据
- **green_exact_quad_synthetic_v1/** - 精确四边形合成绿牌
- **green_edgefit_allprov_v1~v4/** - EdgeFit 绿牌数据（多个版本）
- **green_edgefit_tier3_*/** - 三档难度 EdgeFit 数据

#### 其他数据
- **CBLPRD-330k_v1/** - CBLPRD 大规模车牌数据
- **suhu_*/** - 苏沪车牌数据
- **targeted_green_missing_18/** - 目标省份绿牌补充数据
- **qa_samples/** - QA 质检样本
- **external_detectors/** - 外部检测器模型

---

### 📚 docs/ - 文档
项目文档按类型分类：

#### docs/reports/ - 实验报告
- `LPR_EXPERIMENT_SUMMARY_*.md` - 实验总结
- `RK3568_BOARD_BASELINE_REPORT.md` - 板端基线报告
- `DEBUG_LPRNET_FINE_TUNE.md` - 调试记录
- `BASELINE_FREEZE.md` - 冻结基线

#### docs/notes/ - 研究笔记
- `PLATE_KEYPOINT_FOLLOWUP.md` - 关键点追踪
- `PLATE_QUAD_MODEL_RESEARCH.md` - 四边形模型研究
- `GEOM_QA_RULES_AFTER_AUDIT.md` - 几何 QA 规则
- `MANIFEST_V*_NOTE.md` - Manifest 版本说明
- `NONCCPD_OBB_AUTOLABEL_STATUS.md` - 非 CCPD 标注状态

#### docs/plans/ - 计划和规范
- `PROJECT_RULES_STRICT.md` - 项目严格规则
- `NEXT_STEP_PLAN.md` - 下一步计划
- `ROUND1_GREEN_CONSERVATIVE_PLAN.md` - 绿牌保守策略
- `WINDOWS_TRAINING.md` - Windows 训练指南

---

### 🧪 experiments/ - 实验结果
94 个实验目录，命名规范：`{实验类型}_{实验代号}/`

#### 主要实验系列
- **green_h20 ~ green_h36/** - 绿牌实验系列
- **green_balance_*/** - 数据均衡实验
- **green_multihead_*/** - 多任务头实验
- **tilt_*/** - 倾斜增强实验
- **crop_aligned_*/** - 裁剪对齐实验
- **first_board_*/** - 板端基线实验

每个实验目录包含：
- `weights_*/` - 训练权重
- `logs/` - 训练日志
- `checkpoints/` - 检查点
- `config.yaml` - 实验配置

---

### 📋 manifests/ - 数据清单
51 个 manifest CSV 文件：

| 文件名模式 | 说明 |
|-----------|------|
| `unified_manifest_v*.csv` | 统一清单 |
| `unified_manifest_green_*.csv` | 绿牌清单 |
| `unified_manifest_v4_*.csv` | V4 版本清单 |
| `*.summary.json` | 清单统计信息 |
| `cblprd_cv_geom_manifest.csv` | CBLPRD 清单 |

---

### 🏋️ models/ - 模型权重

#### models/weights/ - 训练权重
- `official/` - 官方发布权重
- `red_stage3/` - 红牌 stage3 权重
- `core/` - 核心模型定义
- `baselines/` - 基线权重

#### models/detectors/ - 检测器模型
- `yolo26n.pt` - YOLOv5 检测器
- `yolov8n_obb_trained_70epoch/` - YOLOv8 OBB 检测器

#### models/checkpoints/ - ONNX 检查点
- `check0_base_optimize.onnx`
- `check1_fold_constant.onnx`
- `check2_correct_ops.onnx`
- `check3_fuse_ops.onnx`

---

### 📝 logs/ - 日志文件
- `运行日志.txt` - 主运行日志

---

### 📰 reports/ - 实验报告
详细实验报告（Markdown 格式）：
- `GREEN_*.md` - 绿牌实验报告
- `H*_EXPERIMENT_REPORT.md` - 实验 H 系列报告
- `WORKSPACE_LAYOUT.md` - 工作区布局

---

### 📈 runs/ - TensorBoard 日志
TensorBoard 可视化日志目录

---

### 🚀 scripts/ - 脚本

#### scripts/train/ - 训练脚本（62个）
命名规范：`run_{实验类型}_{实验代号}.sh`

```bash
# 示例
run_green_h36a_pos0head.sh
run_green_multihead_round3.sh
run_tilt_obbwarp_experiment.sh
```

#### 其他脚本
- `export_first_board_rknn.sh` - RKNN 导出
- `run_nonccpd_obb_pipeline.py` - OBB 流水线

---

### 💻 src/ - 源代码
121 个 Python 脚本，按功能分类：

#### src/ - 核心模型
| 文件 | 说明 |
|------|------|
| `LPRNet.py` | LPRNet 基础模型 |
| `LPRNet_multihead.py` | 多任务头模型 |
| `load_data.py` | 数据加载器 |

#### src/training/ - 训练代码
- `train_LPRNet.py` - 主训练脚本

#### src/evaluation/ - 评估代码
- `test_LPRNet.py` - 测试脚本
- `eval_*.py` - 各种评估工具
- `evaluate_*.py` - 评估脚本
- `compare_*.py` - 对比分析

#### src/export/ - 导出代码
- `export_onnx.py` - ONNX 导出
- `export_onnx_rknn_compatible.py` - RKNN 兼容导出
- `export_onnx_rknn_multihead.py` - 多任务头导出
- `convert.py` - 格式转换
- `rewrite_onnx_outputs.py` - ONNX 输出重写

#### src/manifest/ - 清单构建（28个脚本）
- `build_manifest_*.py` - 各种 manifest 构建
- `build_green_*.py` - 绿牌清单构建
- `append_*.py` - 清单追加

#### src/analysis/ - 分析诊断
- `analyze_*.py` - 数据分析
- `diag_*.py` - 诊断脚本

#### src/utils/ - 工具脚本（57个）
| 类别 | 脚本 |
|------|------|
| 数据准备 | `prepare_*.py` |
| 自动标注 | `auto_label_*.py` |
| 数据生成 | `generate_*.py` |
| 样本采样 | `sample_*.py` |
| 验证检查 | `check_*.py`, `verify_*.py` |
| 困难样本 | `mine_*.py` |
| 推理工具 | `infer_*.py` |
| 数据合并 | `merge_*.py` |
| 报告生成 | `generate_experiment_report.py` |

---

### 🧪 tests/ - 测试代码
- `test_freeze_baseline.py` - 冻结基线测试
- `test_train_manifest_mode.py` - Manifest 模式测试
- `test_unified_manifest.py` - 统一清单测试
- `samples/` - 测试样本（raw.ppm, ocrin_*.ppm）

---

### 🔨 tmp_scripts/ - 临时脚本
11 个临时调试和测试脚本：
- `tmp_*.py` - 临时 Python 脚本
- `tmp_*.txt` - 临时数据文件

---

### 📦 artifacts/ - 构建产物
构建输出目录

---

### 📂 data/ - 数据目录
内部数据存储（原项目结构）

---

## 🚀 快速开始

### 环境配置
```bash
# 激活环境
source config/USE_ENV.sh

# 或使用 conda
conda env create -f config/environment.train.yml
```

### 运行训练
```bash
# 运行绿牌实验
bash scripts/train/run_green_h36a_pos0head.sh

# 运行多任务头实验
bash scripts/train/run_green_multihead_round3.sh
```

### 运行评估
```bash
python src/evaluation/test_LPRNet.py --config experiments/green_h36/params.yaml
```

### 构建 Manifest
```bash
python src/manifest/build_manifest_v4_board_aligned.py
```

### 导出模型
```bash
python src/export/export_onnx_rknn_multihead.py \
    --weights experiments/green_h36/weights_best.pth \
    --output models/green_h36.onnx
```

---

## 📊 项目统计

| 类别 | 数量 |
|------|------|
| 数据集 | 37 个 |
| 实验 | 94 个 |
| Manifest | 51 个 |
| Python 脚本 | 121 个 |
| 训练脚本 | 62 个 |
| 文档 | 24 个 |

---

## 📝 命名规范

### 实验命名
```
{类型}_{代号}_{描述}/

例如：
- green_h36a_pos0head    (绿牌实验 H36A - POS0 头)
- tilt_obbwarp_v7        (倾斜实验 V7 - OBB 变形)
- first_board_baseline   (板端基线实验)
```

### Manifest 命名
```
unified_manifest_{版本}_{描述}.csv

例如：
- unified_manifest_v4_board_aligned.csv
- unified_manifest_green_balance_round2.csv
```

### 脚本命名
```
{动作}_{目标}_{描述}.py

例如：
- build_manifest_v4.py
- eval_green8_by_province.py
- generate_edgefit_tier3.py
```

---

## 🔗 相关链接

- [工作区结构文档](docs/WORKSPACE_STRUCTURE.md)
- [项目规则](docs/plans/PROJECT_RULES_STRICT.md)
- [实验总结](docs/reports/LPR_EXPERIMENT_SUMMARY_20260326_20260330.md)

---

*整理时间: 2025-04-05*
