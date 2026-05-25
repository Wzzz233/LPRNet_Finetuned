# LPRNet 工作区目录结构

本文档说明整理后的工作区目录结构。

## 目录概览

```
LPRNet/
├── datasets/          # 数据集目录
├── src/               # 源代码目录
├── experiments/       # 实验结果目录
├── manifests/         # 数据清单目录
├── models/            # 模型权重目录
├── scripts/           # 脚本目录
├── reports/           # 报告目录
├── docs/              # 文档目录
├── artifacts/         # 构建产物
├── runs/              # 训练运行日志
├── tests/             # 测试代码
└── [根目录文件]        # 配置文件和文档
```

## 详细说明

### 📁 datasets/ - 数据集
存放所有训练和测试数据集：
- **CCPD2019/**, **CCPD2020/** - CCPD 数据集
- **CRPD_all/**, **CRPD_raw_ccpd_board_v1/** - CRPD 数据集
- **git_plate/** - GitHub 车牌数据集
- **green_exact_quad_synthetic_v1/** - 绿牌合成数据
- **green_edgefit_*/** - EdgeFit 绿牌数据集
- **CBLPRD-330k_v1/** - CBLPRD 数据集
- **suhu_*/** - 苏沪车牌数据集
- **targeted_green_missing_18/** - 目标绿牌补充数据
- **qa_samples/** - QA 样本目录
- **external_detectors/** - 外部检测器模型

### 📁 src/ - 源代码
代码按功能分类：

#### src/training/
训练相关代码：
- `train_LPRNet.py` - 主训练脚本

#### src/evaluation/
评估相关代码：
- `test_LPRNet.py` - 测试脚本
- `eval_*.py` - 各种评估脚本
- `evaluate_*.py` - 评估工具
- `compare_*.py` - 对比分析

#### src/export/
模型导出相关：
- `export_onnx.py` - ONNX 导出
- `export_onnx_rknn*.py` - RKNN 兼容导出
- `convert.py` - 格式转换

#### src/manifest/
数据清单构建：
- `build_*.py` - 各种 manifest 构建脚本

#### src/analysis/
分析和诊断：
- `analyze_*.py` - 数据分析
- `diag_*.py` - 诊断脚本

#### src/utils/
工具脚本（57个工具脚本）：
- `prepare_*.py` - 数据准备
- `auto_label_*.py` - 自动标注
- `generate_*.py` - 数据生成
- `sample_*.py` - 采样工具
- `check_*.py`, `verify_*.py` - 验证工具
- `mine_*.py` - 困难样本挖掘
- `infer_*.py` - 推理工具

#### src/ 根目录
核心模型定义：
- `LPRNet.py` - LPRNet 模型定义
- `LPRNet_multihead.py` - 多任务头版本
- `load_data.py` - 数据加载器

### 📁 experiments/ - 实验结果
存放所有实验的训练结果（94个实验）：
- `green_h*/` - 绿牌实验 H20-H36
- `green_balance_*/` - 均衡实验
- `green_multihead_*/` - 多任务头实验
- `tilt_*/` - 倾斜增强实验
- `crop_aligned_*/` - 对齐裁剪实验
- `first_board_*/` - 板端基线实验

### 📁 manifests/ - 数据清单
存放所有 manifest 文件（51个）：
- `unified_manifest_v*.csv` - 统一清单
- `unified_manifest_green_*.csv` - 绿牌清单
- `green_balance_*.csv` - 均衡清单
- `cblprd_cv_geom_manifest.csv` - CBLPRD 清单
- `*.summary.json` - 清单统计信息

### 📁 models/ - 模型权重
- **weights/** - 训练好的权重
  - `official/` - 官方权重
  - `red_stage3/` - 红牌 stage3 权重
  - `core/` - 核心模型定义（原 model/）
  - `baselines/` - 基线权重
- **detectors/** - 检测器模型
  - `yolo26n.pt` - YOLO 检测器
  - `yolov8n_obb_trained_70epoch/` - YOLOv8 OBB

### 📁 scripts/ - 脚本
- **train/** - 训练脚本（62个）
  - `run_green_*.sh` - 绿牌实验脚本
  - `run_h*.sh` - H系列实验
  - `run_official_*.sh` - 官方实验
- `export_first_board_rknn.sh` - 导出脚本
- `run_nonccpd_obb_pipeline.py` - OBB 流水线

### 📁 reports/ - 报告
实验报告和总结文档

### 📁 docs/ - 文档
项目文档

### 📁 tmp_scripts/ - 临时脚本
临时调试和测试脚本

## 根目录保留文件

### 文档
- `README.md` - 项目说明
- `WORKSPACE_STRUCTURE.md` - 本文件
- `LPR_EXPERIMENT_SUMMARY_*.md` - 实验总结
- `*_REPORT.md`, `*_NOTE.md` - 各种报告和笔记
- `RK3568_BOARD_BASELINE_REPORT.md` - 板端基线报告
- `PROJECT_RULES_STRICT.md` - 项目规则

### 配置
- `.gitignore` - Git 忽略规则
- `requirements-train.txt` - 训练依赖
- `environment.train.yml` - Conda 环境
- `USE_ENV.sh`, `USE_ENV.ps1` - 环境脚本

### 其他
- `check*.onnx` - ONNX 检查点
- `raw.ppm`, `ocrin_*.ppm` - 测试图像
- `board_anchor_labels.txt` - 锚点标签
- `运行日志.txt` - 运行日志

## 使用说明

### 运行训练
```bash
bash scripts/train/run_green_h36a_pos0head.sh
```

### 运行评估
```bash
python src/evaluation/test_LPRNet.py --config xxx
```

### 构建 Manifest
```bash
python src/manifest/build_manifest_xxx.py
```

---

*目录结构整理时间: 2025-04-05*
