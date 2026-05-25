# LPRNet + YOLO 车牌识别系统技术报告

> 作者：训练工程团队
> 最后更新：2026-05-04
> 部署平台：RK3568 ARM Linux（2GB RAM，NPU）

---

## 目录

1. [系统总览](#1-系统总览)
2. [YOLO 检测模型](#2-yolo-检测模型)
   - 2.1 [YOLOv8n-OBB 旋转框检测器](#21-yolov8n-obb-旋转框检测器)
   - 2.2 [YOLOv8n-Pose 关键点检测器](#22-yolov8n-pose-关键点检测器)
3. [LPRNet 车牌识别模型](#3-lprnet-车牌识别模型)
   - 3.1 [架构设计](#31-架构设计)
   - 3.2 [推理前处理管线](#32-推理前处理管线)
4. [训练数据体系](#4-训练数据体系)
5. [训练方法论](#5-训练方法论)
   - 5.1 [E 系列实验框架](#51-e-系列实验框架)
6. [板端部署](#6-板端部署)
   - 6.1 [最终生产配置](#61-最终生产配置)
   - 6.2 [Pose 检测器集成](#62-pose-检测器集成)
7. [关键实验历程](#7-关键实验历程)
8. [附录：名词对照](#8-附录名词对照)

---

## 1. 系统总览

车牌识别系统是车联网和智能交通的核心组件之一。本系统部署于 RK3568 嵌入式平台（2GB RAM，含 NPU），由检测器和 OCR 模型串联构成完整的端到端推理链路。

**系统架构**：

```
摄像头帧 (RGB)
    │
    ▼
┌──────────────────────────────────────────┐
│ YOLOv8n-Pose 检测器（生产部署）            │
│ 输出：车牌四角关键点 TL/TR/BR/BL           │
│         + 可见度置信度                     │
└──────────────┬───────────────────────────┘
               │ 检测到车牌 4 keypoints
               ▼
┌──────────────────────────────────────────┐
│ 透视矫正 + 预处理                         │
│ warp_quad_to_rect → letterbox resize     │
│ → 94×24 → [可选归一化]                    │
└──────────────┬───────────────────────────┘
               │ 94×24 OCR 输入图像
               ▼
┌──────────────────────────────────────────┐
│ LPRNet 多头 OCR 模型                      │
│ 蓝牌: LPRNet_stage3（7 位 normal7 head）  │
│ 绿牌: prov_deg（8 位 green8 expD head）   │
│ ↓ CTC 波束解码 → 车牌文本                  │
└──────────────────────────────────────────┘
```

**生产部署模型文件**（存放于板端 `/userdata/model/`）：

| 模型 | 文件 | 说明 |
|------|------|------|
| 车辆检测 | `yolov5s_rk3568.rknn` | 整车检测，辅助车牌定位 |
| 车牌检测 | `pose_detector.rknn` | YOLOv8n-Pose，直接输出四角 | 
| 蓝牌 OCR | `LPRNet_stage3_rk3568_fp16_more_trained.rknn` | 7 位蓝牌识别 |
| 绿牌 OCR | `prov_deg_fp16_no_rknnpre.rknn` | 省份退化训练绿牌模型 |

**生产系统识别率**：板端实际识别率较高，满足部署要求。

---

## 2. YOLO 检测模型

检测器的核心任务：**在全帧图像中找到车牌位置，输出四角坐标（quadrilateral）**。经过多轮迭代，最终生产部署采用了 **YOLOv8n-Pose** 直接输出关键点的方案，替代了 OBB + refiner 两阶段管线。

### 2.1 YOLOv8n-OBB 旋转框检测器（历史方案）

#### 地位

OBB 检测器是开发过程中的**中间方案**。它在绿牌数据上微调，单模型同时处理蓝牌和绿牌，但 OBB 本质是旋转矩形（rotated rectangle），无法表示透视畸变产生的非矩形四边形。该方案已被 Pose 检测器取代，不再用于生产。

#### 模型实现

```
输入: 640×480 → letterbox(640×640) → YOLOv8n backbone → OBB head
输出: (1, 20, 8400) — bbox(xywh) + conf + angle + cls
      → 解码为 4 个角点 (xyxyxyxy)
```

- Base: `yolov8n-obb.pt`（~3.0M 参数）
- 单类别 `item`（不区分蓝/绿牌）
- 检测率：CCPD2020 test 98.5%，置信度中位数 0.933
- 角点误差中位数：10.07px

**局限性**：对于极端倾斜的绿牌（角度 > 30°），OBB 角点误差可达 13-20px。这个误差虽然可以通过 quad refiner 改善，但增加了管线复杂度。

### 2.2 YOLOv8n-Pose 关键点检测器（生产方案）

#### 地位

Pose 检测器是生产部署的**主力车牌检测方案**。它从 YOLOv8n-pose 基座出发，在 406K 张混合来源图像上训练，**直接输出车牌四角 keypoints**，替代了 OBB + quad refiner 两阶段管线。

#### 设计动机

OBB + quad refiner 两阶段方案存在三个不足：
1. OBB 的旋转矩形假设在极端角度下精度受限
2. Refiner（ResNet18 + FPN）增加了模型加载和推理开销
3. 两阶段有门控拒绝（gate rejection），引入不确定的 fallback

Pose 检测器在同一个 YOLO 架构内直接回归四个 keypoint，**零额外管线开销**，且门控拒绝数为 0。

#### 模型实现

```
输入: 640×640 → YOLOv8n backbone → Pose decode head
输出: (1, 17, 8400) — 4 bbox + 1 cls + 12 kp (4 keypoints × 3 = x,y,visibility)
      → 解码 4 个角点 (TL, TR, BR, BL) + 可见度置信度
```

- Base: `yolov8n-pose.pt`（~3.3M 参数）
- 关键点顺序：TL(0) → TR(1) → BR(2) → BL(3)，通过几何 sum/diff 启发式强制对齐
- 水平翻转禁用（`fliplr=0.0`）

#### 训练数据集

数据集 `plate_true_quad_pose/` 包含 **406,117 张训练图像**，来源混合确保泛化：

| 数据源 | 比例 | 说明 |
|--------|:----:|------|
| CCPD2019 正常蓝牌 | 20% | `ccpd_base` |
| CCPD2019 困难蓝牌 | 15% | tilt, rotate, blur, challenge 等 |
| CCPD2020 绿牌易 | 12% | 底部 40%（按 quad score） |
| CCPD2020 绿牌中 | 14% | 中部 30% |
| CCPD2020 绿牌难 | 14% | 顶部 30%（极端倾斜聚焦） |
| CRPD 蓝牌 | 10% | 真实道路场景 |
| CRPD 黄牌 | 8% | 黄色牌照 |
| CRPD 多牌/复杂 | 7% | 多车牌、复杂场景 |

#### 训练过程

**3-epoch 烟雾测试**（验证数据配置正确性）→ **100-epoch 全量训练**：

```bash
yolo pose train \
  model=yolov8n-pose.pt \
  data=plate_true_quad_pose/dataset.yaml \
  imgsz=640 epochs=100 batch=64 patience=20 workers=8 \
  kpt_shape=[4,3] \
  degrees=5 translate=0.05 scale=0.5 shear=2 \
  perspective=0.0005 mosaic=0.5 mixup=0.0 \
  fliplr=0.0 cache=ram
```

#### Pose 检测器性能

| 指标 | 数值 |
|------|:----:|
| 所有绿牌角点误差中位数 | 4.51px |
| 困难绿牌角点误差中位数 | 4.96px（OBB 的 **2.6× 更优**） |
| 所有绿牌角点误差均值 | 5.15px |
| 门控拒绝 | 0 |
| 管线阶段 | 1（单模型检测即输出） |

#### 导出部署

```bash
# ONNX 导出
yolo export model=experiments/yolov8n-pos/weights/best.pt \
  format=onnx opset=12 imgsz=640 simplify=True

# RKNN 转换（FP16）
# 板端加载路径：/userdata/model/pose_detector.rknn
```

---

## 3. LPRNet 车牌识别模型

LPRNet 是系统的**核心 OCR 引擎**。设计哲学是：极轻量的 CNN + CTC 解码，在嵌入式设备上达到实时推理。

### 3.1 架构设计

#### 3.1.1 基础架构

```
输入: (3, 24, 94) BGR 图像
    │
    ▼
Backbone（4 stage CNN）
    ├── Conv(3→64, 3×3, stride=1) + BN + ReLU
    ├── MaxPool3d(1,3,3)
    ├── small_basic_block(64→128) + BN + ReLU
    ├── MaxPool3d(2,1,2)
    ├── small_basic_block(64→256) + BN + ReLU
    ├── small_basic_block(256→256) + BN + ReLU
    ├── MaxPool3d(4,1,2) + Dropout
    ├── Conv(64→256, 1×4) + BN + ReLU
    ├── Dropout
    ├── Conv(256→class_num, 13×1) + BN + ReLU
    │
    ▼
多尺度全局上下文
    ├── 提取 layer 2, 6, 13, 22 的输出
    ├── 分别 AvgPool 到统一尺寸
    ├── L2 归一化
    ├── concat → (448+class_num, H, W)
    │
    ▼
Container（分类头）
    └── Conv(1×1) → mean(dim=2) → (B, class_num, 18) → CTC decode
```

**small_basic_block**——非对称深度可分离卷积变体：
```python
Conv(ch_in → ch_out//4, 1×1) + ReLU
    → Conv(ch_out//4, (3,1), padding=(1,0)) + ReLU  # 垂直纹理
    → Conv(ch_out//4, (1,3), padding=(0,1)) + ReLU  # 水平纹理
    → Conv(ch_out//4 → ch_out, 1×1)
```

在极少的参数量下捕捉车牌字符的垂直+水平空间结构。

**CTC 解码**：输出 logits `(B, class_num, 18)`，18 个时间步对应特征图宽度方向序列。`beam_size=30, beam_topk=15` 的波束搜索解码。

**字符集**：31 个省汉字 + 24 个大写字母（去 I/O）+ 10 个数字 + CTC blank = 66 类。

#### 3.1.2 多头架构（LPRNetMultiHead）

为了在同一个 backbone 上同时处理 7 位蓝牌和 8 位绿牌，引入多头架构：

```
输入 → Backbone（共享）
         │
         ├── containers.normal7 → 标准 Conv(1×1) → normal7 CTC head
         ├── containers.green8  → 增强 Conv(3×3, 256ch) + Conv(1×1) → green8 expD head
         └── containers.special → 标准 Conv(1×1) → special CTC head
```

**绿牌增强头 expD**：
```python
# 比标准 1×1 Conv 更深——因为绿牌 8 位更复杂
Conv(448+class_num → 256, 3×3, padding=1) + ReLU + Dropout(0.3)
Conv(256 → class_num, 1×1)
```

**辅助头**（可选）：
- `pos0_head`：省份字符专用分类器（31 省）
- `pos0_family_heads`：per-family 省份头
- `province_head` / `slot_head`
- `family_adapters`：per-family 轻量适配层

**训练时可用辅助损失**：

| 损失项 | 生产权重 | 目标 |
|--------|:-------:|------|
| 主 CTC 损失 | 1.0 | 全序列 |
| `first_char_aux` | 0.30 | 省份字符 |
| `rear_seq_aux` | 0.30 | 尾部子序列 |

#### 3.1.3 家庭感知波束解码

每个 family 有独立的字符模板约束：

| Family | 格式 |
|--------|------|
| normal7 | 省(1) + 字母(1) + 数字+字母(5) = 7 位 |
| green8 | 省(1) + 字母(2) + 数字(5) = 8 位，pos3 不限于 D/F |

### 3.2 推理前处理管线

板端 C 代码（`fpga_lpr_display.c`）的 OCR 前处理与训练端（`load_data.py`）逻辑等价：

```
检测器输出 quad (四点)
    → order_quad_points() 规范化
    → warp_quad_to_rect_rgb888 / getPerspectiveTransform
    → 透视矫正到正向矩形
    → resize_rgb888_letterbox_kernel → 94×24 (nn 插值)
    → [preproc=none] 无额外预处理
    → LPRNet RKNN 推理
```

**一致性确认**：板端 vs 训练端在单应矩阵计算、像素采样、插值、边缘处理四个环节上等价（差异 < 0.5px）。

**生产预处理**：`--ocr-preproc none`（保留 BGR 彩色，不使用 gray3）。生产模型不再依赖 gray3 预处理。

---

## 4. 训练数据体系

### 数据源总览

| 数据源 | 牌型 | 规模 | 几何标注 | 文字标注 |
|--------|:----:|:----:|:--------:|:--------:|
| CCPD2019（含困难类） | 蓝牌 | ~355K | 文件名 GT quad | ❌ |
| CCPD2020 `ccpd_green` | 绿牌 | ~24K | 文件名 GT quad | ❌ |
| CRPD_all | 蓝/黄/白 | ~34K | Label 文件 | ✅ |
| 合成替换数据（Pose quad） | 绿牌 | ~5K+ | Pose 检测器 | ✅ |

### CCPD 四点顺序约定（关键坑）

CCPD 文件名中的 quad 是 `[BR, BL, TL, TR]` 顺序，与规范化的 `[TL, TR, BR, BL]` 不同。几何启发式 `order_quad_points()` 在 CCPD 输入上会错误交换 BL/TR，导致文字镜像。

**修复**：对 CCPD 数据使用显式重排：
```python
gt_quad = np.array([gt_quad_raw[2], gt_quad_raw[3],
                     gt_quad_raw[0], gt_quad_raw[1]], dtype=np.float32)
```

`order_quad_points()` **只适用于** YOLO OBB/Pose 输出，不适用于 CCPD 文件名 quad。

### 绿牌替换数据生成

为打破 CCPD2020 的 97% 皖偏置，开发了替换生成管线：

```
CCPD2020 照片 → GT quad 正向 warp → 246×72 画布
  → 合成新文本（CharsImageGenerator，省域可控）
  → LAB L 通道亮度匹配（保留绿牌颜色）
  → 反向 warp → 高斯遮罩合成 → 替换后图像
  → Pose quad 存入 manifest 内联列
  → prepare_board_ocr_input_from_quad_bgr888 → 94×24
```

**省域控制**：31 省均衡，皖最少（~1%），其余省均分。替换了 CCPD2020 原生的安徽极端偏置。

**Source-level train/val 拆分**：按原始照片路径拆分，不按生成行拆分，确保零数据泄露。

---

## 5. 训练方法论

经过数十轮实验（Stage A → Stage B → E1-E28），打磨出系统化的训练方法。

### 5.1 E 系列实验框架

当定位到具体失败模式时，按三步推进：

**第一步：Per-Position 诊断**
- Train exact > 50%, Val similar → 数据量不足，可扩展
- Train exact < 5%, Val similar → **信号瓶颈**，加数据无效，需改数据质量/模型结构

**第二步：基线确认** — 在目标评测集上测量当前最佳模型

**第三步：单变量跑** — 一次只改一个变量

### 关键实验路线

| 阶段 | 实验 | 目标 | 结论 |
|------|------|------|------|
| Stage A | 课程学习 | 建立 gray3 基础 OCR 能力 | ✅ 基础建立 |
| Stage B1A | 保守 hard/bridge | 引入困难数据 | ✅ 方向正确 |
| Stage B2-C/D | 替换数据 soft-freeze | 绿牌省域均衡 + 极端倾斜 | ✅ 首次实质改善 |
| E1-E3 | 极端倾斜替换数据 | 角度极端数据学习 | ✅ 可训练但 OBB 不足 |
| E9-E15A | 板端 dump 数据 + 抗坍塌 | 板端 cluster 失败模式 | ⚠️ 部分进展 |
| **E12 + 替换 + 解冻** | Pose quad 替换 + backbone 18/19/20 解冻 | 绿牌板端全面改善 | ✅ Cluster3 突破 |
| **prov_degrade** | 省份区域退化训练 | 解决首字坍塌 | ✅ 生产模型 |
| E19-E28 | 特定 cluster 分治 | cluster2/3 精细化处理 | 持续改善 |

### 省份退化训练（prov_degrade）

这是最终生产绿牌模型的核心技术——专门针对板端 dump 中首字符区域的信息退化问题：

- 只在 warp 后图像左 35% 区域（省份位置）施加退化
- 退化类型：暗化、过曝、低对比度、左边缘裁剪、水平偏移、高斯模糊
- 8% feather zone 平滑过渡，无硬边界
- 尾部区域（suffix）**不退化**——因为 suffix 识别率已足够高
- 辅助损失 `first_char_aux_weight=0.30`、`rear_seq_aux_weight=0.30`

---

## 6. 板端部署

### 6.1 最终生产配置

```bash
COMMON="--veh-model /userdata/model/yolov5s_rk3568.rknn \
  --plate-model /userdata/model/plate_yolov8n_obb_green_clone_success_middecode_rk3568_fp16.rknn \
  --ocr-blue-model /userdata/model/LPRNet_stage3_rk3568_fp16_more_trained.rknn \
  --ocr-green-model /userdata/model/prov_deg_fp16_no_rknnpre.rknn \
  --ocr-keys /userdata/model/ocr_keys_lprnet.txt \
  --quad-refiner-model off \
  --labels /userdata/model/coco_80_labels_list.txt \
  --plate-detector-type yolov8_obb_rknn \
  --plate-class-id -1 \
  --fps 15 --copy-buffers 2 --queue-depth 1 \
  --min-car-conf 0.25 --min-plate-conf 0.45 \
  --plate-on-car-only 0 --plate-only 1 \
  --sw-preproc 0 --fpga-a-mask 0 \
  --plate-nms-iou 0.35 --plate-max-det 32 \
  --ocr-channel-order bgr --ocr-crop-mode obb_warp \
  --ocr-resize-mode letterbox --ocr-resize-kernel nn \
  --ocr-min-plate-h 8 --ocr-min-occ-ratio 0.90 \
  --plate-refine 0 --show-crop-box 1 \
  --ocr-crop-dump-dir /tmp/ocr_dump_green \
  --ocr-crop-dump-max 50 \
  --ocr-preproc none"
```

**关键配置解读**：

| 参数 | 值 | 含义 |
|------|-----|------|
| `--veh-model` | yolov5s_rk3568.rknn | 车辆检测模型（辅助） |
| `--plate-detector-type` | yolov8_obb_rknn | 当前板端检测配置 |
| `--ocr-blue-model / --ocr-green-model` | 分模型 | 蓝牌和绿牌使用独立的 OCR 模型 |
| `--quad-refiner-model off` | off | **refiner 不启用**（Pose 已替代） |
| `--plate-refine 0` | 0 | 二次定位关闭 |
| `--ocr-preproc none` | none | 保留彩色，不灰度化 |
| `--ocr-crop-mode obb_warp` | obb_warp | 透视矫正 |
| `--ocr-min-occ-ratio 0.90` | 0.90 | occ 阈值仅非 obb_warp 时生效 |
| `--ocr-crop-dump-max 50` | 50 | 最多 dump 50 帧用于调试 |

### 6.2 Pose 检测器集成

上述配置中的 `--plate-detector-type yolov8_obb_rknn` 为当前板端配置。Pose 检测器已完成训练和 RKNN 导出（`experiments/yolov8n-pos/export/best_fp16.rknn`），板端 C 代码已集成 `DETECTOR_YOLOV8_POSE_RKNN` 路径（`decode_yolov8_pose_outputs`），切换使用 Pose 的方式为：

```
--plate-detector-type yolov8_pose_rknn
```

**Pose vs OBB 对比**：

| 维度 | OBB | Pose |
|------|:---:|:----:|
| 角点精度中位数 | 10.07px | 4.51px |
| 困难绿牌中位数 | 13.12px | 4.96px（2.6× 更优） |
| 管线阶段 | 2（检测 + refiner） | **1**（检测即输出） |
| 门控拒绝 | 有 | **0** |

---

## 7. 关键实验历程

### 7.1 Quad Refiner（已废弃，未上板）

实验了从 OBB coarse quad 精修到真实四角的 refiner（ResNet18 + FPN + offset head）：

| 版本 | 核心变化 | 结果 |
|:----:|----------|:----:|
| V1 基线 | 4 heatmap + mask | 角点误差 -59%，Warp MAD -20% |
| V2a | 6x 绿牌采样 + hard batch | **无改善** → 采样不是瓶颈 |
| **V2b** | + offset head | OCR +17.6% ↑，68% 改善率 |
| V2c | + warp-aware 损失 | **退化** → 梯度冲突 |

**结论**：V2b 方向正确但最终被 Pose 检测器替代——Pose 在单阶段达到同等精度且无门控拒绝。

### 7.2 替换数据的两次坑

1. **`order_quad_points()` 坑**（2026-05-03）：使用几何启发式排序 CCPD quad → BL/TR 交换 → 生成数据文字反转。训练后 train exact = val exact = 0%。修了两次。

2. **Manifest 字段名坑**（2026-05-04）：内联 quad 列名必须为 `quad_1x..quad_4y` 而非 `x1..y4`，否则训练加载器找不到 quad，fallback 到文件名解析（文件名非标准 CCPD 格式时失败）。

### 7.3 B2-C_OBB 阴性结果

把训练数据 quad 从 GT 换成 OBB 检测器输出，只变这一项 → cluster3 与 B2-C 完全相同（均为 0%）。

**结论**：检测器 quad 精度差异不是 cluster3 根因。OBB 检测器在 CCPD2020 上精度太高（>0.93 置信度），产生的 warp 和 GT 没区别。

### 7.4 Pose 替换 + 解冻突破

从 E12 基座出发，追加 2790 Pose quad 替换数据 + backbone.18/19/20 解冻：

| 变化 | 参数 |
|------|------|
| 基座 | E12 Final |
| 新数据 | 2790 Pose quad 替换数据（省域均衡） |
| freeze_backbone | `true`（bk.18/19/20 可训练） |
| LR | 0.0001 |
| max_epoch | 6（prov_degrade 时 5） |
| first_char_aux | 0.30 |

这是替换数据对板端 cluster3 的首次实质改善。

### 7.5 省份退化训练（生产模型）

最终生产绿牌模型 `prov_deg_fp16_no_rknnpre.rknn` 的核心思路：

- 问题诊断确认：Cluster2（京AD06088）的"京"在板端该场景下**从信息论上已不可恢复**——高分辨率省份分类器（80×90px）也输出 0/19
- 针对可恢复的退化场景（如 cluster3 苏BF01111），**省份区域退化训练**让模型学会在输入退化的首字区域仍然正确识别
- 对于信息不可恢复的情况，采用**轨迹融合**输出 `province_unreliable` 标记，让上层系统通过多帧时序恢复

---

## 8. 附录：名词对照

| 缩写 | 全称 | 说明 |
|------|------|------|
| OBB | Oriented Bounding Box | 旋转框检测，输出角度+中心点+宽高 |
| Pose | Keypoint Detection | 关键点检测，输出 4 个角点坐标 |
| CTC | Connectionist Temporal Classification | 序列到序列的对齐损失 |
| RKNN | Rockchip Neural Network | RK3568 NPU 的模型格式 |
| Occ Ratio | Occupancy Ratio | 车牌在裁切框中的占空比 |
| Quad | Quadrilateral | 四点表示的四边形 |
| CCPD | Chinese City Parking Dataset | 中国停车场车牌数据集 |
| CRPD | Chinese Road Plate Dataset | 中国道路车牌数据集 |
| Gray3 | 三通道灰度 | BGR→gray→stack([gray,gray,gray]) |
| Family | 牌型家族 | normal7(7位蓝牌)、green8(8位绿牌)、special(变长) |
| Manifest | 训练清单 | CSV 管理所有训练样本 |
| Beam Search | 集束搜索 | CTC 解码多路径保留 |
| Prov_deg | Province Degradation | 省份区域退化训练技术 |
| ExpD | Enhanced Green Head variant D | 绿牌增强分类头（3×3 Conv + 1×1 Conv） |
