# FPGA数据到CPU处理、YOLOv8n OBB与LPRNet识别流程说明

更新时间：2026-03-19  
目标：讲清楚两件事
1. CPU（驱动+用户态）如何处理 FPGA 送来的帧数据。
2. YOLOv8n OBB 与 LPRNet 在运行时如何识别同一帧画面。

## 1. 证据来源（本仓库与ARM代码）

- 驱动接口与结构体：`/home/wzzz/ARM/pcie_fpga_dma.h`
- 驱动核心流程：`/home/wzzz/ARM/pcie_fpga_dma.c`
- ARM端主程序（取帧/推理/显示）：`/home/wzzz/ARM/fpga_lpr_display.c`
- LPR最新实验：`/home/wzzz/LPRNet/experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/`
- YOLO OBB 训练结果：`/home/wzzz/LPRNet/yolov8n_obb_trained_70epoch/`
- 实验脚本（OCR契约参数）：`/home/wzzz/LPRNet/run_tilt_obbwarp_experiment.sh`

---

## 2. CPU如何处理FPGA送来的数据

### 2.1 内核驱动层（PCIe DMA）

驱动提供三个关键 IOCTL：
- `FPGA_DMA_GET_INFO`：读设备信息（分辨率、像素格式、stride等）
- `FPGA_DMA_READ_FRAME`：触发一次DMA读帧
- `FPGA_DMA_MAP_BUFFER`：给用户态返回 ring buffer 的 mmap 偏移

核心机制：
- `probe` 阶段完成 PCI 设备使能、BAR0/BAR1 映射、DMA coherent ring buffer 分配。
- DMA优先走 IRQ 帧模式（`DMA_CMD_FRAME_MODE`）；IRQ异常或不可用时可退化到 polling 分块搬运。
- polling 路径按 chunk 写 DMA 控制寄存器，并用尾部 sentinel 检测“FPGA是否已覆盖内存”。
- `READ_FRAME` 根据 `transfer.offset` 选 ring 槽位，DMA完成后可 `copy_to_user` 到用户缓冲。
- `mmap` 路径通过 `dma_mmap_coherent` 将选定 ring 槽位映射到用户态。

简单理解：驱动负责把“FPGA端帧缓存”稳定搬运到“ARM可读的DMA一致性内存”，并向用户态暴露统一读帧接口。

### 2.2 用户态主循环（fpga_lpr_display）

初始化阶段：
- `init_fpga_dma()` 打开 `/dev/fpga_dma0`，通过 `GET_INFO` 读取帧信息（1280x720，BGR565或BGRX8888）。
- 调 `MAP_BUFFER` + `mmap` 建立DMA映射，同时申请 `ctx->dma_copy` 作为每帧CPU私有副本。

每帧主循环（显示线程）：
1. `trigger_frame_dma()`：发 `FPGA_DMA_READ_FRAME`，把一帧复制到 `ctx->dma_copy`。
2. `copy_frame_to_slot565()`：转成内部显示格式（BGR565）写入slot。
3. `push_latest_to_infer()`：把最新原始帧喂给推理线程（异步）。
4. `overlay_results_on_slot()`：把上一轮推理结果叠加到当前显示帧。
5. `gst_app_src_push_buffer()`：送入 `appsrc -> queue(leaky) -> kmssink`。

并发策略：
- 显示与推理解耦：主循环只负责稳定采集+显示，不阻塞等待推理。
- `push_latest_to_infer()` 是“最新帧覆盖”策略：若推理线程来不及，会覆盖旧待推理帧，并累计 `infer_overwrite_count`。
- GStreamer queue 设为 `leaky`，优先实时性，允许丢旧帧，避免端到端延迟越积越大。

### 2.3 CPU对像素与预处理的处理

CPU收到帧后按源格式解码：
- BGR565 路径：`decode_pixel565()` / `raw565_to_rgb888_full()`
- BGRX8888 路径：`bgrx8888_to_rgb888_and_a()`（同时提取A通道）

可选预处理：
- 软件预处理：`sw_preprocess_rgb888()`
- FPGA A通道ROI：`extract_a_channel_roi()`，并结合红灯稳定帧计数做ROI筛选/阈值调节

---

## 3. YOLOv8n OBB如何识别画面帧

入口：`run_detect_on_rgb()`

处理链路：
1. **检测前变换**：`prepare_detect_canvas()`  
   把原图变成 `ALGO_STREAM_SIZE x ALGO_STREAM_SIZE`（letterbox 或 stretch）。
2. **模型输入适配**：若模型输入尺寸不等于检测画布，再次resize到 `m->in_w x m->in_h`。
3. **RKNN推理**：`run_model_detect()` 调 `rknn_inputs_set -> rknn_run -> rknn_outputs_get`。
4. **OBB解码**：`decode_yolov8_obb_outputs()`  
   - 解析 dist/cls/angle 三类输出视图  
   - 用 anchor cache（stride 8/16/32）还原旋转框中心、宽高、角度  
   - 生成四点quad并做 `rotated_nms_inplace()`
5. **坐标回映射**：把检测空间坐标映射回原帧空间（含OBB四点）。

运行时稳态策略（不是训练策略）：
- OBB+letterbox下若首轮空框，自动重试 stretch。
- 规则过滤后若为空，可能保留最高置信 top1（满足阈值）避免全空帧。
- `temporal_confirm_and_update()` 做时序确认，降低闪检。
- 可选 `refine_plate_box_local()` 在局部ROI再检，提高牌框几何质量。

---

## 4. LPRNet如何识别画面帧（OCR）

入口在推理线程中，每个稳定车牌框进入 OCR 流程。

### 4.1 车牌裁剪（重点：OBB透视矫正）

`prepare_plate_crop_rgb888()`：
- 普通模式：按 box/tight/match 等策略裁剪。
- OBB模式：若 `has_obb`，走 `warp_quad_to_rect_rgb888()`，把四边形透视拉正为矩形再送OCR。
- 若占比不足（`ocr_min_occ_ratio`），可触发 recrop（match-ytrim 或扩展框）再试。

### 4.2 OCR输入构造

`prepare_ocr_input_rgb888()`：
- 可选灰度/二值预处理（当前实验锁定 `none`）。
- resize：letterbox 或 stretch（当前实验锁定 letterbox）。
- kernel：NN 或 bilinear（当前实验锁定 NN）。
- 通道顺序：RGB/BGR（当前实验锁定 BGR）。

### 4.3 LPRNet推理与解码

`run_model_ocr()`：
1. RKNN推理得到 logits。
2. `build_ocr_layout()` 识别输出张量时序维/类别维。
3. `ctc_decode_logits()` 进行 CTC 解码：去 blank、去重复、拼接字符 token。
4. 产出 `ocr_text + conf`，并记录 blank top1 比例等诊断量。

### 4.4 OCR结果时序平滑

`ocr_temporal_smooth()` 对连续帧文本做时序融合，降低单帧抖动，最终用于叠字与日志。

---

## 5. 训练日志能说明什么（截至2026-03-19）

### 5.1 LPRNet v7 相对 v6 的验收结果

来自 `acceptance.json`：
- `passed = true`
- hard集整牌准确率：`0.4858219 -> 0.5989145`（相对提升 `23.28%`）
- normal集整牌准确率：`0.4843501 -> 0.6052669`（无回退，实际提升）

阶段日志（Final test Accuracy）：
- StageA：`0.526778`
- StageB：`0.543111`
- StageC(resume)：`0.545889`

补充：最终模型在 hard/normal 的详细指标见：
- `test_metrics_hard.json`
- `test_metrics_normal.json`

### 5.2 YOLOv8n OBB 训练指标

来自 `yolov8n_obb_trained_70epoch/results.csv` 第70轮：
- precision：`0.99996`
- recall：`1.0`
- mAP50：`0.995`
- mAP50-95：`0.99442`

运行参数（`args.yaml`）显示：
- `task=obb`
- `imgsz=640`
- `single_cls=true`

仓库默认训练脚本 `run_yolov8_obb_train.sh` 也保持同方向配置（`task="obb"`、`imgsz=640`、`single_cls=True`）。

### 5.3 与板端OCR契约的一致性

`run_tilt_obbwarp_experiment.sh` 锁定了这组参数：
- `ocr_channel_order=bgr`
- `ocr_crop_mode=obb_warp`
- `ocr_resize_mode=letterbox`
- `ocr_resize_kernel=nn`
- `ocr_preproc=none`

同时 ARM 程序里对 `DETECTOR_YOLOV8_OBB_RKNN` 做了强制保护：若参数不一致，会自动改回上述契约（并打印 `[cfg]` 提示）。

---

## 6. 一句话总结（回答原问题）

- **CPU处理FPGA数据**：内核驱动通过 PCIe DMA 把帧写入 ring buffer，用户态每帧 `READ_FRAME` 取到副本后做像素解码/预处理，主线程负责显示与叠加，推理线程异步消费最新帧。  
- **YOLOv8n OBB识别帧**：先把帧缩放到检测空间，RKNN输出经 OBB 解码+旋转NMS，映射回原图得到车牌旋转框。  
- **LPRNet识别帧**：按牌框（优先 OBB 透视矫正）生成OCR crop，按契约做 resize/通道处理后推理，CTC解码出车牌字符，再做时序平滑输出。
