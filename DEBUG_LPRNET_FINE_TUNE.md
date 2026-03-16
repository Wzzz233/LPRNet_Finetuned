# LPRNet Fine-tune Debug Guide

## 1. 目标

定位为什么微调后的 LPRNet OCR 模型在板端实跑时，效果明显弱于官方模型，并给出一套可复现、可量化、可修复的 debug 流程。

当前工作区涉及两部分代码：

- 训练与导出侧：`C:\Users\Wzzz2\OneDrive\Desktop\LPRNet`
- 板端运行侧：`C:\Users\Wzzz2\OneDrive\Desktop\新建文件夹 (2)\新建文件夹`

## 2. 现象

根据 [`运行日志.txt`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/运行日志.txt#L1)：

- 微调模型：`/userdata/model/LPRNet_stage3_rk3568_fp16_git.rknn`
- 官方模型：`/userdata/model/ocr_lprnet_rk3568_fp16.rknn`
- 两次实跑的 OCR 前处理参数基本一致：
  - `ocr_ch=bgr`
  - `ocr_crop=match`
  - `ocr_resize=letterbox`
  - `ocr_kernel=nn`
  - `ocr_pp=none`
  - `min_occ=0.90`

微调模型的输出特征：

- 经常只出短串：`N`、`GN`、`N8`、`NDP`
- 偶尔出半残串：`苏N8F8`
- 典型表现是字符数偏短，且高置信错误很多

官方模型的输出特征：

- 大多数时候能稳定输出完整 7 字符串
- 例如：`粤N8P8F8`、`京N8P8F8`

这说明：

- 板端部署链路本身不是完全坏掉
- 更像是微调权重本身退化了
- 退化模式很像 CTC blank/删字主导

## 3. 已确认的关键证据

### 3.1 板端两次运行的 OCR 前处理没有本质差异

日志中两次 `Start LPR loop` 的 OCR 配置一致：

- 微调模型配置见 [`运行日志.txt`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/运行日志.txt#L32)
- 官方模型配置见 [`运行日志.txt`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/运行日志.txt#L126)

因此，模型差距不能优先归咎于板端命令行参数不同。

### 3.2 板端 OCR 输入链路与训练侧并不完全同分布

训练侧预处理：

- 直接 `cv2.imread(BGR)`
- 强制 `resize(94,24)`
- `(img - 127.5) * 0.0078125`
- 见 [`load_data.py`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/load_data.py#L65)

板端 OCR 输入链路：

- 从全图裁 plate crop
- 如果占比低于阈值，执行 `match-ytrim` recrop
- 再做 `letterbox`
- 再根据 `ocr_channel_order` 决定是否交换 RGB/BGR
- 见 [`fpga_lpr_display.c`](C:\Users\Wzzz2\OneDrive\Desktop\新建文件夹 (2)\新建文件夹\fpga_lpr_display.c) 的：
  - `prepare_ocr_input_rgb888` 第 1186-1243 行
  - `match-ytrim` recrop 第 4824-4863 行
  - OCR CTC decode 第 1113-1184 行

这意味着训练和上线存在至少三类分布偏移：

- 几何分布偏移：`resize` vs `letterbox`
- 裁剪分布偏移：静态样本 vs detect/refine/recrop 产物
- 清晰度和占比过滤：板端会跳过小框、模糊框

### 3.3 微调训练脚本存在一个高风险 bug：模型大概率一直在 eval 模式训练

训练脚本里：

- `phase_train` 默认是布尔值 `True`
- 见 [`train_LPRNet.py`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/train_LPRNet.py#L71)

模型构建里：

- 只有 `phase == "train"` 时才 `Net.train()`
- 否则直接 `Net.eval()`
- 见 [`LPRNet.py`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/LPRNet.py#L82)

而训练入口实际传入的是：

- `build_lprnet(..., phase=args.phase_train, ...)`
- 见 [`train_LPRNet.py`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/train_LPRNet.py#L111)

所以当前默认逻辑下：

- `args.phase_train` 是 `True`
- `True != "train"`
- 模型会走 `else -> Net.eval()`

这会造成：

- BatchNorm 统计量不更新
- Dropout 失效
- 微调时无法正常适配新数据分布

这是当前最强的根因候选。

### 3.4 训练脚本默认把训练集和测试集指向同一目录

见 [`train_LPRNet.py`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/train_LPRNet.py#L63)：

- `train_img_dirs=./balanced_ccpd_red_ppm`
- `test_img_dirs=./balanced_ccpd_red_ppm`
- `txt_file=./balanced_ccpd_red_ppm/train_labels.txt`

这意味着默认配置下：

- 训练和验证没有严格隔离
- 离线 test acc 不能代表上线泛化能力

### 3.5 学习率调度实现可疑

见 [`train_LPRNet.py`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/train_LPRNet.py#L43)

当前实现逻辑：

- 当 `cur_epoch < 4` 用 `base_lr`
- 当 `cur_epoch < 8` 用 `base_lr * 0.1`
- 当 `cur_epoch < 12` 用 `base_lr * 0.01`
- ...
- 但当 `cur_epoch >= 16` 时，`lr == 0`，最后又被重置成 `base_lr`

这不是常见的衰减逻辑。15 epoch 默认配置下影响有限，但如果阶段训练拉长，这会造成后期 lr 异常回升。

### 3.6 微调模型的板端症状符合 CTC blank 偏高/删字退化

板端 decode 是标准 greedy CTC：

- blank 或重复字符直接跳过
- 见板端 `ctc_decode_logits` 第 1156-1160 行

微调模型在日志中频繁只输出短串：

- [`运行日志.txt`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/运行日志.txt#L36)
- [`运行日志.txt`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/运行日志.txt#L44)
- [`运行日志.txt`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/运行日志.txt#L56)
- [`运行日志.txt`](/C:/Users/Wzzz2/OneDrive/Desktop/LPRNet/运行日志.txt#L77)

这与以下情况一致：

- blank top1 比例偏高
- 时序上大部分位置被 blank 吃掉
- 最终只剩 1 到 4 个字符被输出

## 4. 高概率根因排序

### P0：训练模式 bug

优先级最高。

如果微调全程都在 `eval()`：

- BN 无法更新
- dropout 不工作
- 数据分布一变就容易崩

这足以解释为什么：

- 官方模型正常
- 微调模型反而退化

### P1：训练分布和板端分布不一致

训练时样本是固定标注 crop，直接拉伸到 `94x24`。

板端时样本来自：

- 检测框
- refine 后框
- `match-ytrim` recrop
- `letterbox`
- 有时还会受 FPGA 预处理影响

如果微调数据本身不够多样，模型就会朝“你喂给它的静态样本”过拟合，丢失对真实检测框裁剪的鲁棒性。

### P2：离线评估配置误导

训练和测试默认同目录，导致你可能看到的 test acc 偏乐观。

### P3：微调超参数过激

当前脚本使用：

- `RMSprop`
- `lr=0.001`
- 默认直接加载 pretrained 然后全量微调

如果数据量不大，或者 stage3 数据分布很窄，`0.001` 仍可能太大，造成原始能力被覆盖。

## 5. 板端代码路径结论

### 5.1 通道顺序不是当前第一嫌疑

板端内部先把 full frame 解码成 RGB 顺序，再在 `ocr_channel_order == bgr` 时交换 `R/B`，见 `prepare_ocr_input_rgb888` 第 1231-1239 行。

由于训练侧使用的是 OpenCV BGR，板端设置 `--ocr-channel-order bgr` 是合理的。

并且官方模型在同样设置下表现正常，这进一步说明通道顺序不是主因。

### 5.2 RKNN 输入格式看起来也正常

板端 OCR 模型输入：

- `RKNN_TENSOR_UINT8`
- `NHWC`
- 输入尺寸 `94x24x3`
- 见 `run_model_ocr` 第 1268-1274 行

而导出/转换脚本中，RKNN 配置了：

- `mean_values=[[127.5,127.5,127.5]]`
- `std_values=[[128.0,128.0,128.0]]`

这与训练侧归一化是一致的。

### 5.3 当前板端日志已经支持 blank_top1 诊断

板端已有 `--ocr-ctc-diag 1`：

- 启用后会打印 `[ctc] ... blank_top1=...`
- 见 `fpga_lpr_display.c` 第 4913-4919 行

这是定位“模型是否 blank 塌陷”的最佳实时指标。

## 6. 建议的完整 debug 流程

下面按优先级给出完整排查路径。

### Step 1：先修训练代码，再重新训练一个最小对照模型

必须先修这个问题，否则后面的任何对比都不干净。

建议修改：

1. 修 `build_lprnet` 的 phase 判定
2. 明确打印 `lprnet.training`
3. 将 train/val 拆开

最小修法建议：

```python
def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)
    if phase is True or phase == "train":
        return Net.train()
    return Net.eval()
```

同时在训练启动后打印：

```python
print("model.training =", lprnet.training)
```

预期：

- 修完后打印必须是 `True`

### Step 2：用板端 `ctc_diag` 先验证是不是 blank 塌陷

在板子上对微调模型开启：

```bash
sudo ./run_lpr_kms.sh \
  --veh-model /userdata/model/yolov5s_rk3568.rknn \
  --plate-model /userdata/model/plate_yolov5n_rk3568.rknn \
  --ocr-model /userdata/model/LPRNet_stage3_rk3568_fp16_git.rknn \
  --ocr-keys /userdata/model/ocr_keys_lprnet.txt \
  --labels /userdata/model/coco_80_labels_list.txt \
  --fps 15 \
  --plate-only 1 \
  --sw-preproc 0 \
  --fpga-a-mask 0 \
  --ocr-channel-order bgr \
  --ocr-crop-mode match \
  --ocr-resize-mode letterbox \
  --ocr-resize-kernel nn \
  --ocr-preproc none \
  --ocr-min-occ-ratio 0.90 \
  --ocr-ctc-diag 1 \
  --pred-log /tmp/score/pred_finetune_diag.csv
```

观察：

- `[ctc] ... blank_top1=...`
- `[pred] ... text=...`

判据：

- 如果微调模型 `blank_top1` 明显高于官方模型，且短串很多，说明就是 CTC 塌陷

### Step 3：跑板端现成的 root-cause 脚本

板端已经有现成脚本：

- `run_ocr_root_cause.sh`
- `ocr_phase0_metrics.py`
- `ocr_demo_compare.py`

建议先跑：

```bash
sudo ./run_ocr_root_cause.sh \
  --veh-model /userdata/model/yolov5s_rk3568.rknn \
  --plate-model /userdata/model/plate_yolov5n_rk3568.rknn \
  --ocr-model /userdata/model/LPRNet_stage3_rk3568_fp16_git.rknn \
  --ocr-keys /userdata/model/ocr_keys_lprnet.txt \
  --labels /userdata/model/coco_80_labels_list.txt \
  --gt-text 你的车牌号 \
  --demo-bin /userdata/path/to/rknn_lprnet_demo
```

这个脚本会自动跑 4 组：

- `base`: `rgb + fixed + stretch + none`
- `ch`: `bgr + fixed + stretch + none`
- `roi`: `bgr + box-pad + letterbox + none`
- `pp`: `bgr + box-pad + letterbox + bin`

然后生成：

- `metrics_*.json`
- `metrics_*.txt`
- `demo_compare_summary.json`

判据：

- 如果 `ch` 明显优于 `base`，通道顺序有影响
- 如果 `roi` 明显优于 `ch`，ROI/letterbox 是关键
- 如果 `demo` 在同一张 dump 图上明显优于 app，说明 app 侧输入构造或 decode 还有问题
- 如果 `demo` 和 app 都差，说明模型本身差

### Step 4：做“同输入对比”

这是区分“模型问题”和“板端输入问题”的核心动作。

思路：

1. 打开 `--ocr-crop-dump-dir`
2. 让 app 保存实际送给 OCR 的输入
3. 用 `ocr_demo_compare.py` 调官方 demo 在同一批输入上跑

命令模板：

```bash
sudo ./run_lpr_kms.sh \
  --veh-model /userdata/model/yolov5s_rk3568.rknn \
  --plate-model /userdata/model/plate_yolov5n_rk3568.rknn \
  --ocr-model /userdata/model/LPRNet_stage3_rk3568_fp16_git.rknn \
  --ocr-keys /userdata/model/ocr_keys_lprnet.txt \
  --labels /userdata/model/coco_80_labels_list.txt \
  --ocr-channel-order bgr \
  --ocr-crop-mode match \
  --ocr-resize-mode letterbox \
  --ocr-resize-kernel nn \
  --ocr-preproc none \
  --ocr-crop-dump-dir /tmp/ocr_cmp \
  --ocr-crop-dump-max 40 \
  --ocr-ctc-diag 1 \
  --pred-log /tmp/score/pred_dump.csv
```

然后：

```bash
python3 ./ocr_demo_compare.py \
  --index /tmp/ocr_cmp/index.csv \
  --demo-bin /userdata/path/to/rknn_lprnet_demo \
  --model /userdata/model/LPRNet_stage3_rk3568_fp16_git.rknn \
  --max-samples 20 \
  --out-csv /tmp/score/demo_compare_samples.csv \
  --out-json /tmp/score/demo_compare_summary.json
```

结果解释：

- `app_demo_same_ratio` 高，但都错：模型本身差
- `app` 差、`demo` 好：板端 app 输入或 decode 有问题
- 两者都好：在线 live 的 ROI/crop 选择有问题

### Step 5：用离线图固定 ROI 复现

如果 live 流太不稳定，改用板端 offline 模式固定问题。

`run_lpr_kms.sh` 已支持：

- `--offline-image`
- `--offline-roi`
- `--offline-detect-plate`

建议命令：

```bash
sudo ./run_lpr_kms.sh \
  --offline-image /tmp/test.ppm \
  --offline-roi x1,y1,x2,y2 \
  --offline-detect-plate 0 \
  --plate-model /userdata/model/plate_yolov5n_rk3568.rknn \
  --ocr-model /userdata/model/LPRNet_stage3_rk3568_fp16_git.rknn \
  --ocr-keys /userdata/model/ocr_keys_lprnet.txt \
  --ocr-channel-order bgr \
  --ocr-crop-mode match \
  --ocr-resize-mode letterbox \
  --ocr-resize-kernel nn \
  --ocr-preproc none \
  --ocr-ctc-diag 1
```

这样可以把变量压缩到只剩：

- OCR 模型
- crop 方式
- resize 方式
- decode

### Step 6：重新验证训练侧离线推理

在 PC 侧至少做三组离线验证：

1. 官方权重 + 你自己的训练/验证集
2. 微调前权重 + 板端 dump 出来的 OCR 输入
3. 微调后权重 + 板端 dump 出来的 OCR 输入

如果你能把板端 dump 的 `ocr_input.ppm` 转成训练侧可读样本，就可以非常直观看出：

- 模型到底是被板端输入打坏
- 还是权重本身就已经退化

## 7. 建议直接修的代码问题

### 7.1 必修

- 修复 `phase_train=True` 却让模型走 `eval()` 的 bug
- 将 train/test 路径拆分
- 训练时打印 `model.training`
- 训练后导出前，先在 PC 上跑固定样本对照官方权重

### 7.2 强烈建议

- 把验证集改成板端分布一致的数据
- 从板端 dump OCR 输入，回灌成离线验证集
- 在训练中加入 `letterbox` 或接近板端的 crop 生成方式
- 训练时记录每轮验证的长度分布，而不仅是整牌 acc

### 7.3 可选优化

- 降低微调学习率，例如 `1e-4` 或 `5e-5`
- 先冻结 backbone，只训练后几层
- 给省份汉字和尾部字符分别统计 acc
- 使用 `blank_top1_mean` 作为额外监控指标

## 8. 我对当前问题的最终判断

高置信结论：

1. 当前线上差距的主因更偏向微调权重退化，而不是板端参数错误。
2. 训练脚本里的 `train/eval` 模式 bug 是最需要先修的地方。
3. 训练样本分布与板端真实 OCR 输入分布不一致，是第二个主要问题。
4. 当前离线评估方式存在明显乐观偏差，不能拿它证明模型已经变好。

## 9. 推荐执行顺序

按下面顺序做，不要跳步：

1. 修训练模式 bug。
2. 拆 train/val/test。
3. 重训一个最小对照模型。
4. 板端开 `--ocr-ctc-diag 1`，对比官方和微调模型的 `blank_top1`。
5. 跑 `run_ocr_root_cause.sh`。
6. 跑 `ocr_demo_compare.py` 做同输入对照。
7. 如果确认模型本身退化，再去改数据分布和 fine-tune 策略。

## 10. 验收标准

在你准备认为“微调成功”之前，至少要满足：

- PC 验证集上优于或不低于官方权重
- 板端 live 实跑不再大量出现短串
- `blank_top1_mean` 不明显高于官方模型
- 同一批 dump OCR 输入上，app 与官方 demo 输出一致率高
- ROI/letterbox 路径下的 metrics 不低于 fixed/stretch 路径

如果上述条件没有同时满足，不建议把微调模型替换官方模型上线。
