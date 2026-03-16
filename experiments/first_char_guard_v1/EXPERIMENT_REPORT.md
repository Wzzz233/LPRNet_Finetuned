# First Char Guard V1

## 摘要
- 实验目录：`experiments/first_char_guard_v1`
- 验证集整牌准确率：`0.929717`
- 验证集首字准确率：`0.992850`
- 测试集整牌准确率：`0.170296`
- 测试集首字准确率：`0.755659`
- 真实板图锚点整牌准确率：`1.000000`
- 真实板图锚点首字准确率：`1.000000`

## 数据
- 训练标签：`./prepared_labels/ccpd2019/train_labels.txt`
- 验证标签：`./prepared_labels/ccpd2019/val_labels.txt`
- 测试标签：`./prepared_labels/ccpd2019/test_labels.txt`
- 真实板图锚点：`./board_anchor_labels.txt`
- 训练样本数：`100000`
- 验证样本数：`99996`
- 测试样本数：`141982`
- 切分重叠：train/val=`0` train/test=`0` val/test=`0`

## 处理与训练设置
- 数据模式：`ccpd_board`
- OCR 输入：`channel=bgr` `crop=match` `resize=letterbox` `kernel=nn` `preproc=none` `min_occ=0.90`
- 初始化权重：`./experiments/crop_aligned_v1/weights/Final_LPRNet_model.pth`
- 学习率：`0.0001`，调度：`3,6,8`，epoch：`8`
- batch：train=`64` test=`120`
- 裁切扰动：`mode=jitter_refine` `prob=0.85` `jx=0.06` `jy=0.12` `min_iou=0.75`
- 省份重平衡：`inv_sqrt`
- 真实锚点采样权重：`768`
- 首字辅助损失：`weight=0.40` `time_steps=6`
- checkpoint 选择代理集：`first 5000 val samples`
- 训练损失：首 epoch `AvgLoss=0.6214`，末 epoch `AvgLoss=0.0989`

## 表现
- 验证集：整牌 `92.97%`，首字 `99.28%`，字符 `98.02%`
- 测试集：整牌 `17.03%`，首字 `75.57%`，字符 `56.05%`
- 真实板图锚点：整牌 `100.00%`，首字 `100.00%`，blank-top1 均值 `0.5000`
- 锚点样本：`ocrin_0021_f000042.ppm` gt=`京N8P8F8` pred=`京N8P8F8` top5=[['京', 0.9998797178268433], ['豫', 0.00011823759996332228], ['苏', 9.707342769615934e-07], ['闽', 4.3663717974595784e-07], ['浙', 3.664643202228035e-07]]

## 泛化与过拟合判断
- 验证到测试存在明显泛化差距：整牌差距 0.7594，首字差距 0.2372。 这更像部署域差或训练分布偏置，而不只是普通过拟合。
- 训练损失持续下降但测试集仍明显低于验证集时，更应优先怀疑部署域差、真实板图样本不足或训练分布偏置，而不是单纯增加 epoch。

## 产物
- 最终权重：`experiments/first_char_guard_v1/weights/Final_LPRNet_model.pth`
- `experiments/first_char_guard_v1/weights/LPRNet_stage3_rk3568_fp16.onnx`
- `experiments/first_char_guard_v1/weights/LPRNet_stage3_rk3568_fp16.rknn`
- `experiments/first_char_guard_v1/val_metrics.json`
- `experiments/first_char_guard_v1/test_metrics.json`
- `experiments/first_char_guard_v1/board_anchor_metrics.json`
- `experiments/first_char_guard_v1/train.log`
