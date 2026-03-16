# Crop Aligned V1

## 摘要
- 实验目录：`experiments/crop_aligned_v1`
- 验证集整牌准确率：`0.946648`
- 验证集首字准确率：`0.990420`
- 测试集整牌准确率：`0.129840`
- 测试集首字准确率：`0.588159`
- 真实板图锚点整牌准确率：`0.000000`
- 真实板图锚点首字准确率：`0.000000`

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
- 初始化权重：`./experiments/first_board_baseline_v1/weights/Final_LPRNet_model.pth`
- 学习率：`0.0003`，调度：`4,8,12,14,16`，epoch：`8`
- batch：train=`64` test=`120`
- 裁切扰动：`mode=jitter_refine` `prob=0.85` `jx=0.06` `jy=0.12` `min_iou=0.75`
- 训练损失：首 epoch `AvgLoss=0.0755`，末 epoch `AvgLoss=0.0456`

## 表现
- 验证集：整牌 `94.66%`，首字 `99.04%`，字符 `98.71%`
- 测试集：整牌 `12.98%`，首字 `58.82%`，字符 `50.61%`
- 真实板图锚点：整牌 `0.00%`，首字 `0.00%`，blank-top1 均值 `0.6111`
- 锚点样本：`ocrin_0021_f000042.ppm` gt=`京N8P8F8` pred=`N8P8F8` top5=[['藏', 0.07556901127099991], ['吉', 0.07391538470983505], ['宁', 0.07351440191268921], ['新', 0.07287833839654922], ['桂', 0.07281886786222458]]

## 泛化与过拟合判断
- 验证到测试存在明显泛化差距：整牌差距 0.8168，首字差距 0.4023。 这更像部署域差或训练分布偏置，而不只是普通过拟合。
- 训练损失持续下降但测试集仍明显低于验证集时，更应优先怀疑部署域差、真实板图样本不足或训练分布偏置，而不是单纯增加 epoch。

## 产物
- 最终权重：`experiments/crop_aligned_v1/weights/Final_LPRNet_model.pth`
- `experiments/crop_aligned_v1/val_metrics.json`
- `experiments/crop_aligned_v1/test_metrics.json`
- `experiments/crop_aligned_v1/board_anchor_metrics.json`
- `experiments/crop_aligned_v1/train.log`
