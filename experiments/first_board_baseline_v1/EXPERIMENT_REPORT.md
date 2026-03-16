# First Board Baseline V1

## 摘要
- 实验目录：`experiments/first_board_baseline_v1`
- 验证集整牌准确率：`0.941408`
- 验证集首字准确率：`0.989220`
- 测试集整牌准确率：`0.137067`
- 测试集首字准确率：`0.848178`
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
- 初始化权重：`./weights_red_stage3/Final_LPRNet_model.pth`
- 学习率：`0.001`，调度：`4,8,12,14,16`，epoch：`15`
- batch：train=`64` test=`120`
- 训练损失：首 epoch `AvgLoss=0.4116`，末 epoch `AvgLoss=0.0312`

## 表现
- 验证集：整牌 `94.14%`，首字 `98.92%`，字符 `98.55%`
- 测试集：整牌 `13.71%`，首字 `84.82%`，字符 `51.00%`
- 真实板图锚点：整牌 `0.00%`，首字 `0.00%`，blank-top1 均值 `0.6111`
- 锚点样本：`ocrin_0021_f000042.ppm` gt=`京N8P8F8` pred=`N8P8F8` top5=[['藏', 0.10135094076395035], ['宁', 0.09914412349462509], ['青', 0.09309607744216919], ['吉', 0.08537047356367111], ['琼', 0.08269151300191879]]

## 泛化与过拟合判断
- 验证到测试存在明显泛化差距：整牌差距 0.8043，首字差距 0.1410。 这更像部署域差或训练分布偏置，而不只是普通过拟合。
- 训练损失持续下降但测试集仍明显低于验证集时，更应优先怀疑部署域差、真实板图样本不足或训练分布偏置，而不是单纯增加 epoch。

## 产物
- 最终权重：`experiments/first_board_baseline_v1/weights/Final_LPRNet_model.pth`
- `experiments/first_board_baseline_v1/weights/LPRNet_stage3_rk3568_fp16.onnx`
- `experiments/first_board_baseline_v1/weights/LPRNet_stage3_rk3568_fp16.rknn`
- `experiments/first_board_baseline_v1/val_metrics.json`
- `experiments/first_board_baseline_v1/test_metrics.json`
- `experiments/first_board_baseline_v1/board_anchor_metrics.json`
- `experiments/first_board_baseline_v1/train.log`
