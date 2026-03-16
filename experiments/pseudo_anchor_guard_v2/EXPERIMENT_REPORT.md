# pseudo_anchor_guard_v2

## 摘要
- 实验目录：`experiments/pseudo_anchor_guard_v2`
- 验证集整牌准确率：`0.875055`
- 验证集首字准确率：`0.930217`
- 测试集整牌准确率：`0.093787`
- 测试集首字准确率：`0.340959`
- 真实板图锚点整牌准确率：`1.000000`
- 真实板图锚点首字准确率：`1.000000`
- 伪锚点验证集整牌准确率：`0.722222`
- 伪锚点验证集首字准确率：`0.805556`

## 数据
- 训练标签：`./prepared_labels/ccpd2019/train_labels.txt`
- 验证标签：`./prepared_labels/ccpd2019/val_labels.txt`
- 测试标签：`./prepared_labels/ccpd2019/test_labels.txt`
- 真实板图锚点：`./board_anchor_labels.txt`
- 伪锚点训练集：`./experiments/pseudo_anchor_guard_v2/labels/pseudo_anchor_train_labels.txt`
- 伪锚点验证集：`./experiments/pseudo_anchor_guard_v2/labels/pseudo_anchor_val_labels.txt`
- 训练样本数：`100000`
- 验证样本数：`99996`
- 测试样本数：`141982`
- 切分重叠：train/val=`0` train/test=`0` val/test=`0`
- 伪锚点去重：train/val=`0` train/existing=`0` val/existing=`0`
- 伪锚点样本数：train=`144` val=`36`

## 处理与训练设置
- 数据模式：`ccpd_board`
- OCR 输入：`channel=bgr` `crop=match` `resize=letterbox` `kernel=nn` `preproc=none` `min_occ=0.90`
- 初始化权重：`./experiments/first_char_guard_v1/weights/Final_LPRNet_model.pth`
- 学习率：`0.00002`，调度：`2,4`，epoch：`4`
- batch：train=`64` test=`120`
- 裁切扰动：`mode=jitter_refine` `prob=0.85` `jx=0.06` `jy=0.12` `min_iou=0.75`
- 省份重平衡：`inv_sqrt`
- 真实锚点采样权重：`768`
- 伪锚点采样权重：`16`
- 首字辅助损失：`weight=0.40` `time_steps=6`
- checkpoint 选择代理集：`first 5000 val samples`
- 训练损失：首 epoch `AvgLoss=0.1488`，末 epoch `AvgLoss=0.0942`

## 表现
- 验证集：整牌 `87.51%`，首字 `93.02%`，字符 `96.97%`
- 测试集：整牌 `9.38%`，首字 `34.10%`，字符 `47.99%`
- 真实板图锚点：整牌 `100.00%`，首字 `100.00%`，blank-top1 均值 `0.5000`
- 锚点样本：`ocrin_0021_f000042.ppm` gt=`京N8P8F8` pred=`京N8P8F8` top5=[['京', 0.9998908042907715], ['豫', 0.00010193453636020422], ['苏', 5.809846697957255e-06], ['浙', 1.0495616606931435e-06], ['沪', 1.3530460307720205e-07]]
- 伪锚点验证集：整牌 `72.22%`，首字 `80.56%`，字符 `93.65%`
- 易混省份：粤: n=5 exact=100.00% first=100.00% | 晋: n=1 exact=0.00% first=0.00% | 黑: n=1 exact=0.00% first=0.00% | 苏: n=5 exact=80.00% first=100.00% | 浙: n=5 exact=100.00% first=100.00%

## 泛化与过拟合判断
- 验证到测试存在明显泛化差距：整牌差距 0.7813，首字差距 0.5893。 这更像部署域差或训练分布偏置，而不只是普通过拟合。
- 训练损失持续下降但测试集仍明显低于验证集时，更应优先怀疑部署域差、真实板图样本不足或训练分布偏置，而不是单纯增加 epoch。
- 伪锚点验证集重点关注：粤: n=5 exact=100.00% first=100.00% | 晋: n=1 exact=0.00% first=0.00% | 黑: n=1 exact=0.00% first=0.00% | 苏: n=5 exact=80.00% first=100.00% | 浙: n=5 exact=100.00% first=100.00%

## 产物
- 最终权重：`experiments/pseudo_anchor_guard_v2/weights/Final_LPRNet_model.pth`
- `experiments/pseudo_anchor_guard_v2/val_metrics.json`
- `experiments/pseudo_anchor_guard_v2/test_metrics.json`
- `experiments/pseudo_anchor_guard_v2/board_anchor_metrics.json`
- `experiments/pseudo_anchor_guard_v2/pseudo_anchor_val_metrics.json`
- `experiments/pseudo_anchor_guard_v2/pseudo_anchor_stats.json`
- `experiments/pseudo_anchor_guard_v2/train.log`
