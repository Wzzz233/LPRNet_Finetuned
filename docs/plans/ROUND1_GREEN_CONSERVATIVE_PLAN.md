# 第一轮训练方案：保守加入绿牌

## 1. 目标

这轮目标不是追求绿牌立刻拉满，而是：
- 尽量保住蓝牌识别率
- 只小心引入一部分绿牌能力
- 不在第一轮就让 special 干扰主干

因此本轮采用最稳方案：
- 以当前蓝牌强基线权重为初始化
- 冻结 backbone
- 只训练更靠后的识别层
- 训练集以 `normal7` 为主，只少量加入 `green8`
- 暂不引入 `special` 训练样本
- 验证/测试仍保留全量，以便观察是否掉蓝、以及绿牌是否开始起效

---

## 2. 为什么选“冻结 backbone”而不是更激进方案

原因很简单：
- 你当前最担心的是蓝牌能力回退
- backbone 承载了大量现有蓝牌视觉特征
- 第一轮只想让模型学会“开始容纳绿牌”，而不是重塑整个视觉主干

所以最稳做法是：
- 保留已有视觉主干
- 只让后面的识别部分去适配绿牌分布

这比直接全网微调更稳。

---

## 3. 训练清单

### 本轮训练 manifest
- `/home/wzzz/LPRNet/manifests/unified_manifest_v3_round1_green_conservative.csv`

### 汇总
- `/home/wzzz/LPRNet/manifests/unified_manifest_v3_round1_green_conservative.summary.json`

### 训练集构成
- 蓝牌 train：`278478`
- 绿牌 train：`17769`
- special train：`0`

### 绿牌 train 具体来源
- `ccpd2020_green`：`5769`
- `targeted_green_missing_18`：`2000`
- `targeted_green_missing_18_pseudo_geom`：`2000`
- `cblprd_pseudo_geom`：`4000`
- `cblprd_plain`：`4000`

### 验证/测试
- 保留原有全部 val/test/eval，不裁掉

---

## 4. train_LPRNet.py 已加的安全开关

已新增：
- `--freeze_backbone true`

作用：
- 冻结 `lprnet.backbone`
- 只训练后部可学习层与辅助头

这个开关已经验证可正常解析。

---

## 5. 推荐首轮训练命令

```bash
cd /home/wzzz/LPRNet
./.conda/bin/python train_LPRNet.py \
  --cuda true \
  --data_mode manifest \
  --manifest /home/wzzz/LPRNet/manifests/unified_manifest_v3_round1_green_conservative.csv \
  --pretrained_model /home/wzzz/LPRNet/experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth \
  --freeze_backbone true \
  --max_epoch 6 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0003 \
  --lr_schedule 3 5 \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --first_char_aux_weight 0.4 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --save_folder /home/wzzz/LPRNet/experiments/round1_green_conservative_freeze_backbone
```

---

## 6. 为什么这条命令稳

主要是 4 个保守点：

1. 从当前最好权重起步
不是从头训。

2. 冻结 backbone
最大限度保住蓝牌原有视觉能力。

3. 低学习率
防止识别头适配时把原有分布拉坏。

4. 绿牌只小比例注入
这轮只让模型“先见过、先开始学”，不让它立刻主导分布。

---

## 7. 本轮看什么指标

优先顺序：
1. 蓝牌 val/test 是否明显回退
2. 绿牌是否开始出现可见提升
3. 是否出现对 special 的明显误伤

如果蓝牌稳、绿牌有起色，第二轮再考虑：
- 解冻部分 backbone
- 继续增加 green8 比重
- 再逐步接入更多 special
