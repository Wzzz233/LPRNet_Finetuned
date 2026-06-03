# Police BlueBase CVStrict — 导出前验收清单

## 候选模型
`experiments/police_bluebase_cvstrict_20260603/stageC/best_LRTNet_model.pth`

## 1. PyTorch 原模型复测

- [x] special_v2 val_clean: 1509/1550 = 97.35%
- [x] special_v2 val_hard: 1477/1550 = 95.29%
- [x] real mgC0001J province=蒙: 40/40 = 100%
- [x] real mgC0001J exact: 20/40 = 50%
- [ ] 0 I/O ✅
- [ ] 0 length errors on val_clean ✅
- [ ] 0 jing at pos1 ✅

## 2. ONNX 导出后复测（待执行）

- [ ] ONNX 与原 PyTorch 逐样本对齐（val_clean 1550 样本）
- [ ] ONNX 与 PyTorch 逐样本对齐（val_hard 1550 样本）
- [ ] ONNX 与 PyTorch 逐样本对齐（real mgC0001J 40 样本）
- [ ] 记录差异数量和类型
- [ ] 检查 FixedNorm 是否引入动态 Reshape（之前 embassy 遇到过）

## 3. RKNN 导出后复测（待执行）

- [ ] RKNN 与 ONNX 逐样本对齐
- [ ] RKNN 在 simulator 上跑 real dump 40 帧
- [ ] 记录差异数量
- [ ] 检查 blank_top1 是否符合预期

## 4. 板端 A/B 对比（待执行）

- [ ] v2 official baseline 在板上跑同一批真实帧
- [ ] BlueBase + special_v2 在板上跑同一批真实帧
- [ ] 对比 province、full exact、0/U 混淆
- [ ] 记录 inference time 差异
