# 外部 OBB 检测器试跑结论

## 已下载并实测的候选

### 1. Chinese-ANPR-YOLOv8
- 权重：`/home/wzzz/LPRNet/external_detectors/chinese_anpr_yolov8_last.pt`
- 类型：普通 detect
- 结论：
  - 对 CCPD 绿牌参考组很稳
  - 对非 CCPD 整牌图几乎不可用
  - 不适合作为非 CCPD 自动补框主力

### 2. obb_best.pt
- 权重：`/home/wzzz/LPRNet/external_detectors/obb_best.pt`
- 类型：YOLOv8 OBB
- 结论：
  - 对非 CCPD plain plate 有一定可用性
  - 稳定性不是完美，但足以作为“低成本先补一轮”的工具
  - 因为它能直接输出 OBB / quad，所以最终被选为当前非 CCPD 自动补几何标注的执行模型

---

## 当前项目中的实际采用结论

当前默认外部补几何检测器：
- `obb_best.pt`

当前不采用：
- `chinese_anpr_yolov8_last.pt`

原因：
- 后者更像 CCPD 风格检测器，离开 CCPD 风格后泛化不足
- 前者虽然不是强到可以盲信，但足以先做一轮成功/失败分流式自动补框
