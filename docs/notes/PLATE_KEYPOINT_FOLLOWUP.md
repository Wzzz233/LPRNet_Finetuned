# 车牌四角点 / keypoint / pose 方向二次调研结论

## 1. 这次按用户要求改用的调研方式

- 使用了 `web_search` 直接检索
- 同时确认了本地 `browser-cli` 已安装，但当前 daemon 缺少浏览器扩展连接，仍无法作为完整网页交互检索主力
- 因此这轮“搜索入口”已切换到 `web_search` 为主，本地 CLI 只做环境确认

---

## 2. 这次找到的 keypoint / pose 候选

### 候选 A：theasiko/License-Plate-Corner-Keypoint-Detection
特点：
- 明确就是“四角点回归”项目
- README 说明用 MobileNetV3 回归 8 个角点坐标

问题：
- 仓库里只有 README 和 notebook
- 没有现成权重、没有 release、没有直接可下载 checkpoint
- 当前不能直接在本机拿来推理

结论：
- 方法方向对
- 但当前不是“本机可直接用”的候选

### 候选 B：doansangg/Object-Corner-Detection
特点：
- 明确做 object corner detection，README 里直接写了可用于 license plate detection
- 思路是 corners as keypoints
- README 里提供了一个预训练权重下载链接（Google Drive）

问题：
- 代码是老方案，依赖重，需要额外环境编译 DCNv2
- 预训练权重是越南车牌，不是中国牌
- 当前还没有拿到权重并在本机跑通

结论：
- 是目前最值得继续追的 keypoint 候选
- 但还没达到“本机可直接用”状态

### 候选 C：computervisioneng/train-yolov8-pose-detection-google-colab-license-plate-detection
特点：
- 明确是 YOLOv8 pose 的车牌角点/关键点训练示例
- 方向与我们需求很吻合

问题：
- 仓库只是训练教程
- 没有现成权重，也没有可直接下载 checkpoint

结论：
- 能作为后续自训路线参考
- 不是现成可跑模型

---

## 3. 这轮最重要的事实

目前通过正常搜索能找到的“车牌四角点 / keypoint / pose”项目，大多存在一个共同问题：
- 论文/教程/仓库能找到
- 但现成权重很少
- 或者没有 release
- 或者不能直接在本机落地推理

也就是说：
- 能找到“方法”
- 很难找到“现成可下载且本机立即能用的强权重”

---

## 4. 和上一轮实测结果合并后的结论

当前本机真正满足“能直接跑”的仍然只有：
- `obb_best.pt`（本地已有，真 OBB）
- 两个 detect 模型（CCPD 风格强，但不产四点且跨域差）

而这轮新调研到的 keypoint / pose 项目：
- 方向更对
- 但暂时没有现成可用权重可替换现有方案

所以截至目前：

### 结论一句话
还没有找到一个现成可下载、可在本机直接跑、并且比 `obb_best.pt` 更适合当前 non-CCPD 中国车牌补四角点的 keypoint/pose 模型。

---

## 5. 下一步最值得继续深挖的对象

优先级最高的是：
- `doansangg/Object-Corner-Detection`

原因：
- 它确实是“corner detection”思路，不是普通 detect
- README 明确给了预训练权重下载入口
- 理论上最接近我们想要的“正常框 + 四角点”模型

但需要接受两个现实：
1. 它不是中国车牌专训
2. 它需要额外环境和权重下载验证，成本高于直接跑 YOLO `.pt`

---

## 6. 当前可执行建议

### 路线 A（继续找现成权重）
继续追：
- Google Drive 预训练 keypoint 权重
- HuggingFace / Kaggle / Roboflow 上可导出的 pose / keypoint 车牌模型

### 路线 B（更现实）
如果现成强权重始终找不到，更现实的路线是：
- 直接基于 YOLOv8-pose / YOLO11-pose 自己训四角点模型
- 训练集先从 CCPD 真 quad 出发构造
- 然后用少量非 CCPD 人工校正样本做补充

这条路线虽然不是“捡现成”，但很可能比继续赌外部权重更靠谱。
