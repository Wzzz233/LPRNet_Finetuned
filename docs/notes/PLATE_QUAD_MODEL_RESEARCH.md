# 本机可用车牌几何模型调研与实测

## 1. 调研范围

这次同时做了两件事：
- 网络搜索 / GitHub / HuggingFace 调研当前可下载的车牌检测/角点候选
- 本机实际下载/加载/跑样本，判断谁在当前环境里真能用

同时尝试了本地 `browser-cli`，结果是：
- CLI 已安装：`/root/.nvm/versions/node/v24.14.0/bin/browser-cli`
- 但当前缺少浏览器扩展连接，CLI 只能起 daemon，不能真正抓网页 DOM
- 所以这次网络调研主要还是靠公开 API / HTTP 检索完成

---

## 2. 本机当前实际可用候选

### A. obb_best_local
路径：
- `/home/wzzz/LPRNet/external_detectors/obb_best.pt`

类型：
- `task=obb`

优点：
- 真 OBB，能直接输出四点 quad
- 对非 CCPD plain_plate 和生成绿牌至少有实际检出
- 当前环境下无需额外下载即可直接跑

缺点：
- 在 CCPD 真几何参考上的平均几何贴合度一般
- 我们人工验收已证明：它给 git_plate 补出来的 quad 在板端一致链路下会出现车牌不完整

### B. chinese_anpr_detect_local
路径：
- `/home/wzzz/LPRNet/external_detectors/chinese_anpr_yolov8_last.pt`

类型：
- `task=detect`

优点：
- 在 CCPD 参考组上框很稳
- 轴对齐框与 CCPD GT 的重叠度明显高于 obb_best

缺点：
- 不是 OBB，没有四点
- 对 git_plate/plain 与 targeted_green_plain 基本 0 检出
- 更像“CCPD 风格专用 detector”，不是跨域几何补标器

### C. hf_koushim_detect
来源：
- HuggingFace `Koushim/yolov8-license-plate-detection`
- 已下载到：`/home/wzzz/LPRNet/external_detectors/hf_koushim_best.pt`

类型：
- `task=detect`

优点：
- 能直接下载、能加载、能跑
- 在 CCPD 参考组上表现接近 chinese_anpr_detect_local

缺点：
- 同样只是普通框，不是 OBB
- 对当前非 CCPD plain_plate 也几乎全灭

---

## 3. 本机客观实测结果

实测脚本：
- `/home/wzzz/LPRNet/vet_plate_quad_candidates.py`

结果汇总：
- `/home/wzzz/LPRNet/artifacts/plate_quad_vetting/summary.json`

### 3.1 CCPD 真几何参考组（12 张）
这里用 CCPD 文件名中的真 quad 当参考，做一个“预测几何 vs GT” 的客观对比。

#### obb_best_local
- task = obb
- detection_rate = 1.0
- mean_iou_like_vs_gt = 0.4632
- mean_warp_sharpness = 4833.30

#### chinese_anpr_detect_local
- task = detect
- detection_rate = 1.0
- mean_iou_like_vs_gt = 0.7993
- mean_warp_sharpness = 4186.58

#### hf_koushim_detect
- task = detect
- detection_rate = 1.0
- mean_iou_like_vs_gt = 0.7985
- mean_warp_sharpness = 4141.72

解释：
- 如果只看“在 CCPD 上框对不对”，两个 detect 模型明显优于 obb_best
- 但 detect 模型没有四点，只能把框当矩形四角来凑，不是真透视角点模型

### 3.2 非 CCPD plain / 生成绿牌样本

#### obb_best_local
- git_plate plain 8/8 检出
- targeted_green_plain 6/8 检出
- 是唯一一个在当前 plain 非 CCPD 样本上真正有工作能力的候选

#### chinese_anpr_detect_local
- git_plate plain 0/8
- targeted_green_plain 0/8

#### hf_koushim_detect
- git_plate plain 0/8
- targeted_green_plain 0/8

解释：
- 这两个 detect 模型基本只能吃 CCPD 风格，不具备跨域补标能力
- obb_best 虽然不够准，但至少是当前环境里唯一“能在 plain 样本上产出 quad”的可用候选

---

## 4. 和人工验收结合后的真正结论

用户人工验收已经给出非常关键的结论：
- CRPD 当前框/quad 错误，板端处理后甚至没有车牌内容
- git_plate 当前后补 quad 处理后车牌不完整

所以不能只看“模型能不能输出 quad”，还要看：
- 经过板端一致链路后，结果是否真的可用

把客观统计和人工验收一起看，得到的结论是：

### 当前本机“最强可用”的定义要拆成两种

#### 1) 如果问：谁最会在 CCPD 风格图上给出稳定框？
答案：
- `chinese_anpr_detect_local`
- `hf_koushim_detect`

这两者都比 `obb_best` 更贴 CCPD GT。

但问题是：
- 它们不是 OBB
- 它们对当前 non-CCPD plain 样本基本没法用

#### 2) 如果问：谁是当前本机唯一还能给 non-CCPD plain 样本补出 quad 的候选？
答案：
- `obb_best_local`

但它只能算：
- 当前环境里唯一能工作的“伪几何候选”
- 不能算“已经足够可靠的最强模型”

也就是说：
- 它是当前最能干活的
- 但不是当前最值得信任的

---

## 5. 最终判断（必须下结论）

### 结论一句话
当前本机没有找到一个“既能跨域吃 non-CCPD plain，又能稳定给出正常框和透视角”的真正强模型。

### 更细一点
- 在本机现成可跑候选里，`obb_best.pt` 仍然是唯一能给 non-CCPD plain 样本补 quad 的模型
- 但结合人工验收，它还不够可靠，不能直接视为高质量几何标注器
- 两个 detect 候选在 CCPD 上更准，但对 plain 非 CCPD 几乎没用，而且本身也不产真实四点

所以当前最符合事实的说法是：

`obb_best.pt` 是“当前本机最强可用的跨域补几何候选”，但还不是“可直接信任的最终模型”。

---

## 6. 后续建议

### 最现实路线
1. 保留 `obb_best.pt` 作为低成本补一版的候选
2. 但它产出的数据一律视为 `pseudo_geom`
3. 必须继续人工抽检 / 板端链路验收
4. 不要再把它补出来的 quad 直接等同于真几何

### 如果目标是找“真正强的框+四点模型”
下一步应当优先找：
- 明确做 license plate corner/keypoint detection 的模型
- 或支持 pose/keypoint 的 plate detector
- 而不是继续在普通 detect/obb 模型里碰运气

但就当前本机已能直接用、已实测过的候选来说，还没有发现比 `obb_best.pt` 更强、同时又真正能跨域工作的模型。
