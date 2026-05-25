我主要将 Codex 作为科研与开发过程中的高级辅助工具，而不是替代核心创作的工具。在课题推进中，我会使用 Codex 的联网检索、代码理解、文档处理和结构化分析能力，辅助完成文献调研、方案梳理、代码优化、文档润色和图表绘制等非核心但耗时的环节。

在文献调研阶段，我会让 Codex 帮助我快速整理相关研究方向、提取论文中的关键方法、对比不同技术路线的优缺点，并将零散资料归纳成清晰的阅读笔记。这样可以提高前期信息收集效率，但论文筛选、观点判断和最终研究思路仍由我自己完成。

在方案设计阶段，我会使用 Codex 帮助梳理系统架构、模块划分、实现流程和可能的技术风险。它可以把我的初步想法整理成更清晰的逻辑框架，帮助我发现方案中表述不充分或衔接不严密的地方，但核心方案选择和创新点设计仍由我负责。

在代码开发阶段，我会把 Codex 作为代码审查和优化辅助工具使用，例如让它检查代码结构、定位潜在 bug、优化函数封装、补充注释、改进变量命名或生成测试思路。对于关键算法和核心逻辑，我会先独立完成，再借助 Codex 做规范性和可维护性检查。

在文档写作阶段，我会使用 Codex 对报告、说明书、实验记录等材料进行语言润色、格式统一和逻辑调整，使表达更加准确、连贯、符合学术或工程文档规范。它主要帮助我改善表达质量，而不是替我生成最终观点。

在图表辅助方面，我会使用 Codex 帮助整理实验数据、生成绘图代码、设计表格结构或优化图注说明，使结果展示更加清晰直观。图表所反映的数据来源、实验结论和分析判断仍基于我自己的实验与理解。

总体来说，我使用 Codex 的方式是把它作为提升效率和质量的辅助工具，主要承担资料整理、结构梳理、代码检查、文字润色和图表生成等支持性工作。核心研究判断、方案创新、实验设计和最终结论仍由我独立完成。


# 使用 Hermes Agent 辅助 LPRNet 模型训练调优

> 本文档记录了在整个 LPRNet 绿牌车牌识别项目中，如何利用 Hermes Agent（AI 编程助手）的高级功能加速模型训练、数据生成、实验管理和知识沉淀的全流程方法论。

---

## 目录

1. [背景：为什么需要 AI Agent](#1-背景为什么需要-ai-agent)
2. [Skill 技能系统：将方法沉淀为可复用资产](#2-skill-技能系统将方法沉淀为可复用资产)
3. [内存系统：用户偏好与项目事实的持久化](#3-内存系统用户偏好与项目事实的持久化)
4. [跨会话上下文检索](#4-跨会话上下文检索)
5. [后台任务托管与自动通知](#5-后台任务托管与自动通知)
6. [数据探查与 QA 生成](#6-数据探查与-qa-生成)
7. [系统化实验审计与报告](#7-系统化实验审计与报告)
8. [知识演进：从单次 debug 到可复用技能](#8-知识演进从单次-debug-到可复用技能)
9. [工作流全景：一次典型实验的生命周期](#9-工作流全景一次典型实验的生命周期)

---

## 1. 背景：为什么需要 AI Agent

LPRNet 绿牌车牌识别项目的特点是：

- **高复杂性**：涉及 YOLOv8 检测器（OBB / Pose）、LPRNet 多头 OCR、quad refiner 三个模型系统，各有独立训练流程
- **长周期**：实验跨度数十轮（Stage A → Stage B → E1 ~ E28），每轮涉及数据生成、训练、评测、分析、回归
- **多环节**：数据生成（替换管线、brightness matching、quad 排序陷阱）→ 数据审计 → 训练（GPU/CPU 检测、参数筛查）→ 多维度评测 → 根因分析 → 实验设计
- **容易遗忘**：一个实验的细节（参数、数据配比、已知坑）在 3 天后就会模糊

传统工作方式是：在笔记里手写实验记录 → 人脑记忆参数 → 依赖 bash history 回忆命令。这种方式在项目复杂度超过一定阈值后会彻底不可持续。

Hermes Agent 提供了一个**可交互的、有记忆的、能自主执行操作**的 AI 协作层。不是"问一句答一句"的聊天机器人，而是**可以托管后台任务、执行代码、跨会话回忆、并将经验沉淀为可复用技能**的工程伙伴。

---

## 2. Skill 技能系统：将方法沉淀为可复用资产

### 2.1 什么是 Skill

Skill 是 Hermes Agent 的**程序性知识库**——以 Markdown 文件（含 YAML 元信息）组织的工作流程文档。加载 Skill 后，Agent 能够精确执行该领域的所有步骤、避免已知陷阱。

在 LPRNet 项目中，最终沉淀了 **12 个核心 Skill**：

| Skill 名称 | 覆盖范围 | 吸收的子 skill |
|------------|---------|----------------|
| `lprnet-board-pipeline` | 板端推理管线、OCR dump 评测、透视矫正链路审计 | 8 个 |
| `lprnet-data-generation` | 合成数据生成（edgefit/替换管线）、quad 排序、亮度匹配、省域控制 | 12 个 |
| `lprnet-training-evaluation` | 训练启动检查、GPU 检测、三类评测、BOM 编码陷阱 | 10 个 |
| `lprnet-multihead-architecture` | 多头架构、family adapter、pos0 头、family-aware 解码 | 7 个 |
| `lprnet-quad-refiner` | OBB→quad 精修、V1→V2 顺序消融、RKNN 导出、板端 C 集成 | 9 个 |
| `lprnet-true-quad-pose-training` | YOLOv8n-Pose 数据集构建、训练、eval、导出 | — |
| `lprnet-cluster-analysis` | 三个 cluster 失败模式诊断框架 | 8 个 |
| `lprnet-firstchar-province` | 省份崩溃诊断、退化训练、混淆对分析 | 7 个 |
| `lprnet-three-stage-curriculum-training` | 三阶段课程训练方法论 | — |
| `lprnet-green-systematic-experimentation` | 系统化实验设计流程 | — |
| `lprnet-manifest-batch-pipeline` | Manifest 批量构建 | — |
| `lprnet-board-pipeline` 下含 reference 文件 | 板端亮度差异、occ 阈值绕行、轨迹融合等深度分析 | — |

### 2.2 Skill 的实际工作方式

当启动一项任务时（例如"检查 E12 训练的 GPU 占用"），Agent 自动加载 `lprnet-training-evaluation` Skill，然后**精确按照 Skill 中的步骤执行**：

```
Skill 中记载的步骤：
1. 验证 --cuda True 在启动命令中
2. 快速 CUDA 检查：python3 -c "import torch; print(torch.cuda.is_available())"
3. 等待 10-15 秒后 nvidia-smi 检查 GPU-Util > 0%
4. 如果 GPU-Util 一直为 0%，说明命令缺少 --cuda True
5. 检查 train.log 第一行是否显示 device=cuda:0
```

Agent**不会**猜测步骤——它按照 Skill 执行，每个步骤都有实际命令和预期的输出信号。

### 2.3 Skill 的演进：从一次 debug 到永久资产

Skill 不是一次性写好的。它经历了一条**知识演进链**：

```
[2026-04-30] 第一次训练启动失败：GPU 占用为 0%
  → 发现根因：--cuda 默认 False，但脚本中没写
  → 修正：在 lprnet-training-evaluation Skill 中新增 "训练启动检查清单"

[2026-05-01] 第二次 GPU 挂掉：nvidia-smi 显示 Python 进程但 GPU-Util=0%
  → 发现根因：CUDA 错误 → GPU 热节流
  → 修正：在 Skill 中新增 "CUDA 崩溃恢复" 章节

[2026-05-03] 发现实验目录只有 best.pth，没有 train.log
  → 发现根因：训练跑在 CPU 上且 shell redirect 失败
  → 修正：在 Skill 中新增 "Post-Mortem: 红绿灯检查清单"
```

第一次犯的错误被记入 Skill → 第二次不会再犯 → 第三次即使换人也能规避。

### 2.4 Skill 的版本控制

每个 Skill 有 `version` 和 `author` 字段，修改历史通过 `skill_manage(action='patch')` 记录。关键修正（如板端透视矫正链路的发现）有日期标注：

```
版本 2.1.0（2026-05-01）:
- 重大纠错：板端确实有透视矫正（warp_quad_to_rect_rgb888）
- 此前版本错误声称 "板端不做透视矫正"
- 根据 ARM C 源代码确认
```

---

## 3. 内存系统：用户偏好与项目事实的持久化

### 3.1 用户 Profile 记忆

Agent 维护了一个**用户偏好数据库**，记录项目中形成的工程原则：

| 记忆 | 来源 | 影响 |
|------|------|------|
| "不要直接甩 JSON，要文字汇报和数据表格" | 用户直接纠正 | 所有实验报告转为 Markdown 表格 + 文字分析 |
| "不允许在没有数据支撑的情况下进行结论分析" | 用户要求 | 每个结论必须有实际测量数据 |
| "实验执行必须按预定顺序全部跑完才能下结论" | 用户纠正 | 禁止中间分析，V2a→V2b→V2c 全跑完再下结论 |
| "必须先出视觉 QA 再推进训练/结论" | 用户纠正 | 所有数据生成后先出 contact sheet → 用户确认 → 再训练 |
| "只有明确下令后才可启动训练/评测/排查" | 用户限制 | Agent 不主动发起训练，只按要求执行 |
| "替换图的文字方向必须人工确认再提交" | 两次坑后确立 | CCPD quad 排序双重检查 + contact sheet 可视化 |
| "Quad Refiner 实验已全部完成，最终不上板" | 实验结论 | 不再建议 refiner 相关实验 |

### 3.2 工程事实记忆

环境层面的持久化事实（tool quirks、数据路径、模型位置）：

| 事实 | 用途 |
|------|------|
| 板端 YOLOv8 OBB 检测器单统一模型路径 | 每次提及检测器时定位正确权重 |
| CCPD quad 排序是 [BR, BL, TL, TR]，不要用 `order_quad_points()` | 生成替换数据时避免文字反转 |
| manifest 字段名必须是 `quad_1x..quad_4y` 而非 `x1..y4` | 训练加载器能正确读取 |
| 训练 manifest 编码必须用 `utf-8` 不能用 `utf-8-sig` | 避免 BOM 导致 KeyError: 'img_path' |
| `--ocr-preproc` manifest 字段会覆盖 CLI 参数 | 数据生成时设置 `ocr_preproc=none` 让 CLI 生效 |
| 训练 conda 环境在 `/home/wzzz/LPRNet/.conda/` | 所有训练命令的正确 python 路径 |

### 3.3 记忆的作用方式

每次会话开始时，Agent 自动加载用户 profile 和记忆。这意味着：

- 用户不需要重复说"不要给我 JSON"
- 用户不需要每个 session 重新解释 CCPD quad 排序规则
- Agent 遇到新的数据生成任务时，自动记得 "要出 QA 图给用户确认"
- Agent 在做实验设计时，自动回避已被结论否定的方向（如 refiner、bundled ablation）

**这是传统聊天式 AI 做不到的**——每次对话都是全新开始，没有记忆持久化。

---

## 4. 跨会话上下文检索

### 4.1 Session Search：过去 30 轮实验的实时召回

项目经历了数十轮实验（A→B→E1~E28），每轮的参数、数据配比、结论散布在不同的会话中。Agent 的 `session_search` 功能允许**通过关键词瞬间回溯**：

```bash
# 用户问："我们之前 E2 平衡实验用了什么数据配比？"
# Agent 内部执行：
session_search("E2 balanced extreme data ratio")

# 返回：
# Session 2026-05-03: "E2从E1 best出发，extreme 42.3% via 10× replication"
# 省域 balance: inv_sqrt
# blue_crpd recovered to 64.30% (vs original 65.25%)
```

不需要翻笔记本、不需要翻 train.log——直接召回相关会话的内容摘要。

### 4.2 关键场景：避免重复造轮子

```
用户："我觉得可以试试OBB quad替换数据的方案"
Agent 内部：
  1. session_search("B2-C OBB") 
  2. 发现已经做过：B2-C_OBB 实验阴性结果——OBB quad 训练不能改善 cluster3
  3. 回复："这个方案已经在 B2-C_OBB 实验验证过..."
```

无需重新论证，直接引用历史结论。

---

## 5. 后台任务托管与自动通知

### 5.1 长时间训练的托管

LPRNet 训练一次通常 1-4 小时。传统工作方式是：打开终端 → 运行命令 → 坐在工位等 → 时不时看一下。Hermes Agent 提供了**后台任务托管**：

```python
terminal(
  command="""
    cd /home/wzzz/LPRNet && export PYTHONPATH=/home/wzzz/LPRNet/src
    bash scripts/train/run_green_e12_pose_replace_append_stage2.sh
  """,
  background=True,
  notify_on_complete=True,
  timeout=7200
)
```

执行后：
1. 训练在后台启动
2. 用户可以退出会话（关闭终端）
3. Agent 系统监控进程状态
4. 训练完成后**自动通知用户**

### 5.2 Cron 定时任务：课后评测自动化

训练完成后，通常需要跑一组评测脚本评测结果。通过 cron job 预定自动化流程：

```python
cronjob(
  action='create',
  schedule='30m',   # 30 分钟后执行
  prompt='''检查 E18A 训练是否完成，如果完成了：
  1. 跑 cluster2/cluster3 板端 dump 评测
  2. 跑标准评测套件的 6 个 benchmark
  3. 汇总结果到表格
  4. 通知用户''
)
```

### 5.3 价值：从"守着等"到"完成后自动汇报"

在没有后台托管前：
- 训练启动后用户不能离开
- 如果有 crash 只能等下次回来才发现
- 时间估算不准就浪费 GPU 空闲时间

有了后台托管：
- 训练启动 → 退出 → 收到通知 → 回来查看结果
- 即使半夜完成，第二天早上看到的就是完整评测报告

---

## 6. 数据探查与 QA 生成

### 6.1 编写 Python 脚本进行数据审计

Agent 可以自主编写并执行 Python 脚本进行数据质量审计：

```python
# 典型审计：检查替换数据的文字是否正确
execute_code("""
from hermes_tools import read_file, terminal

# 1. 随机检查 20 张替换图像的 94×24 gray3
# 2. 通过训练管线处理，显示文字
# 3. 目测每张图的文字方向是否正确
# 4. 输出 contact sheet

# 结果：发现文字反转（order_quad_points bug）
# → 阻止了一次无效训练
""")
```

### 6.2 系统化 QA 图片生成

Agent 按照 Skill 中的 QA 协议，自动生成 contact sheet：

```
1. 读取 manifest 中的 12 个样本
2. 对每个样本生成：
   - 原图缩略图
   - GT quad 94×24 gray3
   - 替换后全图
   - Noisy quad 94×24 gray3
3. 拼接 3×4 网格
4. 输出到 /mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/
5. 通知用户查看
```

### 6.3 价值

数据生成中的异常（文字反转、亮度不匹配、编码问题）通常在 QA 环节就被截住，不会浪费 GPU 时间跑一次无效训练。

---

## 7. 系统化实验审计与报告

### 7.1 多维度评测套件的自动执行

每次训练后，Agent 按 Skill 中的"评测套件模式"自动跑 6 个 benchmark：

```bash
# 1. 真实测试集
eval_green8_metrics_only.py --manifest real_test.csv

# 2. 替换数据 val
eval_green8_metrics_only.py --manifest replacement_val.csv

# 3. Cluster2 板端 dump
eval_gray3_board_dump.py --csv cluster2_wsl.csv

# 4. Cluster3 板端 dump
eval_gray3_board_dump.py --csv cluster3_wsl.csv

# 5. pos_ocr_dump
# 6. pos_ocr_dump_2
```

### 7.2 实验报告自动生成

所有结果汇总为 Markdown 表格 + 文字分析，**不甩 JSON**（这是用户明确的偏好）。

### 7.3 单变量归因

Agent 的 Skill 中内置了"顺序消融"方法论。当设计实验时，自动遵循：

```
✅ 正确模式：
  V2a: 只改采样 → 测量 → 判定
  V2b: 保留 V2a + offset head → 测量 → 判定
  V2c: 保留 V2b + warp loss → 测量 → 判定

❌ 错误模式（Agent 会主动拦截）：
  一次改三个东西 → 无法归因
```

---

## 8. 知识演进：从单次 debug 到可复用技能

这是整个方法论中**最核心**的环节。

### 8.1 知识演进链

```
┌─────────────────────────────────────────────────┐
│ Step 1: 遇到问题                                  │
│ 例："训练启动后 GPU-Util 一直是 0%"                │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│ Step 2: 在当次会话中 debug 解决                    │
│ 根因：--cuda True 遗漏                             │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│ Step 3: 将诊断过程写入 Skill                      │
│ 动作：skill_manage(action='patch', ...)          │
│ 在 lprnet-training-evaluation 中新增              │
│ "训练启动检查清单"章节                             │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│ Step 4: 后续所有 session 自动加载该 Skill          │
│ 下次启动训练 → Skill 中明确步骤：                  │
│ "1. 验证 --cuda True  2. nvidia-smi 检查..."     │
│ → 不会再犯同一错误                                │
└─────────────────────────────────────────────────┘
```

### 8.2 真实演进案例

#### 案例 1：板端透视矫正链路（2026-05-01）

```
初始 Skill 记录："板端不做透视矫正，训练端做"
  → 用户纠正："板端有透视矫正，看 ARM 代码"
  → Skill patch：删除错误记录，替换为 ARM C 代码确认的完整 pipeline
  → 新增：板端 warp_quad_to_rect 实现细节、gray3 系数差异
  → 新增：训练 vs 板端 8 步一致性对照表
  → 新增：发现过程章节（"发现过程和验证方法"）
```

#### 案例 2：CCPD quad 排序陷阱（2026-05-03/04）

```
第一次：替换数据文字反转
  → 引入 order_quad_points() 修复
  → 记录入 data-generation Skill

第二次（同一天）：仍然反转
  → 发现 order_quad_points() 本身在 CCPD 输入上逻辑错误
  → Skill patch：详细说明为什么 order_quad_points() 会错误交换 BL/TR
  → 新增：CCPD 的 [BR, BL, TL, TR] → [TL, TR, BR, BL] 显式重排代码
  → 新增：验证方法（如何从生成的图片判断文字方向）
  → 新增：历史时间线（三次踩坑记录）
```

#### 案例 3：Manifest 字段命名（2026-05-04）

```
问题：替换数据训练后 exact=0%
  → 发现 manifest 用 x1/y1 而非 quad_1x/quad_1y
  → Skill patch：注明训练加载器具体在哪一行代码读什么字段名
  → 新增：manifest 字段验收清单
```

### 8.3 Skill 结构最佳实践（从项目中总结）

经过 12 个 Skill 的编写实践，形成了 Skill 模板：

```yaml
---
name: skill-name
description: 一句话说明适用范围
version: x.y.z
author: Hermes Agent
---

# 正文结构

## 1. 时机判断（When To Use）
清晰定义什么场景应该/不应该用这个 Skill

## 2. 快速参考（Quick Reference）
关键命令、参数、文件路径

## 3. 核心流程（逐步骤）
每个步骤有：命令 + 预期输出 + 检查点

## 4. 已知陷阱（Pitfalls）
每个陷阱有：症状 → 根因 → 修复 → 诊断命令

## 5. 参考文献
外部链接、reports、相关 Skill
```

---

## 9. 工作流全景：一次典型实验的生命周期

```
Step 1 ─── 实验设计 ─────────────────
  用户："跑一个 E20A cluster2 北京前缀对比实验"
  │ Agent 加载技能：lprnet-training-evaluation
  │                 + lprnet-cluster-analysis
  │                 + lprnet-data-generation
  │                 + lprnet-firstchar-province
  │ 跨会话检索：session_search("cluster2 京AD06088")
  │ 内存读取：实验必须用户确认后才启动
  │ Agent 输出实验计划：参数、数据配比、评测方案
  ▼

Step 2 ─── 用户审批 ─────────────────
  用户："可以开始"
  ▼

Step 3 ─── 数据准备 ─────────────────
  Agent 检查：manifest 编码（BOM？）、quad 字段名（quad_1x？）
  │ → 如果已确认的数据不变，跳过；如果有新数据，出 QA 图
  │ → 等待用户确认 QA
  ▼

Step 4 ─── 训练启动 ─────────────────
  Agent 执行 GPU 检查清单
  │ → 验证 --cuda True → nvidia-smi → train.log
  │ → 托管后台：background=True, notify_on_complete=True
  ▼

Step 5 ─── 等待完成 ─────────────────
  用户退出会话
  ▼

Step 6 ─── 训练完成（自动）──────────
  Agent 被唤醒 → 评测套件自动执行
  │ → 6 个 benchmark 全部跑完
  │ → 汇总结果表格
  │ → 通知用户
  ▼

Step 7 ─── 结果分析 ─────────────────
  用户回来查看报告
  │ Agent：输出人可读的 Markdown 表格 + 文字分析
  │ 如果需要：运行 trajectory fusion 分析失败帧
  │ 如果需要：生成 confusion matrix 分析省份混淆
  ▼

Step 8 ─── 知识沉淀 ─────────────────
  如果途中有新发现（新的坑、新的最佳实践）：
  │ → skill_manage(action='patch') 更新相关 Skill
  │ → mem0_conclude 记录用户新的偏好
  │ → session_search 记录在案供后续召回
  │
  如果实验结果是阴性的（如 B2-C_OBB）：
  │ → 记录入 memory：该方向已排除，不再重复
```

---

## 总结：Hermes Agent 在项目中的核心价值

| 传统工作方式 | 使用 Hermes Agent 后 |
|-------------|-------------------|
| 实验记录靠人脑/笔记本 | 所有历史通过 session_search 即时召回 |
| 每次 debug 的结论下次可能忘 | 沉淀为 Skill → 每次自动加载 |
| 用户偏好需要反复告知 | mem0 持久化，自动遵守 |
| 训练时需要守在终端前 | 后台托管 + 完成后自动通知 |
| 数据生成后手工检查 | Agent 自动出 QA contact sheet |
| 实验结论靠人脑梳理 | Agent 自动跑评测 + 汇总 |
| 新手接手需要手把手教 | 加载 Skill → Agent 精确执行 |

**最终效果**：用户不再需要告诉 Agent 重复的事情（不要 JSON、先出 QA、CCPD 四点顺序、manifest 字段名），Agent 不再需要用户解释历史实验（自动跨会话召回），项目知识不再流失（每次发现都记入 Skill）。这使得用户可以将精力集中在**实验设计决策**上，而不是过程执行和记忆维护上。
