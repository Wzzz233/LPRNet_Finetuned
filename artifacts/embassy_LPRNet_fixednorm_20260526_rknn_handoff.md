# Embassy RKNN 转换交接包 — 2026-05-27

## 目标

记录已通过 PyTorch/ONNX 验收并在本机 WSL2 RKNN 环境完成转换的 embassy 专家模型。
本机使用 `/root/miniconda3/envs/rknn_env` 中的 rknn-toolkit2 2.3.2 完成 RKNN fp16 转换。

## 输入文件

| 文件 | 路径 | SHA256 |
|------|------|--------|
| **ONNX (推荐)** | `artifacts/embassy_LPRNet_fixednorm_20260526_op11.onnx` | `59941f49930d7005cbb13a0c696ac2566d874ba6f1508f6964f8a70dfc2cd52c` |
| ONNX (备选) | `artifacts/embassy_LPRNet_fixednorm_20260526.onnx` | `d18a53a47bfabdc1fc3d86435cedc84c2a4b3c7fe0074b2a5e4124d2ba7683b5` |
| Keys | `keys/embassy_keys.txt` | `e781ae8cc97567d61bb69cb33392397b035154c397e985615a0b55ea44da5c6a` |
| PyTorch checkpoint | `experiments/embassy_formal_fixednorm_20260526/best_LPRNet_model.pth` | `b30d109710578add20844ad9d747a0b627abdb6d42f8cc846516b5a7609acabe` |
| Export report | `artifacts/embassy_LPRNet_fixednorm_20260526_export_report.txt` | — |
| Val manifest | `manifests_rebased/special_split_20260526/val_embassy_only.csv` | 300 samples |

## 验收基准

| 指标 | 值 | 说明 |
|------|:---:|------|
| PyTorch fixednorm eval | **271/300 = 90.33%** | per-sample normalization, single-image |
| Batch consistency (1 vs 300) | **300/300 preds agree** | batch-invariance verified |
| Illegal chars | **0** | no foreign characters in output |
| ONNX/PyTorch decode | **300/300 match** | both opset 11 and opset 18 |
| Max numerical diff | 2.29e-04 | documented, does NOT affect CTC greedy decode |

## Keys 信息

- 文件内容：`0-9 + 使`（11 chars + CTC blank）
- class_num = 12
- 输出格式：`使 + 6位数字`（如 `使693015`）
- blank_idx = 11 (class_num - 1, '-' appended at runtime)

## ONNX 信息

### op11 (推荐)

- 大小：907K
- 导出方式：legacy TorchScript path (`torch.onnx.export(..., dynamo=False)`)
- 理由：更小的 opset 版本，RKNN toolkit 兼容性更广
- 已验证：300/300 decode match with PyTorch

### op18 (备选)

- 大小：84K
- 导出方式：new `torch.export` path (PyTorch 2.11 default)
- 理由：如果 op11 无法通过 RKNN 转换，尝试此版本
- 已验证：300/300 decode match with PyTorch

## RKNN 转换参数

**状态：已完成 (2026-05-27)**

| 参数 | 值 |
|------|-----|
| Toolkit | rknn-toolkit2 2.3.2 |
| Environment | /root/miniconda3/envs/rknn_env (WSL2) |
| target_platform | rk3568 |
| do_quantization | False (fp16) |
| mean_values | [[0,0,0]] |
| std_values | [[1,1,1]] |

### 产物

| RKNN | 路径 | SHA256 | 大小 | 源 ONNX |
|------|------|--------|:--:|------|
| Primary | `artifacts/embassy_LPRNet_fixednorm_20260526_fp16.rknn` | `4159d2ede625426cdc09df98be16584366681b7d400b23d92240854be2d9cce9` | 745 KB | op11 |
| Fallback | `artifacts/embassy_LPRNet_fixednorm_20260526_fp16_op18.rknn` | `91dfb7a6020249f36023c1364a14d4489681bde0fd82dcc0cebda1129909f3ee` | 742 KB | op18 |

### 构建日志摘要

- Load ONNX: OK (<0.1s)
- OpFusing: completed
- Build: return 0 (success)
- Warnings: 5x "Unkown op target: 0" (both variants, non-blocking, common with rknn-toolkit2)
- Export: OK

### 注意事项

1. **不要使用 batch-dependent normalization**：ONNX 内部已使用 per-sample normalization。
2. **不要添加 input pre-processing**：ONNX 期望输入为 `[-1, 1]` 范围的 float32 tensor（BGR 通道顺序，24x94 分辨率）。
3. 如果 RKNN toolkit 要求 uint8 输入，需要另做预处理适配（当前 ONNX 不包含 uint8→float 的转换）。

## Simulator 状态

**未运行** — `init_runtime(target='rk3568')` 需要 ADB 连接的 RK3568 板端。
本机无板端连接。板端 decode verification 仍需在连接板端后进行。

## 期望输出（已生成）

```
artifacts/embassy_LPRNet_fixednorm_20260526_fp16.rknn
```

## 转换后验收

### 步骤 1：记录转换信息 ✅

- RKNN toolkit 版本: 2.3.2
- 转换日志摘要: 见上方 "构建日志摘要"
- RKNN 文件 sha256: 见上方产物表
- 转换是否成功: ✅ build=0, export=0

### 步骤 2：Simulator decode check ⏸️

使用 `manifests_rebased/special_split_20260526/val_embassy_only.csv` 前 50 张做 simulator 推理：

- 对比 RKNN simulator vs PyTorch 输出
- 必须 50/50 decode match
- 记录 numerical max diff

### 步骤 3：板端接入

- Embassy 模型 **不能**替换现有 `--ocr-special-model`
- 当前 ARM 只有 `--ocr-special-model` 单一 special 模型槽位
- Embassy RKNN 需要等待 ARM UNKNOWN 二级路由实现后才能接入
- 接入方式：板端颜色分类为 UNKNOWN 后，二级路由判断车型/号牌格式，分派给 embassy 模型

## Caveats

- Val 仅 300 张合成数据，板端增量样本复核仍需要
- ARM 当前不支持 police/embassy 双专家，需等路由改造
- 本交接不包含板端 C++ 接入代码变更
- 若 RKNN 转换失败，优先排查 op11/op18 兼容性，其次是 per-sample normalization 在 RKNN 中的等价性

## 相关文档

- 拆分方案：[SPECIAL_POLICE_EMBASSY_SPLIT_20260526.md](../docs/SPECIAL_POLICE_EMBASSY_SPLIT_20260526.md)
- 导出报告：[embassy_LPRNet_fixednorm_20260526_export_report.txt](embassy_LPRNet_fixednorm_20260526_export_report.txt)
- 验收报告：[final_acceptance_report.txt](../experiments/embassy_formal_fixednorm_20260526/final_acceptance_report.txt)
